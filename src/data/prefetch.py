"""
src/data/prefetch.py
─────────────────────
Athanor Alpha — DataPrefetcher singleton.

Fetches ALL market data for ALL tickers in a single batch cycle.
No analyst agent ever calls yfinance directly — they all read from
the payload this module writes into AgentState.

Data fetched per ticker
-----------------------
  yfinance:
    • OHLCV daily          — 1 year history
    • OHLCV intraday 5m    — last 5 trading days
    • info                 — market cap, P/E, sector, etc.
    • financials           — income stmt, balance sheet, cash flow (annual + quarterly)
    • institutional holders

Output shape
------------
  Dict[ticker, TickerPayload] where TickerPayload is a plain dict:

  {
    "ohlcv_daily":    pd.DataFrame  (index=Date, cols=Open/High/Low/Close/Volume)
    "ohlcv_weekly":   pd.DataFrame  (index=Date, weekly candles — 2 years)
    "ohlcv_4h":       pd.DataFrame  (index=Datetime, 4h candles — 60 days)
    "ohlcv_5m":       pd.DataFrame  (index=Datetime)
    "info":           dict
    "income_stmt":    pd.DataFrame
    "income_stmt_q":  pd.DataFrame
    "balance_sheet":  pd.DataFrame
    "balance_sheet_q":pd.DataFrame
    "cash_flow":      pd.DataFrame
    "cash_flow_q":    pd.DataFrame
    "holders":        pd.DataFrame  (institutional holders)
    "fetched_at":     str           (ISO-8601 UTC)
  }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

import yfinance as yf

from src.data.ttl_cache import TTL_FINANCIALS, TTL_PRICES, get_ttl_cache

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────

TickerPayload = dict[str, Any]
PrefetchResult = dict[str, TickerPayload]


# ─────────────────────────────────────────────
# Singleton DataPrefetcher
# ─────────────────────────────────────────────

class DataPrefetcher:
    """
    Singleton batch fetcher for market data.

    Usage
    -----
    prefetcher = DataPrefetcher.get_instance()
    data = prefetcher.fetch_all(["AAPL", "NVDA", ...])
    """

    _instance: Optional["DataPrefetcher"] = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._cache = get_ttl_cache()

    # ── singleton constructor ──────────────────

    @classmethod
    def get_instance(cls) -> "DataPrefetcher":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.debug("DataPrefetcher singleton created")
        return cls._instance

    # ── public API ────────────────────────────

    def fetch_all(self, tickers: list[str]) -> PrefetchResult:
        """
        Fetch all data for *tickers* in a single batch cycle.

        Uses the TTL cache — if data is fresh, returns cached payload
        without hitting the network.

        Returns
        -------
        Dict[ticker -> TickerPayload]
        """
        result: PrefetchResult = {}

        for ticker in tickers:
            cached = self._cache.get(f"{ticker}:payload")
            if cached is not None:
                logger.debug("%s — serving from cache", ticker)
                result[ticker] = cached
            else:
                logger.info("%s — fetching from yfinance …", ticker)
                payload = self._fetch_ticker(ticker)
                if payload:
                    # financials cached for 24h, prices for 1 min
                    # We cache the whole payload at the price TTL (shortest)
                    # Individual components use their own TTL key if needed
                    self._cache.set(f"{ticker}:payload", payload, ttl=TTL_PRICES)
                    result[ticker] = payload
                else:
                    logger.warning("%s — fetch returned empty payload, skipping", ticker)

        logger.info(
            "Prefetch complete: %d/%d tickers fetched successfully",
            len(result), len(tickers),
        )
        return result

    # ── private helpers ───────────────────────

    def _fetch_ticker(self, ticker: str) -> Optional[TickerPayload]:
        """Fetch all yfinance data for a single ticker. Returns None on hard failure."""
        try:
            t = yf.Ticker(ticker)
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # --- OHLCV daily (1 year) ---
            ohlcv_daily = self._safe_download(ticker, period="1y", interval="1d")

            # --- OHLCV weekly (2 years — trend di fondo) ---
            ohlcv_weekly = self._safe_download(ticker, period="2y", interval="1wk")

            # --- OHLCV 4h (60 days — entry timing) ---
            ohlcv_4h = self._safe_download(ticker, period="60d", interval="1h")

            # --- OHLCV intraday 5m (5 days) ---
            ohlcv_5m = self._safe_download(ticker, period="5d", interval="5m")

            # --- Company info ---
            info = self._safe_call(t.info, {})

            # --- Financial statements (annual) ---
            income_stmt    = self._safe_df(t.income_stmt)
            balance_sheet  = self._safe_df(t.balance_sheet)
            cash_flow      = self._safe_df(t.cash_flow)

            # --- Financial statements (quarterly) ---
            income_stmt_q   = self._safe_df(t.quarterly_income_stmt)
            balance_sheet_q = self._safe_df(t.quarterly_balance_sheet)
            cash_flow_q     = self._safe_df(t.quarterly_cash_flow)

            # --- Institutional holders ---
            holders = self._safe_df(t.institutional_holders)

            return {
                "ohlcv_daily":     ohlcv_daily,
                "ohlcv_weekly":    ohlcv_weekly,
                "ohlcv_4h":        ohlcv_4h,
                "ohlcv_5m":        ohlcv_5m,
                "info":            info,
                "income_stmt":     income_stmt,
                "income_stmt_q":   income_stmt_q,
                "balance_sheet":   balance_sheet,
                "balance_sheet_q": balance_sheet_q,
                "cash_flow":       cash_flow,
                "cash_flow_q":     cash_flow_q,
                "holders":         holders,
                "fetched_at":      now,
            }

        except Exception as exc:
            logger.error("%s — unexpected error during fetch: %s", ticker, exc, exc_info=True)
            return None

    @staticmethod
    def _safe_download(ticker: str, period: str, interval: str) -> Any:
        """Download OHLCV data, returning empty DataFrame on failure."""
        try:
            import pandas as pd
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            # yf.download returns multi-level columns when single ticker —
            # flatten if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as exc:
            logger.warning("%s — OHLCV %s download failed: %s", ticker, interval, exc)
            import pandas as pd
            return pd.DataFrame()

    @staticmethod
    def _safe_call(fn: Any, default: Any) -> Any:
        """Call *fn* (a property or callable), returning *default* on failure."""
        try:
            return fn() if callable(fn) else fn
        except Exception as exc:
            logger.warning("safe_call failed: %s", exc)
            return default

    @staticmethod
    def _safe_df(attr: Any) -> Any:
        """Return a DataFrame attribute, or empty DataFrame on failure."""
        try:
            import pandas as pd
            df = attr
            return df if df is not None else pd.DataFrame()
        except Exception as exc:
            logger.warning("safe_df failed: %s", exc)
            import pandas as pd
            return pd.DataFrame()


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    prefetcher = DataPrefetcher.get_instance()
    result = prefetcher.fetch_all(["AAPL", "MSFT"])

    for ticker, payload in result.items():
        ohlcv = payload.get("ohlcv_daily")
        info  = payload.get("info", {})
        print(f"\n{'─'*40}")
        print(f"  {ticker}")
        print(f"  OHLCV rows  : {len(ohlcv) if ohlcv is not None else 'N/A'}")
        print(f"  Last close  : {ohlcv['Close'].iloc[-1]:.2f}" if ohlcv is not None and len(ohlcv) > 0 else "  Last close  : N/A")
        print(f"  Sector      : {info.get('sector', 'N/A')}")
        print(f"  Market cap  : {info.get('marketCap', 'N/A')}")
        print(f"  Fetched at  : {payload.get('fetched_at')}")
