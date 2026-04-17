"""
src/data/macro_fetcher.py
─────────────────────────
Athanor Alpha — MacroFetcher (FIX C2).

Fetches global macro indicators from yfinance (no external APIs).
Called once per pipeline run from data_prefetch_agent.
Data is NOT per-ticker — it is global market context.

Tickers fetched:
  ^VIX  — CBOE Volatility Index
  ^TNX  — 10-Year Treasury Yield (%)
  ^IRX  — 13-Week T-Bill Rate / 3M proxy (%)
  ^FVX  — 5-Year Treasury Yield (%)

Output shape (written to state["data"]["macro_data"]):
  {
    "vix":                float | None   — current VIX level
    "yield_10y":          float | None   — 10Y yield (%)
    "yield_3m":           float | None   — 3M yield (%)
    "yield_5y":           float | None   — 5Y yield (%)
    "yield_spread_10y3m": float | None   — 10Y - 3M (pp); negative = inverted curve
    "rate_trend_20d":     float | None   — 20-day change in 10Y yield (pp)
    "vix_trend_10d":      float | None   — 10-day change in VIX
    "fetched_at":         str            — ISO-8601 UTC
  }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MacroData = dict[str, Any]


class MacroFetcher:
    """
    Stateless global macro data fetcher.

    Usage
    -----
    macro_data = MacroFetcher.fetch()
    """

    @classmethod
    def fetch(cls) -> MacroData:
        """
        Fetch VIX + yield curve data from yfinance.

        Returns a dict with all macro indicators.
        Returns a dict with None values on hard failure so the
        downstream macro_agent degrades gracefully to NEUTRAL.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        vix        = cls._fetch_last("^VIX",  period="15d")
        yield_10y  = cls._fetch_last("^TNX",  period="30d")
        yield_3m   = cls._fetch_last("^IRX",  period="30d")
        yield_5y   = cls._fetch_last("^FVX",  period="30d")

        # Yield spread: 10Y - 3M in percentage points
        # Positive = normal/steep curve (growth expected)
        # Negative = inverted curve (recession risk)
        yield_spread_10y3m: Optional[float] = None
        if yield_10y is not None and yield_3m is not None:
            yield_spread_10y3m = round(yield_10y - yield_3m, 3)

        # 20-day rate trend: measures if rates are rising or falling
        rate_trend_20d = cls._fetch_trend("^TNX", period="35d", lookback=20)

        # 10-day VIX trend: measures if fear is rising or falling
        vix_trend_10d = cls._fetch_trend("^VIX", period="20d", lookback=10)

        result: MacroData = {
            "vix":                vix,
            "yield_10y":          yield_10y,
            "yield_3m":           yield_3m,
            "yield_5y":           yield_5y,
            "yield_spread_10y3m": yield_spread_10y3m,
            "rate_trend_20d":     rate_trend_20d,
            "vix_trend_10d":      vix_trend_10d,
            "fetched_at":         now,
        }

        logger.info(
            "MacroFetcher: VIX=%.1f, 10Y=%.2f%%, 3M=%.2f%%, spread=%.3fpp, "
            "rate_trend_20d=%.3fpp, vix_trend_10d=%.2f",
            vix or 0.0,
            yield_10y or 0.0,
            yield_3m or 0.0,
            yield_spread_10y3m or 0.0,
            rate_trend_20d or 0.0,
            vix_trend_10d or 0.0,
        )

        return result

    # ── private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _fetch_last(symbol: str, period: str = "10d") -> Optional[float]:
        """Return the most recent closing value for *symbol*. None on failure."""
        try:
            df = yf.download(
                symbol,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            if close.empty:
                return None
            return round(float(close.iloc[-1]), 3)
        except Exception as exc:
            logger.warning("MacroFetcher._fetch_last(%s): %s", symbol, exc)
            return None

    @staticmethod
    def _fetch_trend(
        symbol: str,
        period: str = "30d",
        lookback: int = 20,
    ) -> Optional[float]:
        """
        Return close_now - close_{lookback}_trading_days_ago for *symbol*.
        Positive = value rising, negative = value falling.
        None on failure or insufficient data.
        """
        try:
            df = yf.download(
                symbol,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            if len(close) < 2:
                return None
            n = min(lookback, len(close) - 1)
            trend = float(close.iloc[-1]) - float(close.iloc[-(n + 1)])
            return round(trend, 3)
        except Exception as exc:
            logger.warning("MacroFetcher._fetch_trend(%s): %s", symbol, exc)
            return None
