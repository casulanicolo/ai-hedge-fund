"""
src/monitor/price_checker.py
──────────────────────────────
Simplified: fetch OHLCV 5min bars and daily ATR for exit rules.
Old alert logic removed — now owned by exit_rules.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_ohlcv_5min(ticker: str, periods: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch the last `periods` 5-minute OHLCV bars for `ticker`.
    Returns DataFrame with lowercase columns (open, high, low, close, volume)
    or None on failure.
    """
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1d", interval="5m")
        if hist.empty:
            logger.warning("[price_checker] No 5min data for %s", ticker)
            return None
        df = hist.rename(columns=str.lower)
        return df.tail(periods).reset_index(drop=True)
    except Exception as exc:
        logger.error("[price_checker] get_ohlcv_5min %s failed: %s", ticker, exc)
        return None


def get_daily_ohlcv(ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch `days` of daily OHLCV bars.
    Returns DataFrame with lowercase columns or None on failure.
    """
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period=f"{days}d", interval="1d")
        if hist.empty:
            return None
        return hist.rename(columns=str.lower).reset_index(drop=True)
    except Exception as exc:
        logger.error("[price_checker] get_daily_ohlcv %s failed: %s", ticker, exc)
        return None


def get_daily_atr(ticker: str, period: int = 14) -> float:
    """
    Compute ATR(period) from daily bars.
    Returns 0.0 on failure (callers must guard against this).
    """
    from src.indicators.technical_indicators import calculate_atr

    df = get_daily_ohlcv(ticker, days=period + 10)
    if df is None or df.empty:
        return 0.0
    try:
        atr_series = calculate_atr(df, period=period)
        val = float(atr_series.iloc[-1])
        return val if val > 0 else 0.0
    except Exception as exc:
        logger.error("[price_checker] ATR computation for %s failed: %s", ticker, exc)
        return 0.0


def get_avg_daily_volume(ticker: str, days: int = 20) -> Optional[float]:
    """Return average daily volume over the last `days` trading sessions."""
    df = get_daily_ohlcv(ticker, days=days + 5)
    if df is None or "volume" not in df.columns or len(df) < 5:
        return None
    return float(df["volume"].tail(days).mean())


def get_current_price(ticker: str) -> Optional[float]:
    """Return the latest available price via yfinance fast_info."""
    try:
        tk = yf.Ticker(ticker)
        price = getattr(tk.fast_info, "last_price", None)
        if price is not None and price > 0:
            return float(price)
        # Fallback: last close from 5min
        df = get_ohlcv_5min(ticker, periods=1)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None
    except Exception as exc:
        logger.error("[price_checker] get_current_price %s failed: %s", ticker, exc)
        return None
