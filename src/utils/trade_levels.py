"""
src/utils/trade_levels.py
─────────────────────────
Utility condivisa per calcolare entry price, stop loss e take profit
basati su ATR(14) per ogni agente filosofico.

Chiamata da ogni agente dopo che l'LLM ha prodotto la direzione:
    levels = compute_trade_levels(direction, state, ticker)
    signal_dict.update(levels)
"""

from __future__ import annotations

from typing import Any

from src.indicators.technical_indicators import calculate_atr
from src.data.state_reader import get_ohlcv_daily


def compute_trade_levels(
    direction: str,
    state: dict[str, Any],
    ticker: str,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
) -> dict[str, float | None]:
    """
    Calcola entry price, stop loss e take profit basati su ATR(14).

    Parameters
    ----------
    direction : "LONG" | "SHORT" | "NEUTRAL"
    state     : AgentState LangGraph
    ticker    : simbolo azionario (es. "AAPL")
    sl_mult   : moltiplicatore ATR per lo stop loss (default 1.5×)
    tp_mult   : moltiplicatore ATR per il take profit (default 3.0×, R:R = 1:2)

    Returns
    -------
    dict con chiavi: entry_price, stop_loss, take_profit, atr
    Tutti None se direction == "NEUTRAL" o dati OHLCV non disponibili.
    """
    if direction == "NEUTRAL":
        return {"entry_price": None, "stop_loss": None, "take_profit": None, "atr": None}

    ohlcv = get_ohlcv_daily(state, ticker)
    if ohlcv is None or ohlcv.empty:
        return {"entry_price": None, "stop_loss": None, "take_profit": None, "atr": None}

    # Normalizza nomi colonne (yfinance può restituire 'Close' o 'close')
    df = ohlcv.rename(columns=str.lower)

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df.columns):
        return {"entry_price": None, "stop_loss": None, "take_profit": None, "atr": None}

    try:
        entry = float(df["close"].iloc[-1])
        atr_series = calculate_atr(df, period=14)
        atr = float(atr_series.iloc[-1])
    except Exception:
        return {"entry_price": None, "stop_loss": None, "take_profit": None, "atr": None}

    if direction == "LONG":
        sl = entry - sl_mult * atr
        tp = entry + tp_mult * atr
    else:  # SHORT
        sl = entry + sl_mult * atr
        tp = entry - tp_mult * atr

    return {
        "entry_price": round(entry, 4),
        "stop_loss":   round(sl, 4),
        "take_profit": round(tp, 4),
        "atr":         round(atr, 4),
    }
