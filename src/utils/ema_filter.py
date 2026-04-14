"""
src/utils/ema_filter.py
EMA Trend Filter — Athanor Alpha

Fix 5 (2026-04-14):
  Filtro EMA obbligatorio per tutti gli agenti fondamentali.
  Forza l'allineamento del segnale fondamentale al trend tecnico di breve
  per correggere il WR_5d sistematicamente < 50% causato dall'incompatibilita
  tra orizzonte fondamentale (mesi) e swing 3-4gg.

Logica:
  - EMA8 > EMA21 (trend UP)   → LONG passa, SHORT diventa NEUTRAL
  - EMA8 < EMA21 (trend DOWN) → SHORT passa, LONG diventa NEUTRAL
  - NEUTRAL non viene mai modificato
  - Se dati OHLCV non disponibili → direzione originale non modificata (fail-safe)

Uso:
  from src.utils.ema_filter import apply_ema_filter
  filtered_direction = apply_ema_filter(raw_direction, state, ticker)
"""

from __future__ import annotations

import pandas as pd


def _get_ema_trend(state: dict, ticker: str) -> str:
    """
    Calcola il trend EMA8 vs EMA21 sui dati OHLCV daily del ticker.
    Legge da prefetched_data (gia disponibili in state).

    Returns:
        "UP"   — EMA8 > EMA21
        "DOWN" — EMA8 < EMA21
        "FLAT" — dati insufficienti o errore
    """
    try:
        prefetched = state.get("data", {}).get("prefetched_data", {})
        ticker_data = prefetched.get(ticker, {})

        # Prova ohlcv_daily prima, poi ohlcv come fallback
        df = ticker_data.get("ohlcv_daily") or ticker_data.get("ohlcv")

        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 22:
            return "FLAT"

        # Flatten multi-level columns se presenti
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            return "FLAT"

        close = df["Close"].astype(float).dropna()
        if len(close) < 22:
            return "FLAT"

        ema8  = close.ewm(span=8,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()

        last_ema8  = float(ema8.iloc[-1])
        last_ema21 = float(ema21.iloc[-1])

        if last_ema8 > last_ema21:
            return "UP"
        elif last_ema8 < last_ema21:
            return "DOWN"
        return "FLAT"

    except Exception:
        return "FLAT"


def apply_ema_filter(direction: str, state: dict, ticker: str) -> str:
    """
    Applica il filtro EMA alla direzione grezza prodotta dall'agente fondamentale.

    Args:
        direction: "LONG" | "SHORT" | "NEUTRAL"
        state:     AgentState LangGraph (contiene prefetched_data)
        ticker:    simbolo del ticker

    Returns:
        Direzione filtrata: "LONG" | "SHORT" | "NEUTRAL"

    Regole:
        NEUTRAL  → NEUTRAL (invariato)
        LONG  + trend UP   → LONG   (confermato)
        LONG  + trend DOWN → NEUTRAL (filtrato: trend contrario)
        LONG  + trend FLAT → LONG   (fail-safe: dati insufficienti, non bloccare)
        SHORT + trend DOWN → SHORT  (confermato)
        SHORT + trend UP   → NEUTRAL (filtrato: trend contrario)
        SHORT + trend FLAT → SHORT  (fail-safe)
    """
    if direction == "NEUTRAL":
        return "NEUTRAL"

    trend = _get_ema_trend(state, ticker)

    if trend == "FLAT":
        # Dati non disponibili: non modificare il segnale originale
        return direction

    if direction == "LONG":
        return "LONG" if trend == "UP" else "NEUTRAL"

    if direction == "SHORT":
        return "SHORT" if trend == "DOWN" else "NEUTRAL"

    # Fallback (direzione sconosciuta)
    return direction
