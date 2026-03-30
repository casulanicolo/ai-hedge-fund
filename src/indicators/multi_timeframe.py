"""
src/indicators/multi_timeframe.py
Estrae i DataFrame daily e intraday 5m dal prefetched_data
e li passa a compute_indicator_snapshot per l'analisi multi-timeframe.
"""

import pandas as pd
from src.indicators.technical_indicators import compute_indicator_snapshot


def _to_df(records: list[dict] | None) -> pd.DataFrame | None:
    """
    Converte una lista di dizionari OHLCV in un DataFrame pandas pulito.
    Restituisce None se i dati sono assenti o insufficienti.
    """
    if not records or len(records) < 10:
        return None

    df = pd.DataFrame(records)

    # Normalizza i nomi delle colonne (minuscolo)
    df.columns = [c.lower() for c in df.columns]

    # Colonne obbligatorie
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    # Converte in float
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rimuove righe con NaN nelle colonne critiche
    df = df.dropna(subset=list(required))

    if len(df) < 10:
        return None

    df = df.reset_index(drop=True)
    return df


def build_snapshot_for_ticker(ticker_payload: dict) -> dict:
    """
    Dato il payload prefetchato per un singolo ticker,
    estrae daily e intraday 5m e calcola lo snapshot degli indicatori.

    Args:
        ticker_payload: il dizionario prefetched_data[ticker]
                        (prodotto da DataPrefetcher)

    Returns:
        dict con tutti gli indicatori scalari pronti per l'LLM.
        Dizionario vuoto se i dati sono insufficienti.
    """
    if not ticker_payload:
        return {}

    # --- Daily OHLCV ---
    # Il prefetcher può salvare i dati daily in chiavi diverse a seconda
    # dell'implementazione; proviamo le più comuni.
    daily_records = (
        ticker_payload.get("ohlcv_daily")
        or ticker_payload.get("prices")
        or ticker_payload.get("daily")
        or []
    )
    daily_df = _to_df(daily_records)

    # --- Intraday 5m OHLCV ---
    intraday_records = (
        ticker_payload.get("ohlcv_5m")
        or ticker_payload.get("intraday_5m")
        or ticker_payload.get("intraday")
        or []
    )
    intraday_df = _to_df(intraday_records)

    # --- Calcolo snapshot ---
    snapshot = compute_indicator_snapshot(daily_df, intraday_df)

    # Aggiungi meta-info sul timeframe disponibile
    snapshot["has_intraday"] = intraday_df is not None
    snapshot["daily_bars"]   = len(daily_df) if daily_df is not None else 0

    return snapshot
