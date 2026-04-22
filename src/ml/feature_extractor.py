"""
feature_extractor.py — Fase 7
Flat feature dict from one (agent, ticker, signal, context) tuple.
Used by both dataset_builder (batch) and meta_learner (inference).
"""
from __future__ import annotations

SECTOR_MAP: dict[str, str] = {
    "AAPL":    "TECHNOLOGY",
    "MSFT":    "TECHNOLOGY",
    "NVDA":    "TECHNOLOGY",
    "GOOGL":   "TECHNOLOGY",
    "AMZN":    "TECHNOLOGY",
    "TSLA":    "CONSUMER_DISCRETIONARY",
    "MSTR":    "TECHNOLOGY",
    "COIN":    "FINANCIALS",
    "SMCI":    "TECHNOLOGY",
    "MELI":    "CONSUMER_DISCRETIONARY",
    "BTC-USD": "CRYPTO",
    "ETH-USD": "CRYPTO",
    "SOL-USD": "CRYPTO",
}

# Columns that will be one-hot encoded (must stay consistent train ↔ inference)
CATEGORICAL_COLS: list[str] = [
    "agent_id", "ticker", "signal", "horizon", "regime", "sector",
]


def extract_features(
    agent_id: str,
    ticker: str,
    signal: str,
    confidence: float,
    horizon: str,
    context: dict,
) -> dict:
    """
    Build a flat feature dict for one prediction.

    context keys (all optional — sensible defaults applied):
      regime           str   RISK_ON | CAUTION | RISK_OFF | UNKNOWN
      vix_at_prediction float
      realized_vol_20d  float
      sector           str   overrides SECTOR_MAP if provided
      month            int   1-12
      day_of_week      int   0-6 (Monday=0)
      ewa_weight       float current EWA weight for this (agent, ticker)
    """
    return {
        "agent_id":          agent_id,
        "ticker":            ticker,
        "signal":            signal.upper(),
        "confidence":        float(confidence),
        "horizon":           horizon,
        "regime":            str(context.get("regime") or "UNKNOWN"),
        "vix_at_prediction": float(context.get("vix_at_prediction") or 20.0),
        "realized_vol_20d":  float(context.get("realized_vol_20d") or 0.0),
        "sector":            str(context.get("sector") or SECTOR_MAP.get(ticker, "UNKNOWN")),
        "month":             int(context.get("month") or 1),
        "day_of_week":       int(context.get("day_of_week") or 0),
        "ewa_weight":        float(context.get("ewa_weight") or 1.0),
    }
