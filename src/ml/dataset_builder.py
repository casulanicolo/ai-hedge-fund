"""
dataset_builder.py — Fase 7
Builds ML dataset from predictions + outcomes in DB.

Adds context columns (VIX, realized_vol, regime) via yfinance historical lookups.
Saves to cache/ml/dataset_YYYYMMDD.parquet.

Run standalone:
    python -m src.ml.dataset_builder
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.db.init_db import get_connection
from src.ml.feature_extractor import SECTOR_MAP, extract_features

logger = logging.getLogger(__name__)

CACHE_DIR  = Path("cache/ml")
VIX_TICKER = "^VIX"

# Leakage guard: skip predictions with outcomes fresher than this many days
LEAKAGE_CUTOFF_DAYS = 21


# ── VIX + regime helpers ──────────────────────────────────────────────────────

def _fetch_vix_history(start: date, end: date) -> dict[str, float]:
    """Returns {date_str: vix_close} for the given range."""
    try:
        df = yf.download(
            VIX_TICKER,
            start=str(start),
            end=str(end + timedelta(days=1)),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        return {str(idx.date()): float(v) for idx, v in closes.items()}
    except Exception as exc:
        logger.warning("VIX history fetch failed: %s", exc)
        return {}


def _vix_to_regime(vix: float | None) -> str:
    if vix is None:
        return "UNKNOWN"
    if vix < 20:
        return "RISK_ON"
    if vix <= 30:
        return "CAUTION"
    return "RISK_OFF"


# ── Realized volatility ───────────────────────────────────────────────────────

def _fetch_realized_vol(
    tickers: list[str], start: date, end: date
) -> dict[tuple[str, str], float]:
    """Returns {(ticker, date_str): annualized_20d_realized_vol}."""
    result: dict[tuple[str, str], float] = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=str(start - timedelta(days=40)),
                end=str(end + timedelta(days=1)),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes  = df["Close"].dropna()
            returns = closes.pct_change()
            rolling = returns.rolling(20).std() * (252 ** 0.5)
            for idx, v in rolling.dropna().items():
                result[(ticker, str(idx.date()))] = round(float(v), 4)
        except Exception as exc:
            logger.warning("Realized vol fetch failed for %s: %s", ticker, exc)
    return result


# ── Core builder ──────────────────────────────────────────────────────────────

def build_dataset(cutoff_days: int = LEAKAGE_CUTOFF_DAYS) -> pd.DataFrame:
    """
    Query all predictions with outcomes older than cutoff_days.
    Enriches with VIX, realized vol, regime, sector.

    y_binary:     1 if signal directionally correct, 0 otherwise
    y_continuous: signal_signed * actual_return  (primary regression target)
    Window preference: 5d > 1d > 20d.
    """
    conn = get_connection()
    query = """
        SELECT
            p.id            AS prediction_id,
            p.agent_id,
            p.ticker,
            p.signal,
            p.confidence,
            p.timestamp,
            o.window,
            o.actual_return_1d,
            o.actual_return_5d,
            o.actual_return_20d,
            aw.weight       AS ewa_weight
        FROM predictions p
        JOIN outcomes o     ON o.prediction_id = p.id
        LEFT JOIN agent_weights aw
               ON aw.agent_id = p.agent_id AND aw.ticker = p.ticker
        WHERE p.timestamp < date('now', ?)
          AND p.signal IN ('BUY', 'SELL', 'HOLD')
    """
    rows = conn.execute(query, (f"-{cutoff_days} days",)).fetchall()
    conn.close()

    if not rows:
        logger.warning("No predictions+outcomes rows found in DB.")
        return pd.DataFrame()

    logger.info("Loaded %d prediction+outcome rows from DB.", len(rows))

    # Collect date range + tickers for bulk fetches
    dates: list[date] = []
    tickers: set[str] = set()
    for r in rows:
        try:
            dt = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
            dates.append(dt.date())
            tickers.add(r["ticker"])
        except Exception:
            pass

    if not dates:
        return pd.DataFrame()

    min_date, max_date = min(dates), max(dates)
    logger.info("Fetching VIX history %s → %s …", min_date, max_date)
    vix_hist = _fetch_vix_history(min_date, max_date)
    logger.info("Fetching realized vol for %d tickers …", len(tickers))
    rvol_hist = _fetch_realized_vol(list(tickers), min_date, max_date)

    records: list[dict] = []
    for r in rows:
        try:
            dt = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
        except Exception:
            continue

        date_str = str(dt.date())
        vix_val  = vix_hist.get(date_str)
        rvol_val = rvol_hist.get((r["ticker"], date_str), 0.0)
        regime   = _vix_to_regime(vix_val)

        context = {
            "regime":            regime,
            "vix_at_prediction": vix_val or 20.0,
            "realized_vol_20d":  rvol_val,
            "sector":            SECTOR_MAP.get(r["ticker"], "UNKNOWN"),
            "month":             dt.month,
            "day_of_week":       dt.weekday(),
            "ewa_weight":        float(r["ewa_weight"]) if r["ewa_weight"] is not None else 1.0,
        }

        feat = extract_features(
            agent_id=r["agent_id"],
            ticker=r["ticker"],
            signal=r["signal"],
            confidence=float(r["confidence"]),
            horizon=r["window"],
            context=context,
        )

        # Pick actual_return: prefer 5d, then 1d, then 20d
        actual_return = (
            r["actual_return_5d"]  if r["actual_return_5d"]  is not None else
            r["actual_return_1d"]  if r["actual_return_1d"]  is not None else
            r["actual_return_20d"]
        )
        if actual_return is None:
            continue

        sig = r["signal"].upper()
        sig_signed  = 1 if sig == "BUY" else (-1 if sig == "SELL" else 0)
        y_continuous = sig_signed * float(actual_return)
        y_binary     = int(
            (sig == "BUY"  and actual_return > 0) or
            (sig == "SELL" and actual_return < 0)
        )

        feat["y_binary"]       = y_binary
        feat["y_continuous"]   = y_continuous
        feat["prediction_date"] = date_str
        feat["prediction_id"]  = r["prediction_id"]
        records.append(feat)

    if not records:
        logger.warning("No usable rows after processing.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(
        "Dataset ready: %d rows, %d columns. "
        "y_binary mean=%.3f, y_continuous mean=%.4f",
        len(df), len(df.columns),
        df["y_binary"].mean(), df["y_continuous"].mean(),
    )
    _log_distributions(df)
    return df


def _log_distributions(df: pd.DataFrame) -> None:
    logger.info("\nAgent distribution:\n%s", df["agent_id"].value_counts().to_string())
    logger.info("\nRegime distribution:\n%s", df["regime"].value_counts().to_string())
    logger.info("\nTicker distribution:\n%s", df["ticker"].value_counts().to_string())


def build_and_save() -> Path:
    """Build dataset and save to parquet. Returns output path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today    = datetime.now().strftime("%Y%m%d")
    out_path = CACHE_DIR / f"dataset_{today}.parquet"
    df = build_dataset()
    if df.empty:
        logger.warning("Empty dataset — parquet not written.")
        return out_path
    df.to_parquet(out_path, index=False)
    logger.info("Dataset saved → %s (%d rows)", out_path, len(df))
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_and_save()
