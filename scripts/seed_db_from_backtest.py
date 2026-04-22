"""
scripts/seed_db_from_backtest.py — Fase 7 prerequisite

The walk-forward backtest (Fase 5) blocked real LLM calls and returned
deterministic seeded signals via _backtest_seeded_default() in call_llm.
With skip_prediction_log=True those signals were never persisted to the DB.

This script:
  1. Reads all 1831 trading dates from cache/backtest_states/ filenames.
  2. Reproduces the EXACT same seeded signal for every (date, ticker, agent)
     using the same SHA-256 formula (must stay in sync with llm.py).
  3. Bulk-inserts into predictions (dedup via reasoning_hash).
  4. Downloads price history per ticker from yfinance (one call per ticker).
  5. Computes forward returns (1d / 5d / 20d) vectorially with pandas.
  6. Bulk-inserts into outcomes (3 rows per prediction).

Run:
    python scripts/seed_db_from_backtest.py            # full load
    python scripts/seed_db_from_backtest.py --limit 10  # first 10 days (test)
    python scripts/seed_db_from_backtest.py --dry-run   # count only, no writes
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.init_db import get_connection, init_db

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

STATES_DIR = Path("cache/backtest_states")

# Must stay in sync with _AGENT_DIM_MAP in src/utils/llm.py —
# even a typo here will produce wrong seeds and corrupt the DB.
AGENT_DIM_MAP: dict[str, str] = {
    "fundamentals_analyst_agent": "FUNDAMENTALS",
    "warren_buffett_agent":       "FUNDAMENTALS",
    "ben_graham_agent":           "FUNDAMENTALS",
    "charlie_munger_agent":       "FUNDAMENTALS",
    "michael_burry_agent":        "FUNDAMENTALS",
    "bill_ackman_agent":          "FUNDAMENTALS",
    "cathie_wood_agent":          "FUNDAMENTALS",
    "phil_fisher_agent":          "FUNDAMENTALS",
    "mohnish_pabrai_agent":       "FUNDAMENTALS",
    "peter_lynch_agent":          "FUNDAMENTALS",
    "rakesh_jhunjhunwala_agent":  "FUNDAMENTALS",
    "aswath_damodaran_agent":     "FUNDAMENTALS",
    "technical_analyst_agent":    "TECHNICAL",
    "breakout_momentum":          "TECHNICAL",
    "sentiment_agent":            "SENTIMENT",
    "news_sentiment_agent":       "SENTIMENT",
    "macro_agent":                "MACRO",
}

# Ticker sets keyed by the 8-char SHA-1 hash embedded in state filenames.
# Verified by running: hashlib.sha1(",".join(sorted(t.upper() for t in tickers)).encode()).hexdigest()[:8]
KNOWN_TICKER_SETS: dict[str, list[str]] = {
    "88e3b8c8": [
        "AAPL", "MSFT", "NVDA", "TSLA", "MSTR", "COIN",
        "SMCI", "MELI", "BTC-USD", "ETH-USD", "SOL-USD",
    ],
    # Hashes with unknown ticker sets (small runs) are skipped gracefully.
}

WINDOWS = [("1d", 1), ("5d", 5), ("20d", 20)]

INSERT_BATCH = 5_000


# ── Seed formula (mirrors _backtest_seeded_default in src/utils/llm.py) ──────

def _seed_signal(dim: str, ticker_str: str, as_of: str) -> tuple[str, float]:
    """
    Reproduce the deterministic signal from _backtest_seeded_default.
    Returns (BUY | SELL | HOLD, confidence).
    """
    digest = int(
        hashlib.sha256(f"{dim}:{ticker_str}:{as_of}".encode()).hexdigest()[:8], 16
    )
    bucket = digest % 100
    if bucket < 40:
        signal = "BUY"
    elif bucket < 65:
        signal = "SELL"
    else:
        signal = "HOLD"
    confidence = round(0.65 + (digest % 16) / 100.0, 4)
    return signal, confidence


def _pred_hash(agent_id: str, ticker: str, date_str: str) -> str:
    """Stable dedup key: identifies this (agent, ticker, date) triple uniquely."""
    return hashlib.sha256(f"{agent_id}:{ticker}:{date_str}".encode()).hexdigest()


# ── Step 1 — Scan state files ─────────────────────────────────────────────────

def scan_state_files() -> dict[str, str]:
    """Return {date_str: ticker_hash} for every PKL in STATES_DIR."""
    mapping: dict[str, str] = {}
    for f in STATES_DIR.glob("*.pkl"):
        try:
            date_str, ticker_hash = f.stem.rsplit("_", 1)
            mapping[date_str] = ticker_hash
        except ValueError:
            continue
    return mapping


# ── Step 2 — Build prediction rows ───────────────────────────────────────────

def build_prediction_rows(
    date_hash_map: dict[str, str],
    limit: int | None = None,
) -> list[dict]:
    """Generate one dict per (date, ticker, agent) for all known dates."""
    dates = sorted(
        d for d, h in date_hash_map.items() if h in KNOWN_TICKER_SETS
    )
    if limit:
        dates = dates[:limit]

    rows: list[dict] = []
    for date_str in tqdm(dates, desc="Generating signals", unit="day"):
        ticker_hash = date_hash_map[date_str]
        tickers     = KNOWN_TICKER_SETS[ticker_hash]
        ticker_str  = "|".join(sorted(t.upper() for t in tickers))
        timestamp   = f"{date_str}T16:00:00+00:00"
        run_id      = f"backtest_seed_{ticker_hash}"

        for ticker in tickers:
            for agent_id, dim in AGENT_DIM_MAP.items():
                signal, conf = _seed_signal(dim, ticker_str, date_str)
                rows.append({
                    "run_id":         run_id,
                    "agent_id":       agent_id,
                    "ticker":         ticker,
                    "signal":         signal,
                    "confidence":     conf,
                    "reasoning_hash": _pred_hash(agent_id, ticker, date_str),
                    "timestamp":      timestamp,
                })
    return rows


# ── Step 3 — Insert predictions ───────────────────────────────────────────────

def insert_predictions(conn, rows: list[dict]) -> dict[str, int]:
    """
    Bulk-insert prediction rows, skipping any with an existing reasoning_hash.
    Returns {reasoning_hash: prediction_id} for ALL hashes in rows.
    """
    existing_hashes: set[str] = {
        r[0]
        for r in conn.execute(
            "SELECT reasoning_hash FROM predictions WHERE reasoning_hash IS NOT NULL"
        ).fetchall()
    }
    logger.info("Predictions already in DB: %d", len(existing_hashes))

    new_rows = [r for r in rows if r["reasoning_hash"] not in existing_hashes]
    logger.info("New predictions to insert: %d", len(new_rows))

    sql = """
        INSERT INTO predictions
            (run_id, agent_id, ticker, signal, confidence,
             reasoning_hash, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    for i in tqdm(range(0, len(new_rows), INSERT_BATCH), desc="Inserting predictions", unit="batch"):
        batch = new_rows[i : i + INSERT_BATCH]
        conn.executemany(sql, [
            (r["run_id"], r["agent_id"], r["ticker"], r["signal"],
             r["confidence"], r["reasoning_hash"], r["timestamp"])
            for r in batch
        ])
        conn.commit()

    logger.info("Inserted %d new predictions.", len(new_rows))
    return _fetch_hash_to_id(conn, [r["reasoning_hash"] for r in rows])


def _fetch_hash_to_id(conn, hashes: list[str]) -> dict[str, int]:
    """Batch SELECT {reasoning_hash: id} from predictions."""
    result: dict[str, int] = {}
    for i in tqdm(range(0, len(hashes), 900), desc="Fetching prediction IDs", unit="batch", leave=False):
        batch = hashes[i : i + 900]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT id, reasoning_hash FROM predictions "
            f"WHERE reasoning_hash IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows:
            result[row[1]] = row[0]
    return result


# ── Step 4 — Download prices ──────────────────────────────────────────────────

def download_prices(
    tickers: list[str], start_date: str, end_date: str
) -> dict[str, pd.Series]:
    """
    Download adjusted close prices for each ticker.
    Adds 35 calendar days of buffer so the last dates have 20d forward returns.
    Returns {ticker: pd.Series(close, index=DatetimeIndex)}.
    """
    buffer_end = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=35)
    ).strftime("%Y-%m-%d")
    logger.info(
        "Downloading prices: %d tickers, %s → %s", len(tickers), start_date, buffer_end
    )
    prices: dict[str, pd.Series] = {}
    for ticker in tqdm(tickers, desc="Downloading prices", unit="ticker"):
        try:
            df = yf.download(
                ticker, start=start_date, end=buffer_end,
                interval="1d", progress=False, auto_adjust=True,
            )
            if df.empty:
                logger.warning("  %-10s — no data", ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes = df["Close"].dropna()
            if len(closes) < 25:
                logger.warning("  %-10s — only %d rows, skipping", ticker, len(closes))
                continue
            prices[ticker] = closes
            logger.debug("  %-10s — %d rows", ticker, len(closes))
        except Exception as exc:
            logger.warning("  %-10s — download failed: %s", ticker, exc)
    logger.info("Prices available for %d/%d tickers.", len(prices), len(tickers))
    return prices  # noqa — intentionally outside the for loop


# ── Step 5 — Forward returns ──────────────────────────────────────────────────

def build_forward_returns(closes: pd.Series) -> pd.DataFrame:
    """
    Compute forward returns for each trading date.

    Returns DataFrame indexed by date string, columns: fwd_1d, fwd_5d, fwd_20d.
    Values are raw price returns: (close[d+n] - close[d]) / close[d].
    """
    idx  = pd.to_datetime(closes.index).normalize()
    ser  = pd.Series(closes.values, index=idx, dtype=float)
    out  = pd.DataFrame(index=idx)
    for _, n in WINDOWS:
        out[f"fwd_{n}d"] = (ser.shift(-n) - ser) / ser

    # Index by "YYYY-MM-DD" string for O(1) lookup
    out.index = idx.strftime("%Y-%m-%d")
    return out


# ── Step 6 — Insert outcomes ──────────────────────────────────────────────────

def insert_outcomes(
    conn,
    prediction_rows: list[dict],
    hash_to_id: dict[str, int],
    fwd_returns: dict[str, pd.DataFrame],
) -> int:
    """
    Insert 3 outcome rows per prediction (1d / 5d / 20d), skipping predictions
    that already have outcomes or have no price data.
    """
    existing_pred_ids: set[int] = {
        r[0]
        for r in conn.execute("SELECT DISTINCT prediction_id FROM outcomes").fetchall()
    }
    logger.info("Predictions already with outcomes: %d", len(existing_pred_ids))

    sql = """
        INSERT INTO outcomes
            (prediction_id, ticker, actual_return_1d, actual_return_5d,
             actual_return_20d, window, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    evaluated_at   = datetime.now(timezone.utc).isoformat()
    outcome_buffer: list[tuple] = []
    total_inserted = 0
    no_price       = 0
    no_fwd         = 0

    fwd_cols = [f"fwd_{n}d" for _, n in WINDOWS]   # ["fwd_1d", "fwd_5d", "fwd_20d"]

    for pred in tqdm(prediction_rows, desc="Building outcomes", unit="pred"):
        pred_id = hash_to_id.get(pred["reasoning_hash"])
        if pred_id is None or pred_id in existing_pred_ids:
            continue

        ticker   = pred["ticker"]
        date_str = pred["timestamp"][:10]
        fwd_df   = fwd_returns.get(ticker)
        if fwd_df is None:
            no_price += 1
            continue

        if date_str not in fwd_df.index:
            no_fwd += 1
            continue

        row = fwd_df.loc[date_str]

        for (window, _), col in zip(WINDOWS, fwd_cols):
            val = row[col]
            if pd.isna(val):
                continue
            v = float(val)
            r1d  = v if window == "1d"  else None
            r5d  = v if window == "5d"  else None
            r20d = v if window == "20d" else None
            outcome_buffer.append((pred_id, ticker, r1d, r5d, r20d, window, evaluated_at))

        # Flush periodically to avoid huge in-memory buffer
        if len(outcome_buffer) >= INSERT_BATCH * 3:
            conn.executemany(sql, outcome_buffer)
            conn.commit()
            total_inserted += len(outcome_buffer)
            outcome_buffer = []

    # Final flush
    if outcome_buffer:
        for i in tqdm(range(0, len(outcome_buffer), INSERT_BATCH), desc="Inserting outcomes", unit="batch"):
            conn.executemany(sql, outcome_buffer[i : i + INSERT_BATCH])
        conn.commit()
        total_inserted += len(outcome_buffer)

    if no_price:
        logger.info("No price data: %d predictions skipped (tickers missing from yfinance)", no_price)
    if no_fwd:
        logger.info("No forward return: %d predictions skipped (tail of history)", no_fwd)

    return total_inserted


# ── Main ──────────────────────────────────────────────────────────────────────

def main(limit: int | None = None, dry_run: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("=== seed_db_from_backtest.py ===")

    # ── Step 1: scan state files ──────────────────────────────────────────────
    date_hash_map = scan_state_files()
    if not date_hash_map:
        logger.error("No *.pkl files found in %s", STATES_DIR)
        sys.exit(1)

    known_dates = sorted(d for d, h in date_hash_map.items() if h in KNOWN_TICKER_SETS)
    skipped = len(date_hash_map) - len(known_dates)
    logger.info(
        "State files: %d total | known ticker set: %d | skipped (unknown hash): %d",
        len(date_hash_map), len(known_dates), skipped,
    )

    # ── Step 2: build prediction rows ────────────────────────────────────────
    pred_rows = build_prediction_rows(date_hash_map, limit=limit)
    n_agents  = len(AGENT_DIM_MAP)
    n_tickers = len(list(KNOWN_TICKER_SETS.values())[0])
    logger.info(
        "Generated %d rows (%d days × %d tickers × %d agents)",
        len(pred_rows), len(known_dates) if not limit else min(limit, len(known_dates)),
        n_tickers, n_agents,
    )

    if dry_run:
        logger.info("[DRY RUN] Would insert ~%d predictions + ~%d outcomes. Exiting.",
                    len(pred_rows), len(pred_rows) * 3)
        return

    init_db()
    conn = get_connection()

    # ── Step 3: insert predictions ────────────────────────────────────────────
    hash_to_id = insert_predictions(conn, pred_rows)

    # ── Step 4: download prices ───────────────────────────────────────────────
    all_tickers = sorted({t for ts in KNOWN_TICKER_SETS.values() for t in ts})
    prices      = download_prices(
        all_tickers,
        start_date=(known_dates if not limit else known_dates[:limit])[0],
        end_date=(known_dates if not limit else known_dates[:limit])[-1],
    )

    # ── Step 5: build forward returns ────────────────────────────────────────
    fwd_returns: dict[str, pd.DataFrame] = {}
    logger.info("Computing forward returns …")
    for ticker, closes in tqdm(prices.items(), desc="Forward returns", unit="ticker"):
        fwd_returns[ticker] = build_forward_returns(closes)

    # ── Step 6: insert outcomes ───────────────────────────────────────────────
    n_outcomes = insert_outcomes(conn, pred_rows, hash_to_id, fwd_returns)
    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=== Done ===")
    logger.info("Predictions: %d rows processed", len(pred_rows))
    logger.info("Outcomes:    %d rows inserted", n_outcomes)
    logger.info(
        "Run dataset_builder to verify: python -m src.ml.dataset_builder"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed DB from walk-forward backtest state cache (Fase 7 prerequisite)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N trading days (for smoke testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print row counts without writing to the DB"
    )
    args = parser.parse_args()
    main(limit=args.limit, dry_run=args.dry_run)
