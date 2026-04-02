"""
Phase 4.3 — Weight Adjuster
Reads outcomes from SQLite, computes a performance score per (agent_id, ticker),
and updates agent_weights using Exponential Weighted Average (EWA).

Formula: w_new = alpha * performance_score + (1 - alpha) * w_old
- alpha = 0.15 (learning rate)
- floor = 0.05 (minimum weight, no agent is ever fully silenced)

Performance score (0.0 — 1.0) combines:
  - hit_rate:          fraction of directional calls that were correct
  - avg_return:        mean actual return when the signal was correct
  - inverse_drawdown:  1 / (1 + max single-prediction loss), penalises big misses

Run standalone:
    python -m src.feedback.weight_adjuster
"""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DB_PATH = Path("db/hedge_fund.db")

ALPHA = 0.15   # EWA learning rate
FLOOR = 0.05   # minimum weight floor
DEFAULT_WEIGHT = 1.0  # weight assigned to new agents with no history

# Whitelist of currently active agents.
# Add/remove agent names here when agents are added or removed from the pipeline.
ACTIVE_AGENTS = {
    "warren_buffett_agent",
    "ben_graham_agent",
    "charlie_munger_agent",
    "michael_burry_agent",
    "bill_ackman_agent",
    "cathie_wood_agent",
    "technical_analyst_agent",
    "fundamentals_analyst_agent",
    "sentiment_agent",
    "breakout_momentum_agent",
}

# Whitelist of currently active tickers (loaded from config if needed).
# Inactive tickers (e.g. SPY, removed tickers) are excluded automatically.
ACTIVE_TICKERS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "TSLA", "MSTR", "COIN", "SMCI", "MELI",
    "BTC-USD", "ETH-USD", "SOL-USD",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _signal_to_direction(signal: str) -> int:
    """Convert BUY/SELL/HOLD to +1 / -1 / 0."""
    return {"BUY": 1, "SELL": -1, "HOLD": 0}.get(signal.upper(), 0)


def _return_direction(actual_return: float) -> int:
    """Positive return → +1, negative → -1, zero → 0."""
    if actual_return > 0:
        return 1
    elif actual_return < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Core: performance score for one (agent_id, ticker) pair
# ---------------------------------------------------------------------------

def compute_performance_score(agent_id: str, ticker: str, conn: sqlite3.Connection) -> float | None:
    """
    Compute a composite performance score in [0, 1] for one (agent_id, ticker).

    Returns None if there are no outcome rows available yet.

    The score is built from three components weighted equally (1/3 each):
      1. hit_rate       — fraction of directional calls that matched the actual move
      2. avg_return     — normalised average actual return on correct calls (capped at 1.0)
      3. inv_drawdown   — 1 / (1 + |worst_loss|), so big losses push this toward 0
    """
    # Fetch all predictions for this agent/ticker that already have outcomes
    query = """
        SELECT
            p.signal,
            o.actual_return_1d,
            o.actual_return_5d,
            o.actual_return_20d,
            o.window
        FROM outcomes o
        JOIN predictions p ON o.prediction_id = p.id
        WHERE p.agent_id = ?
          AND p.ticker   = ?
    """
    rows = conn.execute(query, (agent_id, ticker)).fetchall()

    if not rows:
        return None

    # For each row pick the non-NULL return value (only one column is filled per row)
    results = []
    for row in rows:
        signal_dir = _signal_to_direction(row["signal"])
        # Find the actual return value for this window
        actual = (
            row["actual_return_1d"]
            if row["actual_return_1d"] is not None
            else row["actual_return_5d"]
            if row["actual_return_5d"] is not None
            else row["actual_return_20d"]
        )
        if actual is None:
            continue  # outcome not yet populated — skip
        results.append((signal_dir, actual))

    if not results:
        return None

    # --- hit rate ---
    # HOLD signals are scored as neutral (not a miss, not a hit)
    directional_calls = [(sd, ar) for sd, ar in results if sd != 0]
    if directional_calls:
        hits = sum(
            1 for sd, ar in directional_calls
            if sd == _return_direction(ar)
        )
        hit_rate = hits / len(directional_calls)
    else:
        # All signals were HOLD — neutral score
        hit_rate = 0.5

    # --- average return on correct calls ---
    correct_returns = [
        ar for sd, ar in directional_calls
        if sd == _return_direction(ar)
    ]
    if correct_returns:
        raw_avg = sum(correct_returns) / len(correct_returns)
        # Normalise: a 5% avg return maps to ~1.0; cap at 1.0
        avg_return_score = min(raw_avg / 0.05, 1.0)
        avg_return_score = max(avg_return_score, 0.0)
    else:
        avg_return_score = 0.0

    # --- inverse drawdown (penalise worst single loss) ---
    losses = [abs(ar) for sd, ar in directional_calls if sd != _return_direction(ar) and ar < 0]
    max_loss = max(losses) if losses else 0.0
    inv_drawdown = 1.0 / (1.0 + max_loss)

    # --- composite score (equal weights) ---
    score = (hit_rate + avg_return_score + inv_drawdown) / 3.0
    score = max(0.0, min(score, 1.0))  # clamp to [0, 1]

    logger.debug(
        f"  {agent_id} / {ticker}: hit_rate={hit_rate:.2f}, "
        f"avg_return_score={avg_return_score:.2f}, inv_drawdown={inv_drawdown:.2f} "
        f"→ score={score:.4f}"
    )
    return score


# ---------------------------------------------------------------------------
# Main: update agent_weights for all (agent_id, ticker) pairs with outcomes
# ---------------------------------------------------------------------------

def adjust_weights(alpha: float = ALPHA, floor: float = FLOOR) -> dict:
    """
    Update the agent_weights table for every (agent_id, ticker) pair
    that has at least one outcome row AND belongs to active agents/tickers.

    Returns a summary dict: { (agent_id, ticker): new_weight }
    """
    conn = _get_connection()
    now = datetime.now(timezone.utc).isoformat()
    summary = {}

    try:
        # Find all distinct (agent_id, ticker) pairs that have outcomes
        pairs_query = """
            SELECT DISTINCT p.agent_id, p.ticker
            FROM outcomes o
            JOIN predictions p ON o.prediction_id = p.id
        """
        pairs = conn.execute(pairs_query).fetchall()

        if not pairs:
            logger.warning("No outcomes found in DB — agent_weights not updated.")
            return summary

        logger.info(f"Found {len(pairs)} (agent_id, ticker) pair(s) with outcomes in DB.")

        skipped = 0
        for pair in pairs:
            agent_id = pair["agent_id"]
            ticker   = pair["ticker"]

            # --- Filter: skip inactive agents and tickers ---
            if agent_id not in ACTIVE_AGENTS:
                logger.debug(f"  Skipping {agent_id}/{ticker} — agent not in ACTIVE_AGENTS.")
                skipped += 1
                continue
            if ticker not in ACTIVE_TICKERS:
                logger.debug(f"  Skipping {agent_id}/{ticker} — ticker not in ACTIVE_TICKERS.")
                skipped += 1
                continue

            # Compute performance score for this pair
            score = compute_performance_score(agent_id, ticker, conn)
            if score is None:
                logger.info(f"  Skipping {agent_id}/{ticker} — no usable outcome data.")
                continue

            # Fetch current weight (or use default if not yet in the table)
            existing = conn.execute(
                "SELECT weight FROM agent_weights WHERE agent_id = ? AND ticker = ?",
                (agent_id, ticker)
            ).fetchone()

            w_old = existing["weight"] if existing else DEFAULT_WEIGHT

            # EWA update
            w_new = alpha * score + (1.0 - alpha) * w_old
            w_new = max(w_new, floor)  # enforce minimum floor

            # Upsert into agent_weights
            conn.execute(
                """
                INSERT INTO agent_weights (agent_id, ticker, weight, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(agent_id, ticker) DO UPDATE SET
                    weight     = excluded.weight,
                    updated_at = excluded.updated_at
                """,
                (agent_id, ticker, w_new, now)
            )

            summary[(agent_id, ticker)] = w_new
            logger.info(
                f"  {agent_id:35s} / {ticker:6s}: "
                f"score={score:.4f}, w_old={w_old:.4f} → w_new={w_new:.4f}"
            )

        if skipped:
            logger.info(f"Skipped {skipped} pair(s) belonging to inactive agents or tickers.")

        conn.commit()
        logger.info(f"agent_weights updated: {len(summary)} row(s) written.")

    finally:
        conn.close()

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== Weight Adjuster (Phase 4.3) ===")
    result = adjust_weights()
    if result:
        logger.info("Summary:")
        for (agent_id, ticker), w in sorted(result.items()):
            logger.info(f"  {agent_id:35s} / {ticker}: {w:.4f}")
    else:
        logger.warning("No weights were updated.")
