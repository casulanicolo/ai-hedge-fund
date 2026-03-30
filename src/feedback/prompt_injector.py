"""
Phase 4.4 — Prompt Self-Tuning / Prompt Injector

Reads past prediction outcomes from SQLite and generates a
"performance context" section to inject into each agent's system prompt.

Example output injected into a prompt:
─────────────────────────────────────────────────
[PERFORMANCE HISTORY — SELF-CORRECTION CONTEXT]

Your recent track record on AAPL (last 1 prediction with outcomes):
  • Accuracy (1d): 0/1 correct  (0.0%)
  • Accuracy (5d): 0/1 correct  (0.0%)
  • Accuracy (20d): 1/1 correct (100.0%)
  • Avg return on correct calls: +0.83%
  • Worst miss: -2.63% on 2026-01-05 (you predicted BUY, stock fell at 5d)

Your recent track record on NVDA (last 1 prediction with outcomes):
  • Accuracy (1d): 1/1 correct  (100.0%)
  • Accuracy (5d): 1/1 correct  (100.0%)
  • Accuracy (20d): 1/1 correct (100.0%)
  • Avg return on correct calls: +2.77%
  • No significant misses recorded.

Use this history to calibrate your confidence and flag uncertainties.
─────────────────────────────────────────────────

Public API:
    inject_into_prompt(system_prompt, agent_id, tickers, conn) -> str
    build_performance_section(agent_id, tickers, conn, n=20) -> str
    get_performance_context(agent_id, ticker, conn, n=20) -> str | None

Run standalone for a preview:
    python -m src.feedback.prompt_injector
"""

import sqlite3
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DB_PATH = Path("db/hedge_fund.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _signal_to_direction(signal: str) -> int:
    return {"BUY": 1, "SELL": -1, "HOLD": 0}.get(signal.upper(), 0)


def _return_matches_signal(signal: str, actual_return: float) -> bool:
    """True if the signal direction matched the actual price move."""
    direction = _signal_to_direction(signal)
    if direction == 0:
        return False  # HOLD is never counted as a hit
    return (direction > 0 and actual_return > 0) or (direction < 0 and actual_return < 0)


def _format_pct(value: float) -> str:
    """Format a decimal return as a signed percentage string."""
    return f"{value * 100:+.2f}%"


def _format_date(iso_timestamp: str) -> str:
    """Extract the date portion from an ISO timestamp string."""
    return iso_timestamp[:10]


# ---------------------------------------------------------------------------
# Step 1: aggregate raw DB rows into per-prediction records
# ---------------------------------------------------------------------------

def _fetch_predictions_with_outcomes(
    agent_id: str,
    ticker: str,
    conn: sqlite3.Connection,
    n: int = 20,
) -> list[dict]:
    """
    Return up to `n` past predictions (with at least one outcome window)
    for the given (agent_id, ticker), grouped so each prediction is one dict:

        {
            "signal":     "BUY" | "SELL" | "HOLD",
            "timestamp":  "2026-01-05T15:00:00+00:00",
            "return_1d":  float | None,
            "return_5d":  float | None,
            "return_20d": float | None,
        }

    The raw table has 3 rows per prediction (one per window), so we group
    them by (prediction_id, signal, timestamp) before returning.
    """
    query = """
        SELECT
            p.id            AS pred_id,
            p.signal,
            p.timestamp,
            o.actual_return_1d,
            o.actual_return_5d,
            o.actual_return_20d,
            o.window
        FROM outcomes o
        JOIN predictions p ON o.prediction_id = p.id
        WHERE p.agent_id = ?
          AND p.ticker   = ?
        ORDER BY p.timestamp DESC
    """
    rows = conn.execute(query, (agent_id, ticker)).fetchall()

    # Group by pred_id — merge the 3 window rows into one record
    grouped: dict[int, dict] = {}
    for row in rows:
        pid = row["pred_id"]
        if pid not in grouped:
            grouped[pid] = {
                "signal":     row["signal"],
                "timestamp":  row["timestamp"],
                "return_1d":  None,
                "return_5d":  None,
                "return_20d": None,
            }
        if row["actual_return_1d"] is not None:
            grouped[pid]["return_1d"] = row["actual_return_1d"]
        if row["actual_return_5d"] is not None:
            grouped[pid]["return_5d"] = row["actual_return_5d"]
        if row["actual_return_20d"] is not None:
            grouped[pid]["return_20d"] = row["actual_return_20d"]

    # Sort by timestamp descending and take up to n
    records = sorted(grouped.values(), key=lambda r: r["timestamp"], reverse=True)
    return records[:n]


# ---------------------------------------------------------------------------
# Step 2: compute metrics and render human-readable performance context
# ---------------------------------------------------------------------------

def get_performance_context(
    agent_id: str,
    ticker: str,
    conn: sqlite3.Connection,
    n: int = 20,
) -> str | None:
    """
    Generate a human-readable performance summary for one (agent_id, ticker).
    Returns None if no outcome data is available yet.
    """
    records = _fetch_predictions_with_outcomes(agent_id, ticker, conn, n)
    if not records:
        return None

    count = len(records)

    # Per-window accuracy
    windows = {
        "1d":  ("return_1d",  []),   # (is_hit, actual_return)
        "5d":  ("return_5d",  []),
        "20d": ("return_20d", []),
    }

    all_returns: list[float] = []   # returns on correct calls
    all_misses: list[tuple[float, str, str]] = []  # (loss, date, window_label)

    for label, (field, bucket) in windows.items():
        for rec in records:
            val = rec[field]
            if val is None:
                continue
            signal = rec["signal"]
            hit = _return_matches_signal(signal, val)
            bucket.append((hit, val, rec["timestamp"], signal))
            if hit:
                all_returns.append(val)
            else:
                if _signal_to_direction(signal) != 0:  # ignore HOLD misses
                    all_misses.append((val, _format_date(rec["timestamp"]), label))

    lines: list[str] = [f"Your recent track record on {ticker} (last {count} prediction(s) with outcomes):"]

    for label, (field, bucket) in windows.items():
        if not bucket:
            continue
        total = len(bucket)
        hits = sum(1 for h, *_ in bucket if h)
        pct = hits / total * 100
        lines.append(f"  • Accuracy ({label}): {hits}/{total} correct ({pct:.0f}%)")

    # Average return on correct calls
    if all_returns:
        avg_correct = sum(all_returns) / len(all_returns)
        lines.append(f"  • Avg return on correct calls: {_format_pct(avg_correct)}")
    else:
        lines.append("  • No correct directional calls recorded yet.")

    # Worst miss
    if all_misses:
        worst = min(all_misses, key=lambda x: x[0])  # most negative return
        worst_val, worst_date, worst_window = worst
        # Find what signal was predicted on that date
        matching = [
            rec for rec in records
            if _format_date(rec["timestamp"]) == worst_date
        ]
        worst_signal = matching[0]["signal"] if matching else "?"
        lines.append(
            f"  • Worst miss: {_format_pct(worst_val)} on {worst_date} "
            f"(you predicted {worst_signal}, stock moved against you at {worst_window})"
        )
    else:
        lines.append("  • No significant misses recorded.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3: assemble the full section for all tickers
# ---------------------------------------------------------------------------

def build_performance_section(
    agent_id: str,
    tickers: list[str],
    conn: sqlite3.Connection,
    n: int = 20,
) -> str:
    """
    Build the complete performance context block for an agent across all tickers.
    Returns an empty string if no data exists for any ticker.
    """
    blocks: list[str] = []

    for ticker in tickers:
        ctx = get_performance_context(agent_id, ticker, conn, n)
        if ctx:
            blocks.append(ctx)

    if not blocks:
        return ""

    header = "[PERFORMANCE HISTORY — SELF-CORRECTION CONTEXT]\n"
    footer = "\nUse this history to calibrate your confidence and flag uncertainties."
    body = "\n\n".join(blocks)
    return header + "\n" + body + footer


# ---------------------------------------------------------------------------
# Step 4: inject into an existing system prompt string
# ---------------------------------------------------------------------------

def inject_into_prompt(
    system_prompt: str,
    agent_id: str,
    tickers: list[str],
    conn: sqlite3.Connection,
    n: int = 20,
) -> str:
    """
    Append a performance context section to an existing system prompt.
    If no data is available, the prompt is returned unchanged.

    Usage inside an agent:
        from src.feedback.prompt_injector import inject_into_prompt
        system_prompt = inject_into_prompt(system_prompt, agent_id, tickers, conn)
    """
    section = build_performance_section(agent_id, tickers, conn, n)
    if not section:
        return system_prompt

    separator = "\n\n" + "─" * 60 + "\n"
    return system_prompt + separator + section


# ---------------------------------------------------------------------------
# Entry point: preview output for all agents that have outcomes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn = _get_connection()

    # Find all (agent_id, ticker) pairs with outcomes
    pairs = conn.execute("""
        SELECT DISTINCT p.agent_id, p.ticker
        FROM outcomes o
        JOIN predictions p ON o.prediction_id = p.id
        ORDER BY p.agent_id, p.ticker
    """).fetchall()

    if not pairs:
        logger.warning("No outcome data found — nothing to preview.")
        conn.close()
        raise SystemExit(0)

    # Group tickers by agent
    agent_tickers: dict[str, list[str]] = defaultdict(list)
    for row in pairs:
        agent_tickers[row["agent_id"]].append(row["ticker"])

    # Preview inject_into_prompt for each agent
    dummy_prompt = "You are a disciplined investment analyst."

    print("\n" + "=" * 70)
    print("PROMPT INJECTOR — PREVIEW")
    print("=" * 70)

    for agent_id, tickers in sorted(agent_tickers.items()):
        result = inject_into_prompt(dummy_prompt, agent_id, tickers, conn)
        print(f"\n>>> Agent: {agent_id}")
        print("-" * 70)
        print(result)
        print()

    conn.close()
    logger.info("Preview complete.")
