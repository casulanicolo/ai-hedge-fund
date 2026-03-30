"""
src/feedback/logger.py
Phase 4.1 - Prediction Logger

Logs every agent signal and the portfolio manager's aggregated signal
to the SQLite database after each graph run.

Table: predictions
  run_id          TEXT    UUID generated once per pipeline run
  agent_id        TEXT    e.g. "technical_analyst", "warren_buffett"
  ticker          TEXT    e.g. "AAPL"
  signal          TEXT    "bullish" | "bearish" | "neutral"
  confidence      REAL    0.0 – 1.0
  reasoning_hash  TEXT    SHA-256 of the reasoning string (first 16 hex chars)
  timestamp       TEXT    ISO-8601 UTC
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Path to the SQLite database ──────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parents[2] / "db" / "hedge_fund.db"

# ── SQL: create the predictions table if it does not exist ───────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    agent_id        TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    signal          TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 0.0,
    reasoning_hash  TEXT    NOT NULL DEFAULT '',
    timestamp       TEXT    NOT NULL
);
"""

# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash_reasoning(text: str) -> str:
    """Return the first 16 hex characters of SHA-256(text)."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_utc() -> str:
    """Current time as ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def _get_connection() -> sqlite3.Connection:
    """Open (and return) a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent writes
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def log_predictions(state: dict[str, Any]) -> str:
    """
    Read every analyst signal from state and write them to the SQLite
    predictions table.  Also logs the portfolio manager's aggregated
    decision for each ticker.

    Parameters
    ----------
    state : dict
        The final AgentState produced by the LangGraph pipeline.

    Returns
    -------
    str
        The run_id used for this batch of inserts (UUID4).
    """
    run_id = str(uuid.uuid4())
    timestamp = _now_utc()
    rows: list[tuple] = []

    data: dict = state.get("data", {})

    # ── 1. Analyst signals ────────────────────────────────────────────────────
    analyst_signals: dict = data.get("analyst_signals", {})

    for agent_id, ticker_signals in analyst_signals.items():
        # ticker_signals is usually  {ticker: {signal, confidence, reasoning, …}}
        if not isinstance(ticker_signals, dict):
            continue

        for ticker, payload in ticker_signals.items():
            if not isinstance(payload, dict):
                continue

            raw_signal = str(payload.get("signal", "neutral")).lower()
            # Normalise to DB CHECK constraint vocabulary (BUY / SELL / HOLD)
            signal_map = {
                "bullish": "BUY", "buy": "BUY",
                "bearish": "SELL", "sell": "SELL",
                "neutral": "HOLD", "hold": "HOLD",
            }
            signal     = signal_map.get(raw_signal, "HOLD")
            raw_conf   = float(payload.get("confidence", 0.0))
            # Agents may return confidence as 0–100 (percent) or 0.0–1.0
            confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
            reasoning  = str(payload.get("reasoning", ""))

            rows.append((
                run_id,
                agent_id,
                ticker,
                signal,
                confidence,
                _hash_reasoning(reasoning),
                timestamp,
            ))

    # ── 2. Portfolio Manager aggregated decisions ─────────────────────────────
    portfolio_decisions: dict = data.get("portfolio_decisions", {})

    for ticker, decision in portfolio_decisions.items():
        if not isinstance(decision, dict):
            continue

        # Portfolio manager uses action (BUY/SELL/HOLD) + conviction
        raw_action = str(decision.get("action", "hold")).lower()

        # Normalise action → DB CHECK constraint vocabulary (BUY / SELL / HOLD)
        action_map = {"buy": "BUY", "sell": "SELL", "hold": "HOLD",
                      "bullish": "BUY", "bearish": "SELL", "neutral": "HOLD"}
        signal = action_map.get(raw_action, "HOLD")

        conviction = float(decision.get("conviction", 0.0))
        reasoning  = str(decision.get("reasoning", ""))

        rows.append((
            run_id,
            "portfolio_manager",       # fixed agent_id for the PM
            ticker,
            signal,
            conviction,
            _hash_reasoning(reasoning),
            timestamp,
        ))

    # ── 3. Write to SQLite ────────────────────────────────────────────────────
    if not rows:
        print("[PredictionLogger] No signals found in state — nothing logged.")
        return run_id

    conn = _get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO predictions
                (run_id, agent_id, ticker, signal, confidence, reasoning_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        print(
            f"[PredictionLogger] run_id={run_id}  "
            f"rows_written={len(rows)}  "
            f"agents={len(analyst_signals)}  "
            f"timestamp={timestamp}"
        )
    finally:
        conn.close()

    return run_id


def prediction_log_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node wrapper around log_predictions().

    Called automatically by the graph after the portfolio manager.
    Writes all signals to SQLite and returns the state unchanged.
    """
    run_id = log_predictions(state)
    # Store the run_id back into state so downstream nodes can reference it
    state.setdefault("data", {})["last_run_id"] = run_id
    return state


def query_last_run(n: int = 5) -> list[dict]:
    """
    Utility: return the last n distinct run_ids with their row counts.
    Useful for a quick sanity check after a pipeline run.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT run_id,
                   COUNT(*)          AS rows,
                   MIN(timestamp)    AS started_at
            FROM   predictions
            GROUP  BY run_id
            ORDER  BY started_at DESC
            LIMIT  ?
            """,
            (n,),
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()
