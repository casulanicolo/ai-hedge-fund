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
  expected_return REAL    estimated % return (-0.10 to +0.10), default 0.0
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

# ── Path to the SQLite database ───────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parents[2] / "db" / "hedge_fund.db"

# ── SQL: create the predictions table if it does not exist ────────────────────
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    agent_id        TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    signal          TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 0.0,
    expected_return REAL    NOT NULL DEFAULT 0.0,
    reasoning_hash  TEXT    NOT NULL DEFAULT '',
    timestamp       TEXT    NOT NULL
);
"""

# ── Internal helpers ──────────────────────────────────────────────────────────

def _hash_reasoning(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_CREATE_TABLE_SQL)
    # Add expected_return column to existing DBs that don't have it yet
    try:
        conn.execute("ALTER TABLE predictions ADD COLUMN expected_return REAL NOT NULL DEFAULT 0.0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def log_predictions(state: dict[str, Any]) -> str:
    """
    Read every analyst signal from state and write them to the SQLite
    predictions table. Also logs the portfolio manager's aggregated
    decision for each ticker.

    Returns the run_id used for this batch of inserts (UUID4).
    """
    run_id    = str(uuid.uuid4())
    timestamp = _now_utc()
    rows: list[tuple] = []

    data: dict = state.get("data", {})

    # ── 1. Analyst signals ────────────────────────────────────────────────────
    analyst_signals: dict = data.get("analyst_signals", {})

    for agent_id, ticker_signals in analyst_signals.items():
        if not isinstance(ticker_signals, dict):
            continue

        for ticker, payload in ticker_signals.items():
            if not isinstance(payload, dict):
                continue

            # Support both old format (signal) and new format (direction)
            direction = payload.get("direction")
            if direction:
                dir_map = {"LONG": "BUY", "SHORT": "SELL", "NEUTRAL": "HOLD"}
                signal  = dir_map.get(str(direction).upper(), "HOLD")
            else:
                raw_signal = str(payload.get("signal", "neutral")).lower()
                signal_map = {
                    "bullish": "BUY", "buy": "BUY",
                    "bearish": "SELL", "sell": "SELL",
                    "neutral": "HOLD", "hold": "HOLD",
                }
                signal = signal_map.get(raw_signal, "HOLD")

            raw_conf   = float(payload.get("confidence", 0.0))
            confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf

            expected_return = float(payload.get("expected_return", 0.0))
            reasoning       = str(payload.get("reasoning", ""))

            rows.append((
                run_id,
                agent_id,
                ticker,
                signal,
                confidence,
                expected_return,
                _hash_reasoning(reasoning),
                timestamp,
            ))

    # ── 2. Portfolio Manager aggregated decisions ─────────────────────────────
    portfolio_decisions: dict = data.get("portfolio_decisions", {})

    for ticker, decision in portfolio_decisions.items():
        if not isinstance(decision, dict):
            continue

        raw_action = str(decision.get("action", "hold")).lower()
        action_map = {
            "buy": "BUY", "sell": "SELL", "hold": "HOLD",
            "bullish": "BUY", "bearish": "SELL", "neutral": "HOLD",
        }
        signal = action_map.get(raw_action, "HOLD")

        conviction      = float(decision.get("conviction", 0.0))
        expected_return = float(decision.get("expected_return", 0.0))
        reasoning       = str(decision.get("reasoning", ""))

        rows.append((
            run_id,
            "portfolio_manager",
            ticker,
            signal,
            conviction,
            expected_return,
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
                (run_id, agent_id, ticker, signal, confidence, expected_return,
                 reasoning_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
    """LangGraph node wrapper around log_predictions()."""
    run_id = log_predictions(state)
    state.setdefault("data", {})["last_run_id"] = run_id
    return state


def query_last_run(n: int = 5) -> list[dict]:
    """Return the last n distinct run_ids with their row counts."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT run_id,
                   COUNT(*)       AS rows,
                   MIN(timestamp) AS started_at
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
