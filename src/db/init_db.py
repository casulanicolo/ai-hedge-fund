"""
src/db/init_db.py
──────────────────
Database initialisation and thin CRUD helpers for Athanor Alpha.

Usage
-----
From CLI (idempotent — safe to run multiple times):

    python -m src.db.init_db

From code:

    from src.db.init_db import get_connection, init_db

    init_db()                          # create tables if missing
    conn = get_connection()            # get a thread-local connection
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from src.db.models import (
    AgentHyperparam,
    AgentWeight,
    Outcome,
    PipelineRun,
    Prediction,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

_ROOT   = Path(__file__).resolve().parents[2]   # project root
DB_PATH = _ROOT / "db" / "hedge_fund.db"
SCHEMA_PATH = _ROOT / "db" / "schema.sql"


# ─────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Return an sqlite3 connection with sensible defaults.
    WAL mode and foreign keys are set on every new connection.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ─────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> None:
    """
    Create all tables from schema.sql (idempotent — uses CREATE IF NOT EXISTS).
    """
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection(db_path)
    with conn:
        conn.executescript(schema)
    logger.info("Database initialised at %s", db_path)


# ─────────────────────────────────────────────
# CRUD — Predictions
# ─────────────────────────────────────────────

def insert_prediction(conn: sqlite3.Connection, p: Prediction) -> int:
    """Insert a Prediction and return its rowid."""
    cur = conn.execute(
        """
        INSERT INTO predictions
            (run_id, agent_id, ticker, signal, confidence, reasoning_hash, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (p.run_id, p.agent_id, p.ticker, p.signal,
         p.confidence, p.reasoning_hash, p.timestamp),
    )
    conn.commit()
    return cur.lastrowid


def get_recent_predictions(
    conn: sqlite3.Connection,
    agent_id: str,
    ticker: str,
    limit: int = 20,
) -> list[sqlite3.Row]:
    """Fetch the N most recent predictions for a given (agent, ticker) pair."""
    return conn.execute(
        """
        SELECT * FROM predictions
        WHERE agent_id = ? AND ticker = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (agent_id, ticker, limit),
    ).fetchall()


# ─────────────────────────────────────────────
# CRUD — Outcomes
# ─────────────────────────────────────────────

def insert_outcome(conn: sqlite3.Connection, o: Outcome) -> int:
    cur = conn.execute(
        """
        INSERT INTO outcomes
            (prediction_id, ticker, window,
             actual_return_1d, actual_return_5d, actual_return_20d, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (o.prediction_id, o.ticker, o.window,
         o.actual_return_1d, o.actual_return_5d, o.actual_return_20d,
         o.evaluated_at),
    )
    conn.commit()
    return cur.lastrowid


# ─────────────────────────────────────────────
# CRUD — Agent Weights
# ─────────────────────────────────────────────

def upsert_agent_weight(conn: sqlite3.Connection, w: AgentWeight) -> None:
    """Insert or update the weight for (agent_id, ticker)."""
    conn.execute(
        """
        INSERT INTO agent_weights (agent_id, ticker, weight, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(agent_id, ticker)
        DO UPDATE SET weight = excluded.weight, updated_at = excluded.updated_at
        """,
        (w.agent_id, w.ticker, w.weight, w.updated_at),
    )
    conn.commit()


def get_agent_weight(
    conn: sqlite3.Connection,
    agent_id: str,
    ticker: str,
    default: float = 1.0,
) -> float:
    row = conn.execute(
        "SELECT weight FROM agent_weights WHERE agent_id = ? AND ticker = ?",
        (agent_id, ticker),
    ).fetchone()
    return float(row["weight"]) if row else default


def get_all_weights(conn: sqlite3.Connection) -> dict[tuple[str, str], float]:
    """Return {(agent_id, ticker): weight} for all known pairs."""
    rows = conn.execute("SELECT agent_id, ticker, weight FROM agent_weights").fetchall()
    return {(r["agent_id"], r["ticker"]): float(r["weight"]) for r in rows}


# ─────────────────────────────────────────────
# CRUD — Agent Hyperparams
# ─────────────────────────────────────────────

def upsert_hyperparam(conn: sqlite3.Connection, h: AgentHyperparam) -> None:
    conn.execute(
        """
        INSERT INTO agent_hyperparams (agent_id, param_name, value, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(agent_id, param_name)
        DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (h.agent_id, h.param_name, h.value, h.updated_at),
    )
    conn.commit()


def get_hyperparam(
    conn: sqlite3.Connection,
    agent_id: str,
    param_name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM agent_hyperparams WHERE agent_id = ? AND param_name = ?",
        (agent_id, param_name),
    ).fetchone()
    return row["value"] if row else default


# ─────────────────────────────────────────────
# CRUD — Pipeline Runs
# ─────────────────────────────────────────────

def start_run(conn: sqlite3.Connection, run: PipelineRun) -> None:
    conn.execute(
        """
        INSERT INTO pipeline_runs (run_id, started_at, status, tickers)
        VALUES (?, ?, ?, ?)
        """,
        (run.run_id, run.started_at, run.status, json.dumps(run.tickers)),
    )
    conn.commit()


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    finished_at: str,
    status: str = "completed",
    error_msg: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE pipeline_runs
        SET finished_at = ?, status = ?, error_msg = ?
        WHERE run_id = ?
        """,
        (finished_at, status, error_msg, run_id),
    )
    conn.commit()


# ─────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    init_db()
    print(f"✓ Database ready at {DB_PATH}")
