"""
src/audit/trail.py — Fase 8
Thread-safe append-only audit trail writer.

Usage:
    from src.audit.trail import log_event
    from src.audit.event_types import EventType, Severity

    log_event(EventType.ORDER_SUBMIT, Severity.INFO, ticker="AAPL",
              run_id="abc123", details={"qty": 10, "notional": 500.0})
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _get_conn():
    try:
        from src.db.init_db import get_connection
        return get_connection()
    except Exception as exc:
        logger.warning("[audit] Cannot get DB connection: %s", exc)
        return None


def log_event(
    event_type: str,
    severity: str = "INFO",
    *,
    ticker: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Append one row to audit_trail. Never raises — failures are logged only."""
    ts = datetime.now(timezone.utc).isoformat()
    details_json = json.dumps(details) if details else None

    with _lock:
        try:
            conn = _get_conn()
            if conn is None:
                return
            conn.execute(
                """
                INSERT INTO audit_trail
                    (event_type, severity, ticker, agent_id, run_id, details, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_type, severity, ticker, agent_id, run_id, details_json, ts),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("[audit] log_event failed (%s): %s", event_type, exc)
