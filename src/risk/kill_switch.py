"""
src/risk/kill_switch.py — Fase 8
Manual kill switch via .athanor_kill flag file.

The file contains JSON: {"reason": str, "armed_at": ISO-8601}

arm(reason)         — write flag file + audit log
disarm()            — remove flag file + audit log
is_armed()          — True if flag file exists
close_all_and_exit(adapter) — close every open position, audit, sys.exit(0)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.audit.event_types import EventType, Severity
from src.audit.trail import log_event

logger = logging.getLogger(__name__)

KILL_SWITCH_PATH = Path(".athanor_kill")


def is_armed() -> bool:
    return KILL_SWITCH_PATH.exists()


def arm(reason: str = "manual") -> None:
    payload = {
        "reason": reason,
        "armed_at": datetime.now(timezone.utc).isoformat(),
    }
    KILL_SWITCH_PATH.write_text(json.dumps(payload), encoding="utf-8")
    log_event(
        EventType.KILL_SWITCH,
        Severity.CRITICAL,
        details={"action": "armed", "reason": reason},
    )
    logger.critical("[kill_switch] ARMED — reason: %s", reason)


def disarm() -> None:
    if KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()
    log_event(
        EventType.KILL_SWITCH,
        Severity.WARNING,
        details={"action": "disarmed"},
    )
    logger.warning("[kill_switch] DISARMED")


def get_reason() -> Optional[str]:
    if not KILL_SWITCH_PATH.exists():
        return None
    try:
        data = json.loads(KILL_SWITCH_PATH.read_text(encoding="utf-8"))
        return data.get("reason")
    except Exception:
        return "unknown"


def close_all_and_exit(adapter=None) -> None:
    """Close every open Alpaca position then sys.exit(0)."""
    logger.critical("[kill_switch] close_all_and_exit triggered")

    if adapter is not None:
        try:
            positions = adapter.get_positions()
            from src.execution.orders import TradeOrder
            from datetime import datetime, timezone as tz
            import uuid

            run_id = f"killswitch-{uuid.uuid4().hex[:8]}"
            for pos in positions:
                try:
                    order = TradeOrder(
                        ticker=pos.ticker,
                        action="CLOSE",
                        quantity=int(abs(float(pos.qty))),
                        notional_usd=None,
                        order_type="MARKET",
                        limit_price=None,
                        stop_loss=None,
                        take_profit=None,
                        time_in_force="DAY",
                        conviction=0.0,
                        weighted_conviction=0.0,
                        regime_at_decision="RISK_OFF",
                        reasoning="Kill switch triggered",
                        agent_contributions={},
                        created_at=datetime.now(tz.utc),
                        run_id=run_id,
                    )
                    result = adapter.submit_order(order)
                    log_event(
                        EventType.CLOSE_ALL,
                        Severity.CRITICAL,
                        ticker=pos.ticker,
                        run_id=run_id,
                        details={"broker_order_id": result.broker_order_id, "status": result.status},
                    )
                    logger.critical("[kill_switch] Closed %s → %s", pos.ticker, result.status)
                except Exception as exc:
                    logger.error("[kill_switch] Failed to close %s: %s", pos.ticker, exc)
        except Exception as exc:
            logger.error("[kill_switch] get_positions failed: %s", exc)

    log_event(EventType.CLOSE_ALL, Severity.CRITICAL, details={"action": "kill_switch_exit"})
    sys.exit(0)
