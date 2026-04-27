"""
src/execution/executor.py
──────────────────────────
TradeExecutor: pre-checks, order sequencing, DB logging, ExecutionReport.

Execution order: CLOSE first (frees capital), then ADJUST, then OPEN/SCALE.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from pydantic import BaseModel

from src.execution.broker_adapter import BrokerAdapter, OrderSubmitResult
from src.execution.orders import TradeOrder

# Fase 8 — Risk & audit (graceful fallback if not installed yet)
try:
    from src.risk.compliance_checks import run_all as _run_compliance, all_passed
    from src.audit.trail import log_event as _audit_log
    from src.audit.event_types import EventType, Severity
    _RISK_AVAILABLE = True
except Exception:
    _RISK_AVAILABLE = False

logger = logging.getLogger(__name__)

DAILY_ORDER_CAP = int(__import__("os").getenv("ALPACA_DAILY_ORDER_CAP", "20"))

# Priority order for execution sequencing
_ACTION_PRIORITY: dict[str, int] = {
    "CLOSE":      0,
    "ADJUST_SL":  1,
    "ADJUST_TP":  1,
    "SCALE_OUT":  2,
    "SCALE_IN":   3,
    "OPEN_LONG":  4,
    "OPEN_SHORT": 4,
    "HOLD":       99,
}


# ─────────────────────────────────────────────────────────────────────────────
# Report model
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionReport(BaseModel):
    run_id: str
    submitted: int = 0
    filled: int = 0
    rejected: int = 0
    skipped: int = 0
    results: list[dict] = []
    errors: list[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_today_submissions() -> int:
    try:
        from src.db.init_db import get_connection
        conn = get_connection()
        today = datetime.now(timezone.utc).date().isoformat()
        row = conn.execute(
            "SELECT COUNT(*) FROM executed_orders WHERE DATE(submitted_at) = ?",
            (today,),
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception as exc:
        logger.warning("[executor] _count_today_submissions failed: %s", exc)
        return 0


def _log_to_db(
    run_id: str,
    order: TradeOrder,
    result: OrderSubmitResult,
    submitted_at: str,
) -> None:
    try:
        from src.db.init_db import get_connection, insert_executed_order
        conn = get_connection()
        insert_executed_order(conn, run_id, order, result, submitted_at)
        conn.close()
    except Exception as exc:
        logger.warning("[executor] _log_to_db failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class TradeExecutor:
    def __init__(
        self,
        adapter: BrokerAdapter,
        run_id: str,
        max_orders_per_day: int = DAILY_ORDER_CAP,
    ):
        self._adapter = adapter
        self._run_id  = run_id
        self._cap     = max_orders_per_day

    # ── Pre-checks ───────────────────────────────────────────────────────────

    def _check_market_open(self) -> bool:
        # Allow pre-market submission: Alpaca queues DAY orders until open.
        # Only block after market close (21:00 UTC) or on weekends.
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        market_close_utc = now.replace(hour=21, minute=0, second=0, microsecond=0)
        if now >= market_close_utc:
            return False
        return True

    def _check_account(self) -> tuple[bool, str]:
        try:
            account = self._adapter.get_account()
            if account.status != "ACTIVE":
                return False, f"Account status: {account.status}"
            return True, "ok"
        except Exception as exc:
            return False, f"get_account failed: {exc}"

    def _check_daily_cap(self) -> tuple[bool, int]:
        today_count = _count_today_submissions()
        return today_count < self._cap, today_count

    # ── Main entry ───────────────────────────────────────────────────────────

    def execute(self, orders: list[TradeOrder]) -> ExecutionReport:
        report = ExecutionReport(run_id=self._run_id)

        # ── Pre-check: market open ────────────────────────────────────────────
        if not self._check_market_open():
            logger.warning("[executor] Market is closed — skipping all orders")
            report.skipped = len(orders)
            report.errors.append("Market closed — no orders submitted")
            return report

        # ── Pre-check: account active ─────────────────────────────────────────
        account_ok, account_msg = self._check_account()
        if not account_ok:
            logger.error("[executor] Account check failed: %s — aborting", account_msg)
            report.skipped = len(orders)
            report.errors.append(f"Account check failed: {account_msg}")
            return report

        # ── Pre-check: daily cap ──────────────────────────────────────────────
        cap_ok, today_count = self._check_daily_cap()
        if not cap_ok:
            msg = (
                f"Daily order cap reached ({today_count}/{self._cap}) — "
                "no orders will be submitted today"
            )
            logger.error("[executor] %s", msg)
            report.skipped = len(orders)
            report.errors.append(msg)
            return report

        remaining_cap = self._cap - today_count

        # ── Sort: CLOSE first, then ADJUST, then OPEN ────────────────────────
        active = [o for o in orders if o.action != "HOLD"]
        holds  = [o for o in orders if o.action == "HOLD"]
        active.sort(key=lambda o: _ACTION_PRIORITY.get(o.action, 99))

        report.skipped += len(holds)

        # ── Fetch open positions once for compliance checks ───────────────────
        open_positions: list = []
        if _RISK_AVAILABLE:
            try:
                open_positions = self._adapter.get_positions()
            except Exception as exc:
                logger.warning("[executor] get_positions for compliance failed: %s", exc)

        # ── Execute ───────────────────────────────────────────────────────────
        submitted_this_run = 0
        for order in active:
            if submitted_this_run >= remaining_cap:
                logger.warning("[executor] Cap reached mid-run — remaining orders skipped")
                report.skipped += 1
                continue

            # ── Compliance pre-check (Fase 8) ─────────────────────────────────
            if _RISK_AVAILABLE:
                compliance_results = _run_compliance(
                    order, open_positions, self._adapter,
                    regime=getattr(order, "regime_at_decision", "UNKNOWN"),
                )
                if not all_passed(compliance_results):
                    failed = [r for r in compliance_results if not r.passed]
                    reasons = "; ".join(r.reason for r in failed)
                    _audit_log(
                        EventType.COMPLIANCE_FAIL,
                        Severity.WARNING,
                        ticker=order.ticker,
                        run_id=self._run_id,
                        details={"action": order.action, "checks": [
                            {"id": r.check_id, "reason": r.reason} for r in failed
                        ]},
                    )
                    logger.warning("[executor] Compliance FAIL %s %s — %s",
                                   order.action, order.ticker, reasons)
                    report.skipped += 1
                    report.errors.append(f"Compliance: {order.action} {order.ticker}: {reasons}")
                    continue

            submitted_at = datetime.now(timezone.utc).isoformat()
            result = self._adapter.submit_order(order)

            _log_to_db(self._run_id, order, result, submitted_at)

            report.results.append({
                "ticker":           order.ticker,
                "action":           order.action,
                "broker_order_id":  result.broker_order_id,
                "status":           result.status,
                "rejection_reason": result.rejection_reason,
            })

            if result.status == "SKIPPED":
                report.skipped += 1
            elif result.success:
                report.submitted += 1
                submitted_this_run += 1
                if _RISK_AVAILABLE:
                    _audit_log(
                        EventType.ORDER_SUBMIT,
                        Severity.INFO,
                        ticker=order.ticker,
                        run_id=self._run_id,
                        details={"action": order.action,
                                 "broker_order_id": result.broker_order_id,
                                 "status": result.status,
                                 "notional_usd": order.notional_usd},
                    )
                logger.info(
                    "[executor] ✓ %s %s — broker_id=%s status=%s",
                    order.action, order.ticker, result.broker_order_id, result.status,
                )
            else:
                report.rejected += 1
                reason = result.rejection_reason or "unknown"
                report.errors.append(f"{order.action} {order.ticker}: {reason}")
                if _RISK_AVAILABLE:
                    _audit_log(
                        EventType.ORDER_REJECT,
                        Severity.WARNING,
                        ticker=order.ticker,
                        run_id=self._run_id,
                        details={"action": order.action,
                                 "reason": reason,
                                 "broker_order_id": result.broker_order_id},
                    )
                logger.error(
                    "[executor] ✗ %s %s — %s",
                    order.action, order.ticker, reason,
                )

        logger.info(
            "[executor] Run %s complete: submitted=%d rejected=%d skipped=%d",
            self._run_id[:8], report.submitted, report.rejected, report.skipped,
        )
        return report
