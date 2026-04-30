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


def sync_from_alpaca(adapter: "BrokerAdapter") -> int:
    """
    Bidirectional sync: import Alpaca-side fills (SL/TP triggers, manual closes)
    into executed_orders, and update stale DB statuses for pending rows.
    Returns count of rows inserted or updated.
    """
    from datetime import timedelta
    from types import SimpleNamespace

    if not hasattr(adapter, "_client"):
        return reconcile_orders(adapter)

    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from src.db.init_db import (
            get_connection, insert_executed_order, update_executed_order_status,
        )
    except Exception as exc:
        logger.warning("[sync] import failed: %s", exc)
        return 0

    try:
        after = datetime.now(timezone.utc) - timedelta(days=5)
        alpaca_orders = adapter._client.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=200,
                after=after,
            )
        )
    except Exception as exc:
        logger.warning("[sync] get_orders failed: %s", exc)
        return 0

    try:
        conn = get_connection()
    except Exception as exc:
        logger.warning("[sync] get_connection failed: %s", exc)
        return 0

    known_ids = {
        r[0]
        for r in conn.execute(
            "SELECT broker_order_id FROM executed_orders WHERE broker_order_id IS NOT NULL"
        ).fetchall()
    }

    _SIDE_TO_ACTION = {"buy": "OPEN_LONG", "sell": "CLOSE"}
    _TERMINAL = {"FILLED", "CANCELED", "EXPIRED", "DONE_FOR_DAY", "REJECTED"}

    def _norm(s) -> str:
        if hasattr(s, "name"):
            return s.name.upper()
        return str(s).split(".")[-1].upper()

    changed = 0
    for o in alpaca_orders:
        oid = str(o.id)
        status = _norm(o.status)
        filled_at_str = o.filled_at.isoformat() if getattr(o, "filled_at", None) else None
        fill_price = float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else None
        filled_qty = float(o.filled_qty) if getattr(o, "filled_qty", None) else None

        if oid in known_ids:
            if status in _TERMINAL:
                update_executed_order_status(
                    conn,
                    broker_order_id=oid,
                    status=status,
                    fill_price=fill_price,
                    filled_qty=filled_qty,
                    filled_at=filled_at_str,
                )
                changed += 1
        elif status == "FILLED" and fill_price:
            side_str = str(getattr(o, "side", "sell")).split(".")[-1].lower()
            action = _SIDE_TO_ACTION.get(side_str, "CLOSE")
            ticker = o.symbol

            run_id_row = conn.execute(
                """
                SELECT run_id FROM executed_orders
                WHERE ticker = ? AND action IN ('OPEN_LONG', 'OPEN_SHORT')
                  AND status = 'FILLED'
                ORDER BY submitted_at DESC LIMIT 1
                """,
                (ticker,),
            ).fetchone()
            sync_run_id = (run_id_row[0] if run_id_row and run_id_row[0] else "alpaca-sync")

            submitted_at = (
                o.submitted_at.isoformat() if getattr(o, "submitted_at", None)
                else datetime.now(timezone.utc).isoformat()
            )
            conn.execute(
                "INSERT OR IGNORE INTO pipeline_runs (run_id, started_at, status, tickers)"
                " VALUES (?, ?, ?, ?)",
                ("alpaca-sync", submitted_at, "completed", "[]"),
            )
            conn.commit()

            fake_order = SimpleNamespace(
                ticker=ticker, action=action,
                quantity=int(filled_qty) if filled_qty else None,
                notional_usd=None, stop_loss=None, take_profit=None,
            )
            fake_result = SimpleNamespace(
                broker_order_id=oid, status="FILLED",
                rejection_reason=None, raw_response={},
            )
            insert_executed_order(conn, sync_run_id, fake_order, fake_result, submitted_at)
            conn.execute(
                "UPDATE executed_orders SET fill_price=?, filled_at=?, filled_qty=?"
                " WHERE broker_order_id=?",
                (fill_price, filled_at_str, filled_qty, oid),
            )
            conn.commit()
            changed += 1
            logger.info(
                "[sync] imported %s %s broker=%s fill=%.4f",
                action, ticker, oid[:8], fill_price,
            )

    conn.close()
    logger.info("[sync] sync_from_alpaca: %d rows inserted/updated", changed)
    return changed


def reconcile_orders(adapter: "BrokerAdapter") -> int:
    """
    Fetch pending orders from DB, query broker for current status, write updates back.
    Returns count of rows updated.
    """
    try:
        from src.db.init_db import get_connection, get_pending_orders, update_executed_order_status
    except Exception as exc:
        logger.warning("[executor] reconcile_orders: DB import failed: %s", exc)
        return 0

    conn = get_connection()
    rows = get_pending_orders(conn)
    updated = 0
    for row in rows:
        broker_order_id = row[1]
        if not broker_order_id:
            continue
        snapshot = adapter.get_order(broker_order_id)
        if snapshot is None:
            continue
        filled_at_str = (
            snapshot.filled_at.isoformat()
            if snapshot.filled_at is not None
            else None
        )
        update_executed_order_status(
            conn,
            broker_order_id=broker_order_id,
            status=snapshot.status,
            fill_price=snapshot.filled_avg_price,
            filled_qty=snapshot.filled_qty if snapshot.filled_qty else None,
            filled_at=filled_at_str,
        )
        updated += 1
        logger.info(
            "[executor] reconcile: %s → status=%s fill=%.4f qty=%s",
            broker_order_id[:8], snapshot.status,
            snapshot.filled_avg_price or 0,
            snapshot.filled_qty,
        )
    conn.close()
    logger.info("[executor] reconcile_orders: updated %d rows", updated)
    return updated

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

        # Reconcile fill status for any orders just submitted
        if report.submitted > 0:
            try:
                reconcile_orders(self._adapter)
            except Exception as exc:
                logger.warning("[executor] post-execute reconcile failed: %s", exc)

        return report
