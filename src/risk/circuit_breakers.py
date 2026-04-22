"""
src/risk/circuit_breakers.py — Fase 8
Five circuit breakers (CB1-CB5).

CB1  Daily portfolio loss > 3%    → halt new OPEN orders for today
CB2  Single position unrealized loss > 8%  → force FULL_EXIT that position
CB3  VIX > 35                     → halt new OPEN orders
CB4  Order rejection rate > 50% (last 10)  → WARNING, soft alert
CB5  Account equity < 85% of last-week baseline → halt ALL new orders

check_all(adapter=None) -> list[CBStatus]
trigger_cb1() / trigger_cb5()    → write flag file + audit log
is_cb_active(cb_id, date=today)  → check flag file
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf

from src.audit.event_types import EventType, Severity
from src.audit.trail import log_event
from src.db.init_db import get_connection

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
CB1_DAILY_LOSS_PCT   = 0.03   # 3% daily loss → halt opens
CB2_POSITION_LOSS_PCT = 0.08  # 8% per-position loss → force close
CB3_VIX_THRESHOLD    = 35.0   # VIX > 35 → halt opens
CB4_REJECTION_WINDOW = 10     # last N orders
CB4_REJECTION_RATE   = 0.50   # >50% rejection rate
CB5_EQUITY_DRAWDOWN  = 0.15   # equity < 85% of baseline → halt all


@dataclass
class CBStatus:
    cb_id: str
    triggered: bool
    reason: str
    severity: str = "INFO"
    details: dict = field(default_factory=dict)


# ── Flag file helpers ─────────────────────────────────────────────────────────

def _flag_path(cb_id: str, for_date: Optional[date] = None) -> Path:
    d = (for_date or datetime.now(timezone.utc).date()).isoformat()
    return Path(f".circuit_breaker_{cb_id.lower()}_{d}")


def is_cb_active(cb_id: str, for_date: Optional[date] = None) -> bool:
    return _flag_path(cb_id, for_date).exists()


def _trigger_flag(cb_id: str, reason: str) -> None:
    path = _flag_path(cb_id)
    path.write_text(reason, encoding="utf-8")
    log_event(
        EventType.CIRCUIT_BREAKER,
        Severity.CRITICAL,
        details={"cb_id": cb_id, "reason": reason},
    )
    logger.critical("[CB] %s TRIGGERED — %s", cb_id, reason)


def reset_cb(cb_id: str, for_date: Optional[date] = None) -> None:
    path = _flag_path(cb_id, for_date)
    if path.exists():
        path.unlink()
    log_event(
        EventType.CIRCUIT_BREAKER_RESET,
        Severity.WARNING,
        details={"cb_id": cb_id},
    )
    logger.warning("[CB] %s RESET", cb_id)


# ── Per-CB checks ─────────────────────────────────────────────────────────────

def _check_cb1(adapter) -> CBStatus:
    """CB1: daily portfolio P&L < -3%."""
    if is_cb_active("cb1"):
        return CBStatus("CB1", True, "Daily loss flag already set today", Severity.CRITICAL)

    if adapter is None:
        return CBStatus("CB1", False, "No adapter — skipped")
    try:
        account = adapter.get_account()
        equity       = float(account.equity)
        last_equity  = float(getattr(account, "last_equity", equity))
        if last_equity <= 0:
            return CBStatus("CB1", False, "last_equity=0 — skipped")
        daily_ret = (equity - last_equity) / last_equity
        if daily_ret < -CB1_DAILY_LOSS_PCT:
            reason = f"Daily P&L={daily_ret:.2%} < -{CB1_DAILY_LOSS_PCT:.0%}"
            _trigger_flag("cb1", reason)
            return CBStatus("CB1", True, reason, Severity.CRITICAL,
                            {"daily_ret": daily_ret})
        return CBStatus("CB1", False, f"Daily P&L={daily_ret:.2%} OK")
    except Exception as exc:
        logger.warning("[CB1] check failed: %s", exc)
        return CBStatus("CB1", False, f"check error: {exc}")


def _check_cb2(adapter) -> list[CBStatus]:
    """CB2: per-position unrealized loss > 8% → one CBStatus per position."""
    results: list[CBStatus] = []
    if adapter is None:
        return results
    try:
        positions = adapter.get_positions()
        for pos in positions:
            try:
                entry = float(pos.avg_entry_price)
                curr  = abs(float(pos.market_value) / float(pos.qty)) if float(pos.qty) else entry
                if entry <= 0:
                    continue
                loss_pct = (curr - entry) / entry
                side = str(pos.side).upper()
                if side == "SHORT":
                    loss_pct = -loss_pct
                if loss_pct < -CB2_POSITION_LOSS_PCT:
                    reason = f"{pos.ticker} loss={loss_pct:.2%} > -{CB2_POSITION_LOSS_PCT:.0%}"
                    log_event(
                        EventType.CIRCUIT_BREAKER,
                        Severity.CRITICAL,
                        ticker=pos.ticker,
                        details={"cb_id": "CB2", "loss_pct": loss_pct, "reason": reason},
                    )
                    logger.critical("[CB2] %s", reason)
                    results.append(CBStatus("CB2", True, reason, Severity.CRITICAL,
                                           {"ticker": pos.ticker, "loss_pct": loss_pct}))
                else:
                    results.append(CBStatus("CB2", False, f"{pos.ticker} loss={loss_pct:.2%} OK"))
            except Exception as exc:
                logger.warning("[CB2] position processing error: %s", exc)
    except Exception as exc:
        logger.warning("[CB2] get_positions failed: %s", exc)
    return results


def _check_cb3() -> CBStatus:
    """CB3: VIX > 35."""
    try:
        vix = yf.Ticker("^VIX").fast_info.get("last_price") or yf.download(
            "^VIX", period="1d", progress=False
        )["Close"].iloc[-1]
        vix = float(vix)
        if vix > CB3_VIX_THRESHOLD:
            reason = f"VIX={vix:.1f} > {CB3_VIX_THRESHOLD}"
            log_event(
                EventType.CIRCUIT_BREAKER,
                Severity.CRITICAL,
                details={"cb_id": "CB3", "vix": vix, "reason": reason},
            )
            logger.critical("[CB3] %s", reason)
            return CBStatus("CB3", True, reason, Severity.CRITICAL, {"vix": vix})
        return CBStatus("CB3", False, f"VIX={vix:.1f} OK")
    except Exception as exc:
        logger.warning("[CB3] VIX fetch failed: %s", exc)
        return CBStatus("CB3", False, f"VIX fetch error: {exc}")


def _check_cb4() -> CBStatus:
    """CB4: rejection rate > 50% in last 10 orders (soft alert)."""
    try:
        conn = get_connection()
        rows = conn.execute(
            """
            SELECT status FROM executed_orders
            ORDER BY submitted_at DESC LIMIT ?
            """,
            (CB4_REJECTION_WINDOW,),
        ).fetchall()
        conn.close()
        if not rows:
            return CBStatus("CB4", False, "No orders to evaluate")
        rejected = sum(1 for r in rows if str(r["status"]).upper() in ("REJECTED", "CANCELED"))
        rate = rejected / len(rows)
        if rate > CB4_REJECTION_RATE:
            reason = f"Rejection rate={rate:.0%} > {CB4_REJECTION_RATE:.0%} (last {len(rows)} orders)"
            log_event(
                EventType.CIRCUIT_BREAKER,
                Severity.WARNING,
                details={"cb_id": "CB4", "rejection_rate": rate, "reason": reason},
            )
            logger.warning("[CB4] %s", reason)
            return CBStatus("CB4", True, reason, Severity.WARNING, {"rejection_rate": rate})
        return CBStatus("CB4", False, f"Rejection rate={rate:.0%} OK")
    except Exception as exc:
        logger.warning("[CB4] check failed: %s", exc)
        return CBStatus("CB4", False, f"check error: {exc}")


def _check_cb5(adapter) -> CBStatus:
    """CB5: equity < 85% of last-week close → halt ALL new orders."""
    if is_cb_active("cb5"):
        return CBStatus("CB5", True, "Equity drawdown flag already set today", Severity.CRITICAL)

    if adapter is None:
        return CBStatus("CB5", False, "No adapter — skipped")
    try:
        account = adapter.get_account()
        equity = float(account.equity)

        # Use last_equity as 1-day proxy if weekly baseline not tracked separately
        last_equity = float(getattr(account, "last_equity", equity))
        if last_equity <= 0:
            return CBStatus("CB5", False, "last_equity=0 — skipped")

        drawdown = (equity - last_equity) / last_equity
        if drawdown < -CB5_EQUITY_DRAWDOWN:
            reason = f"Equity drawdown={drawdown:.2%} < -{CB5_EQUITY_DRAWDOWN:.0%}"
            _trigger_flag("cb5", reason)
            return CBStatus("CB5", True, reason, Severity.CRITICAL,
                            {"drawdown": drawdown, "equity": equity})
        return CBStatus("CB5", False, f"Equity drawdown={drawdown:.2%} OK")
    except Exception as exc:
        logger.warning("[CB5] check failed: %s", exc)
        return CBStatus("CB5", False, f"check error: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def check_all(adapter=None) -> list[CBStatus]:
    """
    Run all CB checks. Returns flat list of CBStatus (one per CB/position).
    CB2 may return multiple statuses (one per position).
    """
    results: list[CBStatus] = []
    results.append(_check_cb1(adapter))
    results.extend(_check_cb2(adapter))
    results.append(_check_cb3())
    results.append(_check_cb4())
    results.append(_check_cb5(adapter))
    return results


def any_halt_opens(statuses: list[CBStatus]) -> bool:
    """CB1, CB3, CB5 triggered → no new OPEN orders."""
    halt_ids = {"CB1", "CB3", "CB5"}
    return any(s.triggered and s.cb_id in halt_ids for s in statuses)


def any_halt_all(statuses: list[CBStatus]) -> bool:
    """CB5 triggered → halt ALL new orders (most severe)."""
    return any(s.triggered and s.cb_id == "CB5" for s in statuses)


def force_close_cb2_positions(adapter, statuses: list[CBStatus]) -> None:
    """Submit CLOSE orders for all CB2-triggered positions."""
    from src.execution.orders import TradeOrder
    import uuid
    from datetime import datetime, timezone as tz

    run_id = f"cb2-{uuid.uuid4().hex[:8]}"
    for s in statuses:
        if s.cb_id == "CB2" and s.triggered:
            ticker = s.details.get("ticker")
            if not ticker:
                continue
            try:
                positions = adapter.get_positions()
                pos = next((p for p in positions if p.ticker == ticker), None)
                if pos is None:
                    continue
                qty = int(abs(float(pos.qty)))
                order = TradeOrder(
                    ticker=ticker,
                    action="CLOSE",
                    quantity=qty,
                    notional_usd=None,
                    order_type="MARKET",
                    limit_price=None,
                    stop_loss=None,
                    take_profit=None,
                    time_in_force="DAY",
                    conviction=0.0,
                    weighted_conviction=0.0,
                    regime_at_decision="RISK_OFF",
                    reasoning=f"CB2: {s.reason}",
                    agent_contributions={},
                    created_at=datetime.now(tz.utc),
                    run_id=run_id,
                )
                result = adapter.submit_order(order)
                log_event(
                    EventType.CIRCUIT_BREAKER,
                    Severity.CRITICAL,
                    ticker=ticker,
                    run_id=run_id,
                    details={"cb_id": "CB2", "action": "FORCE_CLOSE",
                             "broker_id": result.broker_order_id},
                )
                logger.critical("[CB2] Force-closed %s → %s", ticker, result.status)
            except Exception as exc:
                logger.error("[CB2] force_close failed for %s: %s", ticker, exc)
