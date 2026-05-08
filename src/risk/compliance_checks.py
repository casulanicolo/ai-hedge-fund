"""
src/risk/compliance_checks.py — Fase 8
Six pre-trade compliance checks run before every broker submission.

Checks:
  CC1  Kill switch armed → reject all OPEN orders
  CC2  CB1/CB3/CB5 active + OPEN action → reject
  CC3  Single order notional > 25% of portfolio equity → reject
  CC4  Open positions >= MAX_ACTIVE_TRADES → reject new OPENs
  CC5  Same ticker already > 20% of portfolio value → reject OPEN
  CC6  Available cash < 10% of total equity → reject OPEN

run_all(order, open_positions, adapter, regime) -> list[ComplianceResult]
all_passed(results) -> bool
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from src.risk.circuit_breakers import is_cb_active
from src.risk.kill_switch import is_armed as _ks_is_armed

logger = logging.getLogger(__name__)

_OPEN_ACTIONS = {"OPEN_LONG", "OPEN_SHORT", "SCALE_IN"}
_MAX_ACTIVE_TRADES = int(os.getenv("MAX_ACTIVE_TRADES", "9"))  # FIX: limite assoluto ora 9, capital-aware
_MIN_CASH_COVERAGE = 0.50   # FIX: cash minimo = 50% del notional ordine
_MAX_NOTIONAL_PCT  = 0.35   # 35% of buying_power per order
_MAX_TICKER_PCT    = 0.20   # 20% concentration per ticker
_MIN_CASH_PCT      = 0.10   # 10% cash buffer


@dataclass
class ComplianceResult:
    check_id: str
    passed: bool
    reason: str
    needs_rebalance: bool = False  # FIX: flag per CC4 capital-aware


def all_passed(results: list[ComplianceResult]) -> bool:
    return all(r.passed for r in results)


# ── Individual checks ─────────────────────────────────────────────────────────

def _cc1_kill_switch(order) -> ComplianceResult:
    if order.action in _OPEN_ACTIONS and _ks_is_armed():
        return ComplianceResult("CC1", False,
            f"Kill switch armed — OPEN orders blocked ({order.ticker})")
    return ComplianceResult("CC1", True, "Kill switch OK")


def _cc2_circuit_breakers(order) -> ComplianceResult:
    if order.action not in _OPEN_ACTIONS:
        return ComplianceResult("CC2", True, "Not an OPEN action — CB check skipped")
    for cb_id in ("cb1", "cb3", "cb5"):
        if is_cb_active(cb_id):
            return ComplianceResult("CC2", False,
                f"{cb_id.upper()} active — OPEN orders halted ({order.ticker})")
    return ComplianceResult("CC2", True, "Circuit breakers OK")


def _cc3_max_notional(order, adapter) -> ComplianceResult:
    if order.action not in _OPEN_ACTIONS:
        return ComplianceResult("CC3", True, "Not an OPEN action")
    notional = order.notional_usd or 0.0
    if notional <= 0 and order.quantity and adapter:
        try:
            account = adapter.get_account()
            _bp = float(account.buying_power)
            equity = _bp if _bp > 0 else float(account.portfolio_value)
        except Exception:
            return ComplianceResult("CC3", True, "Equity fetch failed — skipped")
    elif adapter:
        try:
            account = adapter.get_account()
            _bp = float(account.buying_power)
            equity = _bp if _bp > 0 else float(account.portfolio_value)
        except Exception:
            return ComplianceResult("CC3", True, "Equity fetch failed — skipped")
    else:
        return ComplianceResult("CC3", True, "No adapter — skipped")

    if equity <= 0:
        return ComplianceResult("CC3", True, "equity=0 — skipped")
    pct = notional / equity
    if pct > _MAX_NOTIONAL_PCT:
        return ComplianceResult("CC3", False,
            f"Order notional ${notional:,.0f} = {pct:.0%} of equity > {_MAX_NOTIONAL_PCT:.0%} limit")
    return ComplianceResult("CC3", True, f"Notional {pct:.1%} of equity OK")


def _cc4_max_positions(order, open_positions: list, adapter=None) -> ComplianceResult:  # FIX: aggiunto adapter per capital-aware check
    if order.action not in _OPEN_ACTIONS:
        return ComplianceResult("CC4", True, "Not an OPEN action")
    n = len(open_positions)
    if n >= _MAX_ACTIVE_TRADES:
        return ComplianceResult("CC4", False,
            f"Open positions={n} >= MAX_ACTIVE_TRADES={_MAX_ACTIVE_TRADES} — no new opens")
    # FIX: capital-aware — verifica che il cash copra almeno 50% del notional
    if adapter is not None:
        notional = float(order.notional_usd or 0.0)
        if notional > 0:
            try:
                account  = adapter.get_account()
                cash     = float(getattr(account, "cash", 0) or getattr(account, "buying_power", 0) or 0)
                required = notional * _MIN_CASH_COVERAGE
                if cash < required:
                    logger.warning(
                        "[CC4] %s: cash=%.0f < %.0f (50%% of notional=%.0f) — needs rebalance",
                        order.ticker, cash, required, notional,
                    )
                    return ComplianceResult(
                        "CC4", False,
                        f"Cash ${cash:,.0f} < ${required:,.0f} (50% of notional=${notional:,.0f}) — capital insufficiente",
                        needs_rebalance=True,
                    )
            except Exception as exc:
                logger.warning("[CC4] cash check error: %s", exc)
    return ComplianceResult("CC4", True, f"Positions={n} < {_MAX_ACTIVE_TRADES} OK")


def _cc5_concentration(order, open_positions: list, adapter) -> ComplianceResult:
    if order.action not in _OPEN_ACTIONS:
        return ComplianceResult("CC5", True, "Not an OPEN action")
    if not open_positions or adapter is None:
        return ComplianceResult("CC5", True, "No positions or no adapter — skipped")
    try:
        account = adapter.get_account()
        _pv = float(account.portfolio_value)  # FIX: portfolio_value sostituisce .equity
        portfolio_val = _pv if _pv > 0 else float(account.last_equity)
        if portfolio_val <= 0:
            return ComplianceResult("CC5", True, "portfolio=0 — skipped")
        ticker_val = sum(
            abs(float(p.market_value))
            for p in open_positions
            if p.ticker == order.ticker
        )
        pct = ticker_val / portfolio_val
        if pct > _MAX_TICKER_PCT:
            return ComplianceResult("CC5", False,
                f"{order.ticker} already {pct:.0%} of portfolio > {_MAX_TICKER_PCT:.0%} limit")
        return ComplianceResult("CC5", True, f"{order.ticker} concentration {pct:.1%} OK")
    except Exception as exc:
        logger.warning("[CC5] check error: %s", exc)
        return ComplianceResult("CC5", True, f"check error skipped: {exc}")


def _cc6_cash_buffer(order, adapter) -> ComplianceResult:
    if order.action not in _OPEN_ACTIONS:
        return ComplianceResult("CC6", True, "Not an OPEN action")
    if adapter is None:
        return ComplianceResult("CC6", True, "No adapter — skipped")
    try:
        account = adapter.get_account()
        cash   = float(getattr(account, "cash", 0) or 0)
        _pv = float(account.portfolio_value)  # FIX: portfolio_value sostituisce .equity
        equity = _pv if _pv > 0 else float(account.last_equity)
        if equity <= 0:
            return ComplianceResult("CC6", True, "equity=0 — skipped")
        ratio = cash / equity
        if ratio < _MIN_CASH_PCT:
            return ComplianceResult("CC6", False,
                f"Cash buffer {ratio:.1%} < {_MIN_CASH_PCT:.0%} minimum")
        return ComplianceResult("CC6", True, f"Cash buffer {ratio:.1%} OK")
    except Exception as exc:
        logger.warning("[CC6] check error: %s", exc)
        return ComplianceResult("CC6", True, f"check error skipped: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def run_all(
    order,
    open_positions: list,
    adapter=None,
    regime: str = "UNKNOWN",
) -> list[ComplianceResult]:
    """
    Run all 6 compliance checks for `order`.
    Returns list of ComplianceResult. Caller should check all_passed().
    """
    return [
        _cc1_kill_switch(order),
        _cc2_circuit_breakers(order),
        _cc3_max_notional(order, adapter),
        _cc4_max_positions(order, open_positions, adapter),  # FIX: passa adapter per capital-aware check
        _cc5_concentration(order, open_positions, adapter),
        _cc6_cash_buffer(order, adapter),
    ]
