"""
tests/risk/test_compliance.py — Fase 8
One test per compliance check (pass + fail scenario).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.risk.compliance_checks import (
    ComplianceResult,
    all_passed,
    run_all,
    _cc1_kill_switch,
    _cc2_circuit_breakers,
    _cc3_max_notional,
    _cc4_max_positions,
    _cc5_concentration,
    _cc6_cash_buffer,
)

KILL_SWITCH_PATH = Path(".athanor_kill")


def _order(action="OPEN_LONG", ticker="AAPL", notional=500.0, quantity=5):
    o = MagicMock()
    o.action = action
    o.ticker = ticker
    o.notional_usd = notional
    o.quantity = quantity
    o.regime_at_decision = "RISK_ON"
    return o


def _adapter(equity=10_000.0, cash=2_000.0, positions=None):
    a = MagicMock()
    account = MagicMock()
    account.equity = equity
    account.cash = cash
    account.status = "ACTIVE"
    a.get_account.return_value = account
    a.get_positions.return_value = positions or []
    return a


def _cleanup_kill():
    if KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()


# ── CC1: kill switch ──────────────────────────────────────────────────────────

class TestCC1:
    def setup_method(self):
        _cleanup_kill()

    def teardown_method(self):
        _cleanup_kill()

    def test_passes_when_disarmed(self):
        r = _cc1_kill_switch(_order("OPEN_LONG"))
        assert r.passed

    def test_fails_when_armed(self):
        KILL_SWITCH_PATH.write_text('{"reason":"test"}')
        r = _cc1_kill_switch(_order("OPEN_LONG"))
        assert not r.passed
        assert r.check_id == "CC1"

    def test_close_always_passes_even_when_armed(self):
        KILL_SWITCH_PATH.write_text('{"reason":"test"}')
        r = _cc1_kill_switch(_order("CLOSE"))
        assert r.passed


# ── CC2: circuit breakers ─────────────────────────────────────────────────────

class TestCC2:
    def test_passes_when_no_cb_active(self):
        import src.risk.compliance_checks as cc
        orig = cc.is_cb_active
        cc.is_cb_active = lambda cb_id: False
        try:
            r = _cc2_circuit_breakers(_order("OPEN_LONG"))
        finally:
            cc.is_cb_active = orig
        assert r.passed

    def test_fails_when_cb1_active(self):
        import src.risk.compliance_checks as cc
        orig = cc.is_cb_active
        cc.is_cb_active = lambda cb_id: cb_id == "cb1"
        try:
            r = _cc2_circuit_breakers(_order("OPEN_LONG"))
        finally:
            cc.is_cb_active = orig
        assert not r.passed
        assert "CB1" in r.reason.upper()

    def test_close_skips_cb_check(self):
        import src.risk.compliance_checks as cc
        orig = cc.is_cb_active
        cc.is_cb_active = lambda cb_id: True
        try:
            r = _cc2_circuit_breakers(_order("CLOSE"))
        finally:
            cc.is_cb_active = orig
        assert r.passed


# ── CC3: max notional ─────────────────────────────────────────────────────────

class TestCC3:
    def test_passes_small_notional(self):
        order = _order("OPEN_LONG", notional=1_000.0)  # 10% of 10k equity
        r = _cc3_max_notional(order, _adapter(equity=10_000))
        assert r.passed

    def test_fails_large_notional(self):
        order = _order("OPEN_LONG", notional=3_000.0)  # 30% of 10k equity
        r = _cc3_max_notional(order, _adapter(equity=10_000))
        assert not r.passed
        assert r.check_id == "CC3"

    def test_hold_skips_check(self):
        r = _cc3_max_notional(_order("HOLD", notional=99_999), _adapter())
        assert r.passed


# ── CC4: max positions ────────────────────────────────────────────────────────

class TestCC4:
    def test_passes_below_limit(self):
        r = _cc4_max_positions(_order("OPEN_LONG"), open_positions=[MagicMock(), MagicMock()])
        assert r.passed

    def test_fails_at_limit(self):
        positions = [MagicMock() for _ in range(3)]  # == MAX_ACTIVE_TRADES default
        r = _cc4_max_positions(_order("OPEN_LONG"), open_positions=positions)
        assert not r.passed
        assert r.check_id == "CC4"

    def test_close_skips_check(self):
        positions = [MagicMock() for _ in range(10)]
        r = _cc4_max_positions(_order("CLOSE"), open_positions=positions)
        assert r.passed


# ── CC5: concentration ────────────────────────────────────────────────────────

class TestCC5:
    def _pos(self, ticker, market_value):
        p = MagicMock()
        p.ticker = ticker
        p.market_value = market_value
        return p

    def test_passes_low_concentration(self):
        positions = [self._pos("AAPL", 1_000)]   # 10% of 10k
        r = _cc5_concentration(_order("OPEN_LONG", ticker="AAPL"), positions, _adapter(equity=10_000))
        assert r.passed

    def test_fails_high_concentration(self):
        positions = [self._pos("AAPL", 2_500)]   # 25% of 10k
        r = _cc5_concentration(_order("OPEN_LONG", ticker="AAPL"), positions, _adapter(equity=10_000))
        assert not r.passed
        assert r.check_id == "CC5"

    def test_hold_skips_check(self):
        positions = [self._pos("AAPL", 9_000)]
        r = _cc5_concentration(_order("HOLD", ticker="AAPL"), positions, _adapter())
        assert r.passed


# ── CC6: cash buffer ──────────────────────────────────────────────────────────

class TestCC6:
    def test_passes_sufficient_cash(self):
        r = _cc6_cash_buffer(_order("OPEN_LONG"), _adapter(equity=10_000, cash=2_000))
        assert r.passed

    def test_fails_insufficient_cash(self):
        r = _cc6_cash_buffer(_order("OPEN_LONG"), _adapter(equity=10_000, cash=500))
        assert not r.passed
        assert r.check_id == "CC6"

    def test_hold_skips_check(self):
        r = _cc6_cash_buffer(_order("HOLD"), _adapter(equity=10_000, cash=0))
        assert r.passed


# ── run_all integration ───────────────────────────────────────────────────────

class TestRunAll:
    def setup_method(self):
        _cleanup_kill()

    def teardown_method(self):
        _cleanup_kill()

    def test_all_pass_clean_state(self):
        import src.risk.compliance_checks as cc
        orig_cb = cc.is_cb_active
        orig_ks = cc._ks_is_armed
        cc.is_cb_active = lambda cb_id: False
        cc._ks_is_armed = lambda: False
        try:
            order = _order("OPEN_LONG", notional=500)
            results = run_all(order, [], _adapter(equity=10_000, cash=2_000))
        finally:
            cc.is_cb_active = orig_cb
            cc._ks_is_armed = orig_ks
        assert all_passed(results)
        assert len(results) == 6

    def test_all_passed_helper(self):
        results = [ComplianceResult("CC1", True, "ok"), ComplianceResult("CC2", True, "ok")]
        assert all_passed(results)

    def test_not_all_passed(self):
        results = [ComplianceResult("CC1", True, "ok"), ComplianceResult("CC2", False, "fail")]
        assert not all_passed(results)
