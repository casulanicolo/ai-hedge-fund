"""
tests/risk/test_circuit_breakers.py — Fase 8
One test per CB: trigger + no-trigger scenario.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.risk.circuit_breakers import (
    CBStatus,
    _check_cb1,
    _check_cb3,
    _check_cb4,
    _check_cb5,
    _check_cb2,
    _flag_path,
    is_cb_active,
    reset_cb,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_adapter(equity: float = 10_000.0, last_equity: float = 10_000.0,
                  positions: list | None = None):
    adapter = MagicMock()
    account = MagicMock()
    account.equity = equity
    account.last_equity = last_equity
    account.cash = equity * 0.15
    account.status = "ACTIVE"
    adapter.get_account.return_value = account
    adapter.get_positions.return_value = positions or []
    return adapter


def _cleanup_flag(cb_id: str):
    p = _flag_path(cb_id)
    if p.exists():
        p.unlink()


# ── CB1: daily portfolio loss ─────────────────────────────────────────────────

class TestCB1:
    def setup_method(self):
        _cleanup_flag("cb1")

    def teardown_method(self):
        _cleanup_flag("cb1")

    def test_cb1_no_trigger_equity_ok(self):
        adapter = _mock_adapter(equity=10_000, last_equity=10_000)
        status = _check_cb1(adapter)
        assert status.cb_id == "CB1"
        assert not status.triggered

    def test_cb1_triggers_on_loss(self):
        adapter = _mock_adapter(equity=9_600, last_equity=10_000)  # -4%
        status = _check_cb1(adapter)
        assert status.triggered
        assert status.cb_id == "CB1"
        assert _flag_path("cb1").exists()

    def test_cb1_flag_already_set(self):
        _flag_path("cb1").write_text("already set")
        adapter = _mock_adapter()
        status = _check_cb1(adapter)
        assert status.triggered


# ── CB2: per-position stop-out ────────────────────────────────────────────────

class TestCB2:
    def _make_position(self, ticker, avg_entry, market_val, qty, side="long"):
        pos = MagicMock()
        pos.ticker = ticker
        pos.avg_entry_price = avg_entry
        pos.market_value = market_val
        pos.qty = qty
        pos.side = side
        return pos

    def test_cb2_no_trigger_small_loss(self):
        pos = self._make_position("AAPL", 100, 97 * 10, 10)  # -3%
        adapter = _mock_adapter(positions=[pos])
        statuses = _check_cb2(adapter)
        assert all(not s.triggered for s in statuses)

    def test_cb2_triggers_on_large_loss(self):
        pos = self._make_position("AAPL", 100, 88 * 10, 10)  # -12%
        adapter = _mock_adapter(positions=[pos])
        statuses = _check_cb2(adapter)
        triggered = [s for s in statuses if s.triggered]
        assert len(triggered) == 1
        assert triggered[0].cb_id == "CB2"
        assert "AAPL" in triggered[0].details.get("ticker", "")

    def test_cb2_no_adapter(self):
        statuses = _check_cb2(None)
        assert statuses == []


# ── CB3: VIX spike ────────────────────────────────────────────────────────────

class TestCB3:
    def _mock_yf(self, vix_value):
        import src.risk.circuit_breakers as cbmod
        mock_yf = MagicMock()
        ti = MagicMock()
        ti.fast_info = {"last_price": vix_value}
        mock_yf.Ticker.return_value = ti
        return mock_yf, cbmod

    def test_cb3_no_trigger_low_vix(self):
        import src.risk.circuit_breakers as cbmod
        orig = cbmod.yf
        try:
            mock_yf = MagicMock()
            ti = MagicMock()
            ti.fast_info = {"last_price": 18.5}
            mock_yf.Ticker.return_value = ti
            cbmod.yf = mock_yf
            status = _check_cb3()
        finally:
            cbmod.yf = orig
        assert not status.triggered

    def test_cb3_triggers_high_vix(self):
        import src.risk.circuit_breakers as cbmod
        orig = cbmod.yf
        try:
            mock_yf = MagicMock()
            ti = MagicMock()
            ti.fast_info = {"last_price": 42.0}
            mock_yf.Ticker.return_value = ti
            cbmod.yf = mock_yf
            status = _check_cb3()
        finally:
            cbmod.yf = orig
        assert status.triggered
        assert status.cb_id == "CB3"

    def test_cb3_vix_fetch_error_does_not_crash(self):
        import src.risk.circuit_breakers as cbmod
        orig = cbmod.yf
        try:
            mock_yf = MagicMock()
            mock_yf.Ticker.side_effect = RuntimeError("network error")
            cbmod.yf = mock_yf
            status = _check_cb3()
        finally:
            cbmod.yf = orig
        assert status.cb_id == "CB3"
        assert not status.triggered


# ── CB4: rejection rate ───────────────────────────────────────────────────────

class TestCB4:
    def _make_rows(self, statuses_list):
        rows = []
        for sv in statuses_list:
            r = MagicMock()
            r.__getitem__ = lambda self, key, _sv=sv: _sv
            rows.append(r)
        return rows

    def test_cb4_no_trigger_good_rate(self):
        import src.risk.circuit_breakers as cbmod
        orig = cbmod.get_connection
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = self._make_rows(["FILLED"] * 10)
        cbmod.get_connection = lambda: mock_conn
        try:
            status = _check_cb4()
        finally:
            cbmod.get_connection = orig
        assert not status.triggered

    def test_cb4_triggers_high_rejection(self):
        import src.risk.circuit_breakers as cbmod
        orig = cbmod.get_connection
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = self._make_rows(
            ["REJECTED"] * 6 + ["FILLED"] * 4
        )
        cbmod.get_connection = lambda: mock_conn
        try:
            status = _check_cb4()
        finally:
            cbmod.get_connection = orig
        assert status.cb_id == "CB4"
        assert status.triggered

    def test_cb4_no_orders(self):
        import src.risk.circuit_breakers as cbmod
        orig = cbmod.get_connection
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        cbmod.get_connection = lambda: mock_conn
        try:
            status = _check_cb4()
        finally:
            cbmod.get_connection = orig
        assert not status.triggered


# ── CB5: equity drawdown ─────────────────────────────────────────────────────

class TestCB5:
    def setup_method(self):
        _cleanup_flag("cb5")

    def teardown_method(self):
        _cleanup_flag("cb5")

    def test_cb5_no_trigger_equity_ok(self):
        adapter = _mock_adapter(equity=10_000, last_equity=10_000)
        status = _check_cb5(adapter)
        assert not status.triggered

    def test_cb5_triggers_on_drawdown(self):
        adapter = _mock_adapter(equity=8_000, last_equity=10_000)  # -20%
        status = _check_cb5(adapter)
        assert status.triggered
        assert status.cb_id == "CB5"
        assert _flag_path("cb5").exists()

    def test_cb5_flag_already_set(self):
        _flag_path("cb5").write_text("already set")
        adapter = _mock_adapter()
        status = _check_cb5(adapter)
        assert status.triggered
