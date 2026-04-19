"""
tests/test_daemon_mock.py
──────────────────────────
Integration-level daemon tests using mocked Alpaca adapter.
Verifies that ActiveMonitorDaemon.run_tick() wires exit rules → broker correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import floor
from unittest.mock import MagicMock, patch

import pytest

from src.monitor.monitor_state import EnrichedPosition, MonitorTick
from src.execution.broker_adapter import PositionSnapshot, ClockInfo, OrderSubmitResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_raw_position(
    ticker="AAPL",
    side="long",
    qty=100.0,
    entry=100.0,
    current=100.0,
) -> PositionSnapshot:
    return PositionSnapshot(
        ticker=ticker,
        side=side,
        qty=qty,
        avg_entry_price=entry,
        market_value=current * qty,
        unrealized_pl=(current - entry) * qty,
        unrealized_pl_pct=(current - entry) / entry,
    )


def _make_clock(minutes_to_close: float = 120.0) -> ClockInfo:
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    return ClockInfo(
        is_open=True,
        next_open=None,
        next_close=now + timedelta(minutes=minutes_to_close),
        timestamp=now,
    )


def _make_submit_result(ticker: str) -> OrderSubmitResult:
    return OrderSubmitResult(
        success=True,
        broker_order_id=f"mock-{ticker}-001",
        status="SUBMITTED",
        rejection_reason=None,
        raw_response={"id": f"mock-{ticker}-001", "status": "accepted"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: 0 positions → no ticks, no broker call
# ─────────────────────────────────────────────────────────────────────────────

def test_no_positions_silent():
    """Daemon with 0 open positions returns empty tick list without errors."""
    mock_adapter = MagicMock()
    mock_adapter.get_positions.return_value = []
    mock_adapter.get_clock.return_value = _make_clock()

    from src.monitor.daemon import ActiveMonitorDaemon
    with (
        patch("src.monitor.daemon._get_position_db_data", return_value={}),
        patch("src.monitor.daemon._count_prior_partials", return_value=0),
        patch("src.monitor.daemon._log_tick_to_db"),
    ):
        daemon = ActiveMonitorDaemon(cycle_seconds=60, adapter=mock_adapter)
        ticks = daemon.run_tick()

    assert ticks == []
    mock_adapter.submit_order.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: position at +1.5×ATR → PARTIAL_EXIT → broker called
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_exit_at_1_5x_atr():
    """
    Position long AAPL, entry=100, current=103, ATR=2 → profit=3=1.5×ATR.
    evaluate_all_rules must fire partial_profit_take (33% off).
    Broker submit_order must be called once with action=SCALE_OUT.
    """
    atr = 2.0
    entry = 100.0
    current = 103.0   # profit = 3.0 = 1.5×ATR

    raw_pos = _make_raw_position(ticker="AAPL", side="long",
                                 qty=100.0, entry=entry, current=current)

    mock_adapter = MagicMock()
    mock_adapter.get_positions.return_value = [raw_pos]
    mock_adapter.get_clock.return_value = _make_clock(minutes_to_close=120.0)
    mock_adapter.submit_order.return_value = _make_submit_result("AAPL")

    import pandas as pd
    flat_ohlcv = pd.DataFrame({
        "open":   [entry] * 60,
        "high":   [entry + 0.1] * 60,
        "low":    [entry - 0.1] * 60,
        "close":  [current] * 60,
        "volume": [50000.0] * 60,
    })

    from src.monitor.daemon import ActiveMonitorDaemon
    with (
        patch("src.monitor.daemon._get_position_db_data", return_value={
            "stop_loss": 97.0, "take_profit": 106.0,
            "run_id": "test-run", "opened_at": None, "original_qty": 100,
        }),
        patch("src.monitor.daemon._count_prior_partials", return_value=0),
        patch("src.monitor.daemon._log_tick_to_db"),
        patch("src.monitor.price_checker.get_current_price", return_value=current),
        patch("src.monitor.price_checker.get_ohlcv_5min", return_value=flat_ohlcv),
        patch("src.monitor.price_checker.get_daily_atr", return_value=atr),
        patch("src.monitor.price_checker.get_avg_daily_volume", return_value=1_000_000.0),
    ):
        daemon = ActiveMonitorDaemon(cycle_seconds=60, adapter=mock_adapter)
        ticks = daemon.run_tick()

    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.ticker == "AAPL"
    assert tick.decision == "PARTIAL_EXIT"
    assert tick.action_taken is True
    assert tick.rule == "partial_1x_atr"

    # Broker must have been called
    mock_adapter.submit_order.assert_called_once()
    submitted_order = mock_adapter.submit_order.call_args[0][0]
    assert submitted_order.action == "SCALE_OUT"
    assert submitted_order.ticker == "AAPL"
    # 33% of 100 shares = 33
    assert submitted_order.quantity == 33


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: kill switch stops daemon within 1 tick
# ─────────────────────────────────────────────────────────────────────────────

def test_kill_switch(tmp_path, monkeypatch):
    """Daemon exits cleanly when .kill_monitor exists before the cycle check."""
    import src.monitor.daemon as daemon_mod

    kill_path = tmp_path / ".kill_monitor"
    kill_path.touch()
    monkeypatch.setattr(daemon_mod, "KILL_SWITCH_PATH", kill_path)

    mock_adapter = MagicMock()
    mock_adapter.get_positions.return_value = []
    mock_adapter.get_clock.return_value = _make_clock()

    with pytest.raises(SystemExit) as exc_info:
        daemon = daemon_mod.ActiveMonitorDaemon(cycle_seconds=60, adapter=mock_adapter)
        daemon.run()

    assert exc_info.value.code == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: HOLD position → no broker call, tick logged with HOLD
# ─────────────────────────────────────────────────────────────────────────────

def test_hold_no_broker_call():
    """Position with no signal → HOLD, broker never called."""
    atr = 2.0
    entry = 100.0
    current = 100.5   # profit = 0.5 = 0.25×ATR → all rules pass, HOLD

    raw_pos = _make_raw_position(ticker="MSFT", side="long",
                                 qty=50.0, entry=entry, current=current)

    mock_adapter = MagicMock()
    mock_adapter.get_positions.return_value = [raw_pos]
    mock_adapter.get_clock.return_value = _make_clock(minutes_to_close=120.0)

    import pandas as pd
    flat = pd.DataFrame({
        "open":   [entry] * 30,
        "high":   [entry + 0.1] * 30,
        "low":    [entry - 0.1] * 30,
        "close":  [current] * 30,
        "volume": [50000.0] * 30,
    })

    from src.monitor.daemon import ActiveMonitorDaemon
    with (
        patch("src.monitor.daemon._get_position_db_data", return_value={}),
        patch("src.monitor.daemon._count_prior_partials", return_value=0),
        patch("src.monitor.daemon._log_tick_to_db"),
        patch("src.monitor.price_checker.get_current_price", return_value=current),
        patch("src.monitor.price_checker.get_ohlcv_5min", return_value=flat),
        patch("src.monitor.price_checker.get_daily_atr", return_value=atr),
        patch("src.monitor.price_checker.get_avg_daily_volume", return_value=500_000.0),
    ):
        daemon = ActiveMonitorDaemon(cycle_seconds=60, adapter=mock_adapter)
        ticks = daemon.run_tick()

    assert len(ticks) == 1
    assert ticks[0].decision == "HOLD"
    assert ticks[0].action_taken is False
    mock_adapter.submit_order.assert_not_called()
