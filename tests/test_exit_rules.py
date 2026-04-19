"""
tests/test_exit_rules.py
─────────────────────────
Unit tests for src/monitor/exit_rules.py — 2 cases per rule (trigger + no-trigger).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.monitor.exit_rules import (
    evaluate_end_of_day,
    evaluate_partial_profit_take,
    evaluate_setup_broken,
    evaluate_time_decay,
    evaluate_trailing_stop,
)
from src.monitor.monitor_state import EnrichedPosition


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pos(
    ticker="AAPL",
    side="long",
    entry=100.0,
    current=100.0,
    qty=100.0,
    stop_loss=None,
    take_profit=None,
    days_open=1.0,
    prior_partials_count=0,
) -> EnrichedPosition:
    return EnrichedPosition(
        ticker=ticker,
        side=side,
        qty=qty,
        avg_entry_price=entry,
        market_value=current * qty,
        unrealized_pl=0.0,
        unrealized_pl_pct=0.0,
        current_price=current,
        stop_loss=stop_loss,
        take_profit=take_profit,
        days_open=days_open,
        prior_partials_count=prior_partials_count,
    )


def _make_clock(minutes_to_close: float = 9999.0, is_open: bool = True):
    clock = MagicMock()
    clock.is_open = is_open
    if minutes_to_close < 9999:
        clock.next_close = datetime.now(timezone.utc) + timedelta(minutes=minutes_to_close)
    else:
        clock.next_close = None
    return clock


def _make_ohlcv(closes: list[float], volume_per_bar: float = 1000.0) -> pd.DataFrame:
    """Build a minimal 5min OHLCV DataFrame."""
    n = len(closes)
    return pd.DataFrame({
        "open":   closes,
        "high":   [c + 0.1 for c in closes],
        "low":    [c - 0.1 for c in closes],
        "close":  closes,
        "volume": [volume_per_bar] * n,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Rule 1 — evaluate_trailing_stop
# ─────────────────────────────────────────────────────────────────────────────

class TestTrailingStop:
    def test_trigger_moves_to_breakeven(self):
        """profit = 1.5×ATR → SL moves to breakeven (entry)."""
        atr = 2.0
        pos = _make_pos(entry=100.0, current=103.0)   # profit = 3 = 1.5×ATR
        decision = evaluate_trailing_stop(pos, atr)
        assert decision is not None
        assert decision.action == "ADJUST_TRAILING_STOP"
        assert decision.new_stop == pytest.approx(100.0, abs=0.01)  # breakeven
        assert decision.rule == "trailing_breakeven"

    def test_no_trigger_below_threshold(self):
        """profit = 0.5×ATR → no trailing stop update."""
        atr = 2.0
        pos = _make_pos(entry=100.0, current=101.0)   # profit = 1 = 0.5×ATR
        decision = evaluate_trailing_stop(pos, atr)
        assert decision is None


# ─────────────────────────────────────────────────────────────────────────────
# Rule 2 — evaluate_partial_profit_take
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialProfitTake:
    def test_trigger_first_partial(self):
        """profit >= 1×ATR, no prior partials → take 33%."""
        atr = 2.0
        pos = _make_pos(entry=100.0, current=102.0, prior_partials_count=0)
        decision = evaluate_partial_profit_take(
            pos, entry=100.0, atr=atr, current_price=102.0, prior_partials_count=0
        )
        assert decision is not None
        assert decision.action == "PARTIAL_EXIT"
        assert decision.percentage == pytest.approx(0.33)
        assert decision.rule == "partial_1x_atr"

    def test_no_trigger_insufficient_profit(self):
        """profit = 0.4×ATR → no partial exit."""
        atr = 2.0
        pos = _make_pos(entry=100.0, current=100.8, prior_partials_count=0)
        decision = evaluate_partial_profit_take(
            pos, entry=100.0, atr=atr, current_price=100.8, prior_partials_count=0
        )
        assert decision is None


# ─────────────────────────────────────────────────────────────────────────────
# Rule 3 — evaluate_setup_broken
# ─────────────────────────────────────────────────────────────────────────────

class TestSetupBroken:
    def _rsi_drop_ohlcv(self) -> pd.DataFrame:
        """25 bars: first 21 rising → RSI high; last 4 sharply dropping → RSI drops >20 pts."""
        up   = list(range(100, 121))              # 21 bars rising
        down = [120 - i * 3 for i in range(4)]   # 4 bars sharply falling
        return _make_ohlcv(up + down)

    def test_trigger_rsi_inversion_long(self):
        """Long + RSI drops >20 pts in 15min → FULL_EXIT."""
        pos = _make_pos(side="long")
        df  = self._rsi_drop_ohlcv()
        decision = evaluate_setup_broken(pos, df)
        assert decision is not None
        assert decision.action == "FULL_EXIT"
        assert decision.rule == "setup_broken_rsi"

    def test_no_trigger_stable_rsi(self):
        """Flat price → RSI stable → no exit."""
        pos = _make_pos(side="long")
        flat = _make_ohlcv([100.0] * 30)
        decision = evaluate_setup_broken(pos, flat)
        assert decision is None


# ─────────────────────────────────────────────────────────────────────────────
# Rule 4 — evaluate_time_decay
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeDecay:
    def test_trigger_stale_position(self):
        """Open 6 days, profit = 0.2×ATR (<0.5 threshold) → FULL_EXIT."""
        atr = 2.0
        pos = _make_pos(entry=100.0, current=100.4, days_open=6.0)
        decision = evaluate_time_decay(pos, atr)
        assert decision is not None
        assert decision.action == "FULL_EXIT"
        assert decision.rule == "time_decay"

    def test_no_trigger_young_position(self):
        """Open 3 days → time decay rule not applicable."""
        atr = 2.0
        pos = _make_pos(entry=100.0, current=100.4, days_open=3.0)
        decision = evaluate_time_decay(pos, atr)
        assert decision is None


# ─────────────────────────────────────────────────────────────────────────────
# Rule 5 — evaluate_end_of_day
# ─────────────────────────────────────────────────────────────────────────────

class TestEndOfDay:
    def test_trigger_near_close_stale_signals(self):
        """10 min to close, signals 5h old, profit < 0.2×ATR → FULL_EXIT."""
        atr = 2.0
        pos = _make_pos(entry=100.0, current=100.3)  # profit = 0.3 = 0.15×ATR < 0.2
        clock = _make_clock(minutes_to_close=10.0)
        decision = evaluate_end_of_day(pos, clock, run_id_age_hours=5.0, atr=atr)
        assert decision is not None
        assert decision.action == "FULL_EXIT"
        assert decision.rule == "end_of_day"

    def test_no_trigger_far_from_close(self):
        """60 min to close → EOD rule not applicable."""
        atr = 2.0
        pos = _make_pos(entry=100.0, current=100.3)
        clock = _make_clock(minutes_to_close=60.0)
        decision = evaluate_end_of_day(pos, clock, run_id_age_hours=5.0, atr=atr)
        assert decision is None
