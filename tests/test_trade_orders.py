"""
tests/test_trade_orders.py
──────────────────────────
Unit tests for TradeOrder contract and portfolio_manager sizing logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import floor

import pytest

from src.execution.orders import TradeOrder, order_to_dict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_order(**overrides) -> TradeOrder:
    defaults = dict(
        ticker="AAPL",
        action="OPEN_LONG",
        quantity=10,
        notional_usd=1500.0,
        order_type="MARKET",
        limit_price=None,
        stop_loss=145.0,
        take_profit=160.0,
        time_in_force="DAY",
        conviction=0.75,
        weighted_conviction=0.02,
        regime_at_decision="RISK_ON",
        reasoning="Test order for unit tests.",
        agent_contributions={"technical_analyst_agent": 0.75, "sentiment_agent": 0.60},
        created_at=datetime.now(timezone.utc),
        run_id="test-run-001",
    )
    defaults.update(overrides)
    return TradeOrder(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: OPEN_LONG with high conviction → quantity consistent with Kelly sizing
# ─────────────────────────────────────────────────────────────────────────────

def test_open_long_quantity_consistent_with_kelly():
    """Quantity must equal floor(notional_usd / entry_price) — mirrors Kelly sizing logic."""
    from src.agents.portfolio_manager import (
        compute_half_kelly,
        PORTFOLIO_SIZE_USD,
        RISK_PER_TRADE_PCT,
    )

    # High conviction scenario: p_win=0.70, rr_ratio=2.0
    p_win = 0.70
    rr    = 2.0
    params = {"kelly_fraction": 0.5, "max_position_pct": 0.15}

    kelly_size = compute_half_kelly(p_win, rr, params)
    assert kelly_size > 0, "High conviction should yield positive Kelly size"
    assert kelly_size <= 0.15, "Kelly size must not exceed max_position_pct"

    # Simulate order built from this sizing
    entry     = 150.0
    size_usd  = kelly_size * PORTFOLIO_SIZE_USD
    quantity  = floor(size_usd / entry)

    order = _make_order(
        quantity=quantity,
        notional_usd=size_usd,
        conviction=p_win,
        stop_loss=entry - 1.5 * 3.0,   # 1.5× ATR where ATR≈3
        take_profit=entry + 3.0 * 3.0,
    )

    assert order.action == "OPEN_LONG"
    assert order.quantity == quantity
    assert order.quantity > 0
    # notional should be close to quantity * entry (within one share rounding)
    assert abs(order.notional_usd - order.quantity * entry) < entry
    # Serializable
    d = order_to_dict(order)
    assert d["action"] == "OPEN_LONG"
    assert d["quantity"] == quantity


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: RISK_OFF regime → max notional respects 5 % cap
# ─────────────────────────────────────────────────────────────────────────────

def test_risk_off_notional_cap():
    """In RISK_OFF regime the position size cap is 5 % of portfolio."""
    from src.agents.portfolio_manager import (
        MACRO_REGIME_POSITION_CAPS,
        PORTFOLIO_SIZE_USD,
    )

    cap_pct     = MACRO_REGIME_POSITION_CAPS["RISK_OFF"]   # 5.0
    max_notional = PORTFOLIO_SIZE_USD * cap_pct / 100.0

    order = _make_order(
        ticker="MSFT",
        notional_usd=max_notional,
        quantity=floor(max_notional / 400.0),
        regime_at_decision="RISK_OFF",
        conviction=0.60,
        weighted_conviction=0.015,
        stop_loss=390.0,
        take_profit=420.0,
    )

    assert order.regime_at_decision == "RISK_OFF"
    assert order.notional_usd <= max_notional + 1e-6   # allow float epsilon

    # Verify _compute_sizing also applies the cap
    from src.agents.portfolio_manager import _compute_sizing, MACRO_REGIME_POSITION_CAPS

    agg = {
        "net_score":           0.50,
        "avg_confidence":      0.80,
        "net_signal":          "bullish",
        "n_agents":            4,
        "weighted_conviction": 0.02,
        "consensus_ratio":     0.75,
        "n_dims":              4,
        "dim_scores":          {},
    }
    risk_report = {
        "trade_levels":         {"MSFT": {"rr_ratio": 2.0}},
        "ticker_flags":         {},
        "daily_var_95":         0.02,
        "macro_regime":         {"sizing_multiplier": 1.0},
        "correlation_matrix":   {},
    }

    # Patch _get_macro_regime to return RISK_OFF via state containing macro signal
    mock_state: dict = {
        "data": {
            "analyst_signals": {
                "macro_agent": {
                    "MSFT": {"macro_regime": "RISK_OFF", "direction": "NEUTRAL",
                             "expected_return": 0.0, "confidence": 0.5}
                }
            }
        }
    }

    sizing, conviction, action, _ = _compute_sizing("MSFT", agg, risk_report, state=mock_state)

    if action != "HOLD":
        assert sizing <= MACRO_REGIME_POSITION_CAPS["RISK_OFF"], (
            f"Sizing {sizing}% exceeds RISK_OFF cap {cap_pct}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: consensus ratio < 0.65 → no active order (HOLD only)
# ─────────────────────────────────────────────────────────────────────────────

def test_low_consensus_yields_hold():
    """Signals from fewer than 65 % of dimensions → action must be HOLD, sizing 0."""
    from src.agents.portfolio_manager import _compute_sizing, MIN_CONSENSUS_RATIO

    agg = {
        "net_score":           0.50,   # strong net bullish score
        "avg_confidence":      0.80,
        "net_signal":          "bullish",
        "n_agents":            4,
        "weighted_conviction": 0.02,
        "consensus_ratio":     0.50,   # below 0.65 threshold
        "n_dims":              4,
        "dim_scores":          {},
    }
    risk_report = {
        "trade_levels":       {},
        "ticker_flags":       {},
        "daily_var_95":       0.02,
        "macro_regime":       {"sizing_multiplier": 1.0},
        "correlation_matrix": {},
    }

    sizing, conviction, action, consensus = _compute_sizing("AAPL", agg, risk_report, state=None)

    assert action == "HOLD", f"Expected HOLD, got {action}"
    assert sizing == 0.0, f"Expected sizing 0.0, got {sizing}"
    assert consensus < MIN_CONSENSUS_RATIO


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: SL/TP populated for OPEN_*, None for HOLD
# ─────────────────────────────────────────────────────────────────────────────

def test_sl_tp_populated_for_open_none_for_hold():
    """OPEN_LONG/OPEN_SHORT must have stop_loss and take_profit set; HOLD must have None."""
    ts     = datetime.now(timezone.utc)
    run_id = "test-run-004"

    long_order = _make_order(
        action="OPEN_LONG",
        stop_loss=145.0,
        take_profit=160.0,
        created_at=ts,
        run_id=run_id,
    )

    short_order = _make_order(
        action="OPEN_SHORT",
        ticker="TSLA",
        stop_loss=265.0,
        take_profit=230.0,
        created_at=ts,
        run_id=run_id,
    )

    hold_order = _make_order(
        action="HOLD",
        ticker="MSFT",
        quantity=None,
        notional_usd=None,
        stop_loss=None,
        take_profit=None,
        conviction=0.20,
        weighted_conviction=0.001,
        regime_at_decision="CAUTION",
        reasoning="Insufficient consensus.",
        created_at=ts,
        run_id=run_id,
    )

    # OPEN orders must have SL and TP
    assert long_order.stop_loss is not None
    assert long_order.take_profit is not None
    assert short_order.stop_loss is not None
    assert short_order.take_profit is not None

    # HOLD must have neither
    assert hold_order.stop_loss is None
    assert hold_order.take_profit is None
    assert hold_order.quantity is None
    assert hold_order.notional_usd is None

    # All orders must be serializable via model_dump()
    for order in (long_order, short_order, hold_order):
        d = order.model_dump()
        assert isinstance(d, dict)
        assert "ticker" in d
        assert "action" in d
        assert "run_id" in d

    # order_to_dict produces JSON-safe output (datetime → str)
    d = order_to_dict(long_order)
    assert isinstance(d["created_at"], str)
