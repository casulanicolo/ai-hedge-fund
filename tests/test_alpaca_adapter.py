"""
tests/test_alpaca_adapter.py
─────────────────────────────
Unit tests for AlpacaBrokerAdapter.
All tests inject a mock TradingClient — no real API calls made.

Requires: pip install 'alpaca-py>=0.30.0'
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

# Skip entire module if alpaca-py is not installed
pytest.importorskip("alpaca", reason="alpaca-py not installed — run: pip install 'alpaca-py>=0.30.0'")

from src.execution.alpaca_adapter import AlpacaBrokerAdapter, _is_retryable_error
from src.execution.orders import TradeOrder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_WHITELIST = {"AAPL", "MSFT", "TSLA", "NVDA"}


def _make_order(**overrides) -> TradeOrder:
    defaults = dict(
        ticker="AAPL",
        action="OPEN_LONG",
        quantity=10,
        notional_usd=1500.0,
        order_type="MARKET",
        limit_price=None,
        stop_loss=145.0,
        take_profit=165.0,
        time_in_force="DAY",
        conviction=0.75,
        weighted_conviction=0.02,
        regime_at_decision="RISK_ON",
        reasoning="Unit test order",
        agent_contributions={"technical_analyst_agent": 0.75},
        created_at=datetime.now(timezone.utc),
        run_id="test-run-alpha",
    )
    defaults.update(overrides)
    return TradeOrder(**defaults)


def _mock_adapter() -> AlpacaBrokerAdapter:
    return AlpacaBrokerAdapter(client=MagicMock(), whitelist=_WHITELIST)


class RateLimitError(Exception):
    """Simulates a 429 from Alpaca."""
    status_code = 429


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: OPEN_LONG → BUY market order with bracket
# ─────────────────────────────────────────────────────────────────────────────

def test_open_long_builds_buy_bracket_request():
    from alpaca.trading.enums import OrderClass, OrderSide

    adapter = _mock_adapter()
    order   = _make_order(action="OPEN_LONG", ticker="AAPL", quantity=10,
                          stop_loss=145.0, take_profit=165.0)

    request = adapter._order_to_alpaca_request(order)

    assert request.side        == OrderSide.BUY
    assert request.symbol      == "AAPL"
    assert request.qty         == 10
    assert request.order_class == OrderClass.BRACKET
    assert request.stop_loss.stop_price    == 145.0
    assert request.take_profit.limit_price == 165.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: OPEN_SHORT → SELL market order with bracket
# ─────────────────────────────────────────────────────────────────────────────

def test_open_short_builds_sell_bracket_request():
    from alpaca.trading.enums import OrderClass, OrderSide

    adapter = _mock_adapter()
    order   = _make_order(
        action="OPEN_SHORT", ticker="TSLA", quantity=5,
        stop_loss=265.0, take_profit=230.0,
    )

    request = adapter._order_to_alpaca_request(order)

    assert request.side        == OrderSide.SELL
    assert request.symbol      == "TSLA"
    assert request.order_class == OrderClass.BRACKET
    assert request.stop_loss.stop_price    == 265.0
    assert request.take_profit.limit_price == 230.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: No SL/TP → simple (non-bracket) order
# ─────────────────────────────────────────────────────────────────────────────

def test_open_long_no_sl_tp_is_simple_order():
    from alpaca.trading.enums import OrderClass

    adapter = _mock_adapter()
    order   = _make_order(action="OPEN_LONG", stop_loss=None, take_profit=None)

    request = adapter._order_to_alpaca_request(order)

    assert request.order_class == OrderClass.SIMPLE
    assert not hasattr(request, "stop_loss") or request.stop_loss is None
    assert not hasattr(request, "take_profit") or request.take_profit is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: SCALE_OUT → close_position called with partial qty
# ─────────────────────────────────────────────────────────────────────────────

def test_scale_out_calls_close_position_with_qty():
    from alpaca.trading.requests import ClosePositionRequest

    mock_client = MagicMock()
    mock_resp   = MagicMock()
    mock_resp.id     = "broker-order-123"
    mock_resp.status = "accepted"
    mock_client.close_position.return_value = mock_resp

    adapter = AlpacaBrokerAdapter(client=mock_client, whitelist=_WHITELIST)
    order   = _make_order(action="SCALE_OUT", ticker="AAPL", quantity=5,
                          stop_loss=None, take_profit=None, notional_usd=None)

    result = adapter.submit_order(order)

    mock_client.close_position.assert_called_once()
    kwargs = mock_client.close_position.call_args.kwargs
    close_options = kwargs.get("close_options")
    assert close_options is not None, "close_options should be passed for partial close"
    assert close_options.qty == "5"   # alpaca-py expects string

    assert result.success
    assert result.broker_order_id == "broker-order-123"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: 429 rate-limit → tenacity retries 3 times, succeeds on 3rd
# ─────────────────────────────────────────────────────────────────────────────

def test_retry_on_429_succeeds_after_two_failures():
    mock_client = MagicMock()

    ok_resp        = MagicMock()
    ok_resp.id     = "broker-order-456"
    ok_resp.status = "accepted"
    ok_resp.model_dump = MagicMock(return_value={"id": "broker-order-456", "status": "accepted"})

    rate_error = RateLimitError("Rate limit exceeded")

    # Fail twice, succeed on third attempt
    mock_client.submit_order.side_effect = [rate_error, rate_error, ok_resp]

    adapter = AlpacaBrokerAdapter(client=mock_client, whitelist=_WHITELIST)
    order   = _make_order(action="OPEN_LONG", ticker="AAPL", quantity=5)

    result = adapter.submit_order(order)

    assert mock_client.submit_order.call_count == 3
    assert result.success
    assert result.broker_order_id == "broker-order-456"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: ticker not in whitelist → SKIPPED, no API call
# ─────────────────────────────────────────────────────────────────────────────

def test_non_whitelisted_ticker_is_skipped():
    mock_client = MagicMock()
    adapter     = AlpacaBrokerAdapter(client=mock_client, whitelist={"AAPL"})
    order       = _make_order(ticker="GOOGL")   # not in whitelist

    result = adapter.submit_order(order)

    mock_client.submit_order.assert_not_called()
    assert result.status == "SKIPPED"
    assert not result.success


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: HOLD action → immediate SKIPPED, no API call
# ─────────────────────────────────────────────────────────────────────────────

def test_hold_order_is_no_op():
    mock_client = MagicMock()
    adapter     = AlpacaBrokerAdapter(client=mock_client, whitelist=_WHITELIST)
    order       = _make_order(
        action="HOLD", quantity=None, notional_usd=None,
        stop_loss=None, take_profit=None,
    )

    result = adapter.submit_order(order)

    mock_client.submit_order.assert_not_called()
    mock_client.close_position.assert_not_called()
    assert result.status == "SKIPPED"
    assert result.success


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: live URL check → RuntimeError
# ─────────────────────────────────────────────────────────────────────────────

def test_live_url_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL",    "https://api.alpaca.markets")
    monkeypatch.setenv("ALPACA_API_KEY_ID",  "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")

    with pytest.raises(RuntimeError, match="paper"):
        AlpacaBrokerAdapter()   # no mock client → reads env vars → should raise


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: _is_retryable_error predicate
# ─────────────────────────────────────────────────────────────────────────────

def test_is_retryable_error():
    err_429 = RateLimitError("rate limit")
    err_503 = Exception("service unavailable")
    err_503.status_code = 503
    err_400 = Exception("bad request")
    err_400.status_code = 400

    assert _is_retryable_error(err_429) is True
    assert _is_retryable_error(err_503) is True
    assert _is_retryable_error(err_400) is False
    assert _is_retryable_error(ValueError("oops")) is False
