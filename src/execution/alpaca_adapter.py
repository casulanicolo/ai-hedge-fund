"""
src/execution/alpaca_adapter.py
────────────────────────────────
AlpacaBrokerAdapter — wraps alpaca-py TradingClient.

Safety rules (hard-coded, non-removable via env):
  1. ALPACA_BASE_URL must contain "paper" → RuntimeError otherwise.
  2. Only tickers in config/tickers.yaml are accepted → others are skipped.
  3. Crypto tickers (*-USD) require a separate CryptoTradingClient → skipped here.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.execution.broker_adapter import (
    AccountSnapshot,
    ClockInfo,
    OrderSnapshot,
    OrderSubmitResult,
    PositionSnapshot,
)
from src.execution.orders import TradeOrder

logger = logging.getLogger(__name__)

_ALPACA_LOG = pathlib.Path("logs/alpaca.jsonl")
_TICKERS_CONFIG = pathlib.Path("config/tickers.yaml")

# ─────────────────────────────────────────────────────────────────────────────
# Retry predicate
# ─────────────────────────────────────────────────────────────────────────────

def _is_retryable_error(exc: BaseException) -> bool:
    """Retry on 429 (rate-limit) and 503 (service unavailable) only."""
    return getattr(exc, "status_code", 0) in (429, 503)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _log_api(
    action: str,
    request_data: dict,
    response_data: dict,
    latency_ms: float,
) -> None:
    entry = {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "action":      action,
        "latency_ms":  round(latency_ms, 1),
        "request":     request_data,
        "response":    response_data,
    }
    _ALPACA_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _ALPACA_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _serialize_resp(resp) -> dict:
    if hasattr(resp, "model_dump"):
        result = resp.model_dump(mode="json")
        if isinstance(result, dict):
            return result
    if hasattr(resp, "dict"):
        result = resp.dict()
        if isinstance(result, dict):
            return result
    return {"id": str(getattr(resp, "id", "")), "status": str(getattr(resp, "status", ""))}


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────────────────────

def _with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on 429/503 (max 3 attempts)."""
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable_error),
        reraise=True,
    ):
        with attempt:
            return fn(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Whitelist loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_whitelist() -> set[str]:
    try:
        import yaml
        with _TICKERS_CONFIG.open(encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f)
        tickers = cfg.get("tickers", []) if isinstance(cfg, dict) else (cfg or [])
        result = {str(t).strip() for t in tickers if t}
        logger.info("[alpaca_adapter] Whitelist loaded: %s", sorted(result))
        return result
    except FileNotFoundError:
        logger.warning("[alpaca_adapter] config/tickers.yaml not found — allowing all tickers")
        return set()   # empty set = allow all (see _is_whitelisted)
    except Exception as exc:
        logger.error("[alpaca_adapter] Failed to load whitelist: %s — allowing all", exc)
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaBrokerAdapter:
    """
    Broker adapter for Alpaca paper trading via alpaca-py.

    Parameters
    ----------
    client   : inject a TradingClient (or mock) — if None, built from env vars.
    whitelist: inject allowed tickers set — if None, loaded from config/tickers.yaml.
               Empty set means "allow all" (used when file is missing).
    """

    def __init__(
        self,
        client=None,
        whitelist: Optional[set[str]] = None,
    ):
        # ── Safety: paper-only check ────────────────────────────────────────
        if client is None:
            base_url = os.getenv("ALPACA_BASE_URL", "")
            if "paper" not in base_url.lower():
                raise RuntimeError(
                    "Refusing to run on live account — "
                    "ALPACA_BASE_URL must contain 'paper'. "
                    f"Got: {base_url!r}"
                )
            api_key    = os.getenv("ALPACA_API_KEY_ID", "")
            api_secret = os.getenv("ALPACA_API_SECRET_KEY", "")
            if not api_key or not api_secret:
                raise RuntimeError(
                    "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY must be set."
                )
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=True,
            )
        else:
            self._client = client

        # ── Whitelist ────────────────────────────────────────────────────────
        self._whitelist: set[str] = whitelist if whitelist is not None else _load_whitelist()

    # ── Whitelist guard ──────────────────────────────────────────────────────

    def _is_whitelisted(self, ticker: str) -> bool:
        if not self._whitelist:        # empty = allow all
            return True
        return ticker in self._whitelist

    # ── Order request builder ────────────────────────────────────────────────

    def _order_to_alpaca_request(self, order: TradeOrder):
        """Convert a TradeOrder to an alpaca-py OrderRequest. Pure, no side effects."""
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        side = (
            OrderSide.BUY
            if order.action in ("OPEN_LONG", "SCALE_IN")
            else OrderSide.SELL
        )
        tif = (
            TimeInForce.DAY if order.time_in_force == "DAY" else TimeInForce.GTC
        )

        has_bracket = (
            order.stop_loss is not None
            and order.take_profit is not None
            and order.quantity is not None   # bracket requires explicit qty
        )
        order_class = OrderClass.BRACKET if has_bracket else OrderClass.SIMPLE

        kwargs: dict = dict(
            symbol=order.ticker,
            side=side,
            time_in_force=tif,
            order_class=order_class,
        )

        if order.quantity is not None:
            kwargs["qty"] = order.quantity
        elif order.notional_usd is not None:
            kwargs["notional"] = order.notional_usd
            # Can't use bracket without explicit qty — downgrade
            if has_bracket:
                kwargs["order_class"] = OrderClass.SIMPLE
                has_bracket = False
                logger.warning(
                    "[alpaca_adapter] %s: bracket removed (no qty, notional-only order)",
                    order.ticker,
                )

        if has_bracket:
            kwargs["take_profit"] = TakeProfitRequest(limit_price=order.take_profit)
            kwargs["stop_loss"]   = StopLossRequest(stop_price=order.stop_loss)

        if order.order_type == "LIMIT" and order.limit_price is not None:
            kwargs["limit_price"] = order.limit_price
            return LimitOrderRequest(**kwargs)

        return MarketOrderRequest(**kwargs)

    # ── Internal order submission with retry ─────────────────────────────────

    def _do_submit(self, alpaca_request):
        return _with_retry(self._client.submit_order, order_data=alpaca_request)

    # ── BrokerAdapter interface ──────────────────────────────────────────────

    def get_account(self) -> AccountSnapshot:
        raw = self._client.get_account()
        return AccountSnapshot(
            account_id=str(raw.account_number),
            buying_power=float(raw.buying_power),
            cash=float(raw.cash),
            portfolio_value=float(raw.portfolio_value),
            status=str(raw.status),
        )

    def get_positions(self) -> list[PositionSnapshot]:
        positions = self._client.get_all_positions()
        return [
            PositionSnapshot(
                ticker=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                unrealized_pl=float(p.unrealized_pl),
                side=str(p.side),
            )
            for p in positions
        ]

    def get_open_orders(self) -> list[OrderSnapshot]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        raw_orders = self._client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        return [
            OrderSnapshot(
                broker_order_id=str(o.id),
                ticker=o.symbol,
                action=str(o.side),
                qty=float(o.qty) if o.qty else None,
                notional=float(o.notional) if o.notional else None,
                status=str(o.status),
                filled_qty=float(o.filled_qty or 0),
                filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                submitted_at=o.submitted_at,
                filled_at=o.filled_at,
            )
            for o in raw_orders
        ]

    def get_clock(self) -> ClockInfo:
        clock = self._client.get_clock()
        return ClockInfo(
            is_open=clock.is_open,
            next_open=clock.next_open,
            next_close=clock.next_close,
            timestamp=clock.timestamp,
        )

    def is_market_open(self) -> bool:
        return self.get_clock().is_open

    def cancel_order(self, broker_order_id: str) -> bool:
        try:
            import uuid
            self._client.cancel_order_by_id(uuid.UUID(broker_order_id))
            return True
        except Exception as exc:
            logger.error("[alpaca_adapter] cancel_order %s failed: %s", broker_order_id, exc)
            return False

    def close_position(self, ticker: str, qty: int | None = None) -> OrderSubmitResult:
        from alpaca.trading.requests import ClosePositionRequest

        t0 = time.perf_counter()
        try:
            if qty is not None:
                resp = _with_retry(
                    self._client.close_position,
                    symbol_or_asset_id=ticker,
                    close_options=ClosePositionRequest(qty=str(qty)),
                )
            else:
                resp = _with_retry(
                    self._client.close_position,
                    symbol_or_asset_id=ticker,
                )

            latency = (time.perf_counter() - t0) * 1000
            resp_dict = _serialize_resp(resp)
            _log_api("CLOSE_POSITION", {"ticker": ticker, "qty": qty}, resp_dict, latency)

            return OrderSubmitResult(
                success=True,
                broker_order_id=str(resp.id) if hasattr(resp, "id") else None,
                status=str(resp.status) if hasattr(resp, "status") else "SUBMITTED",
                raw_response=resp_dict,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            _log_api("CLOSE_POSITION", {"ticker": ticker, "qty": qty}, {"error": str(exc)}, latency)
            logger.error("[alpaca_adapter] close_position %s failed: %s", ticker, exc)
            return OrderSubmitResult(
                success=False,
                broker_order_id=None,
                status="REJECTED",
                rejection_reason=str(exc),
                raw_response={},
            )

    def submit_order(self, order: TradeOrder) -> OrderSubmitResult:
        """Main entry point. Routes TradeOrder to the correct Alpaca API call."""

        # ── HOLD: no-op ──────────────────────────────────────────────────────
        if order.action == "HOLD":
            return OrderSubmitResult(
                success=True, broker_order_id=None,
                status="SKIPPED", raw_response={},
            )

        # ── Crypto: skip (equity TradingClient only) ─────────────────────────
        if order.ticker.endswith("-USD"):
            logger.warning(
                "[alpaca_adapter] Crypto ticker %s skipped — use CryptoTradingClient",
                order.ticker,
            )
            return OrderSubmitResult(
                success=False, broker_order_id=None, status="SKIPPED",
                rejection_reason="Crypto tickers require CryptoTradingClient (Phase 3)",
            )

        # ── Whitelist guard ──────────────────────────────────────────────────
        if not self._is_whitelisted(order.ticker):
            logger.error(
                "[alpaca_adapter] Ticker %s not in whitelist — skipping order",
                order.ticker,
            )
            return OrderSubmitResult(
                success=False, broker_order_id=None, status="SKIPPED",
                rejection_reason=f"Ticker {order.ticker} not in whitelist",
            )

        t0 = time.perf_counter()
        req_data = {"ticker": order.ticker, "action": order.action, "qty": order.quantity}

        try:
            # ── CLOSE (full position) ────────────────────────────────────────
            if order.action == "CLOSE":
                return self.close_position(order.ticker)

            # ── SCALE_OUT (partial close) ────────────────────────────────────
            if order.action == "SCALE_OUT":
                return self.close_position(order.ticker, order.quantity)

            # ── ADJUST_SL / ADJUST_TP ────────────────────────────────────────
            if order.action in ("ADJUST_SL", "ADJUST_TP"):
                return self._adjust_bracket_leg(order)

            # ── OPEN_LONG / OPEN_SHORT / SCALE_IN ────────────────────────────
            if order.action in ("OPEN_LONG", "OPEN_SHORT", "SCALE_IN"):
                if order.quantity is None and order.notional_usd is None:
                    return OrderSubmitResult(
                        success=False, broker_order_id=None, status="REJECTED",
                        rejection_reason="Neither qty nor notional set",
                    )
                alpaca_req = self._order_to_alpaca_request(order)
                resp = self._do_submit(alpaca_req)

                latency = (time.perf_counter() - t0) * 1000
                resp_dict = _serialize_resp(resp)
                _log_api(order.action, req_data, resp_dict, latency)

                return OrderSubmitResult(
                    success=True,
                    broker_order_id=str(resp.id),
                    status=str(resp.status),
                    raw_response=resp_dict,
                )

            raise ValueError(f"Unhandled action: {order.action}")

        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            _log_api(order.action, req_data, {"error": str(exc)}, latency)
            logger.error("[alpaca_adapter] submit_order %s %s failed: %s",
                         order.action, order.ticker, exc)
            return OrderSubmitResult(
                success=False, broker_order_id=None, status="REJECTED",
                rejection_reason=str(exc), raw_response={},
            )

    # ── ADJUST bracket leg ───────────────────────────────────────────────────

    def _adjust_bracket_leg(self, order: TradeOrder) -> OrderSubmitResult:
        """
        Find open child orders for ticker and replace the SL or TP price.
        Best-effort: if no child order found, returns failure.
        """
        import uuid as _uuid
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest

        try:
            open_orders = self._client.get_orders(
                filter=GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    symbols=[order.ticker],
                )
            )
            target_type = "stop" if order.action == "ADJUST_SL" else "limit"
            new_price   = order.stop_loss if order.action == "ADJUST_SL" else order.take_profit

            for o in open_orders:
                if o.type and target_type in str(o.type).lower():
                    replace_req = ReplaceOrderRequest(
                        stop_price=(new_price if order.action == "ADJUST_SL" else None),
                        limit_price=(new_price if order.action == "ADJUST_TP" else None),
                    )
                    resp = _with_retry(
                        self._client.replace_order_by_id,
                        _uuid.UUID(str(o.id)),
                        replace_req,
                    )
                    return OrderSubmitResult(
                        success=True,
                        broker_order_id=str(resp.id),
                        status=str(resp.status),
                        raw_response=_serialize_resp(resp),
                    )

            return OrderSubmitResult(
                success=False, broker_order_id=None, status="REJECTED",
                rejection_reason=f"No open {target_type} order found for {order.ticker}",
            )
        except Exception as exc:
            return OrderSubmitResult(
                success=False, broker_order_id=None, status="REJECTED",
                rejection_reason=str(exc),
            )
