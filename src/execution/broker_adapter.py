"""
src/execution/broker_adapter.py
────────────────────────────────
Protocol interface for broker adapters + shared Pydantic snapshots.

Implementations: AlpacaBrokerAdapter (this phase), IBKRBrokerAdapter (future).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from src.execution.orders import TradeOrder


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot models (broker-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

class AccountSnapshot(BaseModel):
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    last_equity: float = 0.0
    status: str          # "ACTIVE" | "ACCOUNT_CLOSED" | etc.


class PositionSnapshot(BaseModel):
    ticker: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    side: str            # "long" | "short"


class OrderSnapshot(BaseModel):
    broker_order_id: str
    ticker: str
    action: str
    qty: float | None
    notional: float | None
    status: str
    filled_qty: float
    filled_avg_price: float | None
    submitted_at: datetime | None
    filled_at: datetime | None


class OrderSubmitResult(BaseModel):
    success: bool
    broker_order_id: str | None = None
    status: str                          # SUBMITTED | FILLED | SKIPPED | REJECTED
    rejection_reason: str | None = None
    raw_response: dict = {}


class ClockInfo(BaseModel):
    is_open: bool
    next_open: datetime | None
    next_close: datetime | None
    timestamp: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class BrokerAdapter(Protocol):
    def get_account(self) -> AccountSnapshot: ...
    def get_positions(self) -> list[PositionSnapshot]: ...
    def get_open_orders(self) -> list[OrderSnapshot]: ...
    def submit_order(self, order: TradeOrder) -> OrderSubmitResult: ...
    def cancel_order(self, broker_order_id: str) -> bool: ...
    def close_position(self, ticker: str, qty: int | None = None) -> OrderSubmitResult: ...
    def is_market_open(self) -> bool: ...
    def get_clock(self) -> ClockInfo: ...
