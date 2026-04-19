"""
src/monitor/monitor_state.py
──────────────────────────────
Pydantic models for the active monitor daemon.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

from src.execution.broker_adapter import PositionSnapshot


class ExitDecision(BaseModel):
    action: Literal["HOLD", "PARTIAL_EXIT", "FULL_EXIT", "ADJUST_TRAILING_STOP"]
    percentage: Optional[float] = None   # for PARTIAL_EXIT: fraction of position (0-1)
    new_stop: Optional[float] = None     # for ADJUST_TRAILING_STOP: absolute price
    reason: str
    rule: str                            # which rule triggered (for audit)


class MonitorTick(BaseModel):
    timestamp: datetime
    ticker: str
    current_price: Optional[float]
    unrealized_pl_pct: Optional[float]
    decision: str               # HOLD | PARTIAL_EXIT | FULL_EXIT | ADJUST_TRAILING_STOP
    reason: str
    action_taken: bool
    broker_order_id: Optional[str] = None


class EnrichedPosition(PositionSnapshot):
    """
    PositionSnapshot augmented with local DB data needed by exit rules.
    current_price is derived from market_value / qty when not explicitly set.
    """
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: Optional[datetime] = None
    run_id: Optional[str] = None
    days_open: float = 0.0
    original_qty: Optional[float] = None   # qty at open; tracks partial exit history
    prior_partials_count: int = 0          # how many PARTIAL_EXIT ticks today

    def effective_price(self) -> float:
        """Best available current price."""
        if self.current_price and self.current_price > 0:
            return self.current_price
        if self.qty and self.qty != 0:
            return abs(self.market_value / self.qty)
        return self.avg_entry_price

    def profit_dollars(self) -> float:
        sign = 1.0 if self.side == "long" else -1.0
        return sign * (self.effective_price() - self.avg_entry_price) * abs(self.qty)

    def profit_per_share(self) -> float:
        sign = 1.0 if self.side == "long" else -1.0
        return sign * (self.effective_price() - self.avg_entry_price)
