"""
src/execution/orders.py
───────────────────────
Structured trade order contract for Athanor Alpha execution layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TradeOrder(BaseModel):
    ticker: str
    action: Literal["OPEN_LONG", "OPEN_SHORT", "CLOSE", "SCALE_IN", "SCALE_OUT", "ADJUST_SL", "ADJUST_TP", "HOLD"]
    quantity: int | None = None           # shares; None for HOLD / ADJUST
    notional_usd: float | None = None     # dollar amount; required for OPEN/SCALE
    order_type: Literal["MARKET", "LIMIT"]
    limit_price: float | None = None
    stop_loss: float | None = None        # absolute price, not %
    take_profit: float | None = None
    time_in_force: Literal["DAY", "GTC"]
    conviction: float = Field(ge=0.0, le=1.0)
    weighted_conviction: float
    regime_at_decision: Literal["RISK_ON", "CAUTION", "RISK_OFF"]
    reasoning: str = Field(max_length=500)
    agent_contributions: dict[str, float]  # {agent_id: signed_score}
    created_at: datetime
    run_id: str


def order_to_dict(order: TradeOrder) -> dict:
    """Serialize TradeOrder to JSON-safe dict (for logs/runs.jsonl)."""
    return order.model_dump(mode="json")
