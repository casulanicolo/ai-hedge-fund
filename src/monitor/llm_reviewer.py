"""
src/monitor/llm_reviewer.py
────────────────────────────
Optional LLM review layer (Level 2). Active only with --llm-monitor flag.

Calls Claude Haiku once per hour for positions with ambiguous P&L
(between -0.5×ATR and +0.5×ATR). Cost target: < $0.50/day.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from src.monitor.monitor_state import EnrichedPosition, ExitDecision

logger = logging.getLogger(__name__)

# Haiku is cheapest Claude model — use it for high-frequency monitor reviews
_LLM_MODEL = "claude-haiku-4-5-20251001"

# In-memory: {ticker: last_review_datetime}
_last_review: dict[str, datetime] = {}
REVIEW_INTERVAL_HOURS = 1


def should_review(pos: EnrichedPosition, atr: float) -> bool:
    """Return True if this position is due for LLM review."""
    if atr <= 0:
        return False

    profit = pos.profit_per_share()
    profit_in_atr = abs(profit / atr)

    # Only review ambiguous positions (not clearly winning or losing)
    if profit_in_atr > 0.5:
        return False

    last = _last_review.get(pos.ticker)
    if last is None:
        return True
    return datetime.now(timezone.utc) - last >= timedelta(hours=REVIEW_INTERVAL_HOURS)


def review_position(
    pos: EnrichedPosition,
    atr: float,
    ohlcv_5min: Optional[pd.DataFrame] = None,
) -> ExitDecision | None:
    """
    Ask Claude Haiku to assess a position with ambiguous P&L.
    Returns ExitDecision or None if LLM declines to act / call fails.
    """
    try:
        from src.utils.llm import call_llm
        from pydantic import BaseModel
        from typing import Literal

        class LLMReview(BaseModel):
            action: Literal["HOLD", "SCALE_OUT", "FULL_EXIT"]
            reasoning: str

        price   = pos.effective_price()
        profit  = pos.profit_per_share()
        entry   = pos.avg_entry_price

        ohlcv_summary = ""
        if ohlcv_5min is not None and not ohlcv_5min.empty:
            last5 = ohlcv_5min.tail(5).to_dict(orient="records")
            ohlcv_summary = json.dumps(last5, default=str)

        prompt = f"""You are a quantitative risk manager reviewing an open position.

Position: {pos.ticker} {pos.side.upper()}
Entry price: ${entry:.2f}
Current price: ${price:.2f}
P&L per share: ${profit:+.4f} ({profit/entry*100:+.2f}%)
ATR(14): ${atr:.2f}
Days open: {pos.days_open:.1f}
Stop loss: {f"${pos.stop_loss:.2f}" if pos.stop_loss else "none"}
Take profit: {f"${pos.take_profit:.2f}" if pos.take_profit else "none"}

Last 5 bars (5min OHLCV):
{ohlcv_summary or "unavailable"}

P&L is between -0.5×ATR and +0.5×ATR — position is ambiguous.
Assess whether to HOLD, SCALE_OUT (take 25% off), or FULL_EXIT.

Respond with JSON: {{"action": "HOLD"|"SCALE_OUT"|"FULL_EXIT", "reasoning": "..."}}
Be conservative: default to HOLD unless there is a clear reason to exit.
"""

        result = call_llm(
            prompt=prompt,
            pydantic_model=LLMReview,
            agent_name="llm_reviewer",
            model_override=_LLM_MODEL,
        )

        _last_review[pos.ticker] = datetime.now(timezone.utc)

        if not isinstance(result, LLMReview) or result.action == "HOLD":
            return None

        action_map = {"SCALE_OUT": "PARTIAL_EXIT", "FULL_EXIT": "FULL_EXIT"}
        return ExitDecision(
            action=action_map[result.action],
            percentage=0.25 if result.action == "SCALE_OUT" else None,
            reason=f"LLM review: {result.reasoning[:200]}",
            rule="llm_review",
        )

    except Exception as exc:
        logger.warning("[llm_reviewer] review_position %s failed: %s", pos.ticker, exc)
        return None
