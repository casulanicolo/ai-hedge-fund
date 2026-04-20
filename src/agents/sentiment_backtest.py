"""
src/agents/sentiment_backtest.py
────────────────────────────────
Sentiment proxy for backtest mode.

Live sentiment_agent depends on yfinance .news (real-time only) — no
historical archive available pre-2020. For backtest, derive a synthetic
sentiment signal from observable price action:

  - realized_vol_5d normalized to z-score of 60d window
  - volume_surge = volume_5d_avg / volume_60d_avg

Heuristic mapping
-----------------
  surge > 1.5  AND  return_5d > +2%   →  LONG  high urgency
  surge > 1.5  AND  return_5d < -2%   →  SHORT high urgency
  vol_z  > 1.5                         →  signal but lower confidence
  otherwise                            →  NEUTRAL low

Output shape mirrors src/agents/sentiment.py so downstream consumers
(risk_manager, devils_advocate, portfolio_manager) need no changes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.trade_levels import compute_trade_levels

logger = logging.getLogger(__name__)

AGENT_ID = "sentiment_agent"   # impersonate live agent → same key in analyst_signals


def _proxy_signal(daily: pd.DataFrame) -> dict[str, Any]:
    """Compute sentiment proxy from OHLCV daily DataFrame."""
    if daily is None or daily.empty or len(daily) < 60:
        return {
            "direction": "NEUTRAL", "expected_return": 0.0, "confidence": 0.1,
            "sentiment_score": 0.0, "event_type": "other", "urgency": "low",
            "reasoning": "Backtest proxy: insufficient history.",
        }

    close = daily["Close"]
    vol   = daily["Volume"]

    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
    rv_5d  = float(close.pct_change().tail(5).std()  or 0.0)
    rv_60d = float(close.pct_change().tail(60).std() or 0.0)
    vol_z  = (rv_5d - rv_60d) / (rv_60d + 1e-9) if rv_60d > 0 else 0.0

    avg_vol_5d  = float(vol.tail(5).mean()  or 0.0)
    avg_vol_60d = float(vol.tail(60).mean() or 0.0)
    surge = (avg_vol_5d / avg_vol_60d) if avg_vol_60d > 0 else 1.0

    # Direction + urgency
    if surge > 1.5 and ret_5d > 0.02:
        direction, urgency = "LONG", "high"
        expected_return = round(min(0.05, ret_5d * 0.6), 4)
        confidence = min(1.0, 0.4 + 0.2 * (surge - 1.5))
        score = min(1.0, 0.3 + 0.5 * surge / 2.0)
    elif surge > 1.5 and ret_5d < -0.02:
        direction, urgency = "SHORT", "high"
        expected_return = round(max(-0.05, ret_5d * 0.6), 4)
        confidence = min(1.0, 0.4 + 0.2 * (surge - 1.5))
        score = max(-1.0, -0.3 - 0.5 * surge / 2.0)
    elif abs(vol_z) > 1.5:
        direction = "LONG" if ret_5d > 0 else "SHORT"
        urgency = "medium"
        expected_return = round(np.sign(ret_5d) * 0.015, 4)
        confidence = 0.3
        score = float(np.sign(ret_5d) * 0.3)
    else:
        direction, urgency = "NEUTRAL", "low"
        expected_return, confidence, score = 0.0, 0.15, 0.0

    return {
        "direction":       direction,
        "expected_return": expected_return,
        "confidence":      round(confidence, 3),
        "sentiment_score": round(score, 3),
        "event_type":      "other",
        "urgency":         urgency,
        "reasoning": (
            f"Backtest proxy: ret_5d={ret_5d:+.3f}, vol_surge={surge:.2f}x, "
            f"rv_z={vol_z:+.2f}. Heuristic — no LLM call."
        ),
    }


def sentiment_backtest_agent(state: AgentState) -> dict[str, Any]:
    """LangGraph node — backtest replacement for sentiment_agent."""
    data: dict[str, Any] = state.get("data", {})
    tickers: list[str] = data.get("tickers", [])
    analyst_signals: dict[str, Any] = data.setdefault("analyst_signals", {})
    analyst_signals.setdefault(AGENT_ID, {})

    prefetched = data.get("prefetched_data", {})

    for ticker in tickers:
        try:
            payload = prefetched.get(ticker, {})
            daily   = payload.get("ohlcv_daily")
            sig     = _proxy_signal(daily)
            levels  = compute_trade_levels(sig["direction"], state, ticker)
            analyst_signals[AGENT_ID][ticker] = {**sig, **levels}
        except Exception as exc:
            logger.error("sentiment_backtest %s: %s", ticker, exc, exc_info=True)
            levels = compute_trade_levels("NEUTRAL", state, ticker)
            analyst_signals[AGENT_ID][ticker] = {
                "direction": "NEUTRAL", "expected_return": 0.0, "confidence": 0.1,
                "sentiment_score": 0.0, "event_type": "other", "urgency": "low",
                "reasoning": f"Backtest proxy error: {exc}",
                **levels,
            }

    show_agent_reasoning(analyst_signals[AGENT_ID], AGENT_ID + "(backtest)")
    msg = HumanMessage(content=json.dumps(analyst_signals[AGENT_ID]), name=AGENT_ID)
    return {"messages": [msg], "data": data}
