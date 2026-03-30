"""
portfolio_manager.py — Athanor Alpha Phase 3.4
Reads analyst_signals + risk_report from state.
Produces per-ticker recommendations:
  {ticker, action, sizing_pct, conviction, reasoning}
No execution — output only. Writes to state["data"]["portfolio_recommendations"].
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress

AGENT_ID = "portfolio_manager"

# ── Sizing parameters ──────────────────────────────────────────────────────────
MAX_SINGLE_POSITION = 20.0    # % of portfolio, hard cap per ticker
MIN_SINGLE_POSITION = 2.0     # % below this → skip (not worth it)
TOTAL_GROSS_BUDGET = 100.0    # we size to 100% gross, human decides leverage
CONVICTION_SCALE = {          # how conviction maps to base sizing
    "high":   15.0,
    "medium": 9.0,
    "low":    4.0,
}

# Agents excluded from signal aggregation (they don't emit trading signals)
NON_SIGNAL_AGENTS = {"risk_manager", "data_prefetch"}


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic output model
# ══════════════════════════════════════════════════════════════════════════════

class TickerRecommendation(BaseModel):
    ticker: str = Field(description="Equity ticker symbol")
    action: str = Field(description="BUY | SELL | HOLD")
    sizing_pct: float = Field(description="Recommended portfolio weight in percent (0-100)")
    conviction: float = Field(description="Conviction score 0.0-1.0")
    reasoning: str = Field(description="One-sentence rationale")


class PortfolioOutput(BaseModel):
    recommendations: list[TickerRecommendation] = Field(
        description="One recommendation per ticker"
    )
    portfolio_summary: str = Field(
        description="2-3 sentence summary of the overall portfolio stance"
    )
    risk_notes: str = Field(
        description="Any risk warnings the human should be aware of"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Signal aggregation with agent weights
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_signals(state: AgentState, tickers: list[str]) -> dict[str, dict]:
    """
    For each ticker compute a weighted net score across all agents.
    Uses agent_weights from state if available, otherwise uniform weights.
    Returns {ticker: {net_score, avg_confidence, bull_agents, bear_agents, net_signal}}
    """
    analyst_signals: dict[str, Any] = state.get("data", {}).get("analyst_signals", {})
    agent_weights: dict[str, Any] = state.get("data", {}).get("agent_weights", {})

    result: dict[str, dict] = {}

    for ticker in tickers:
        net_score = 0.0
        total_weight = 0.0
        bull_agents: list[str] = []
        bear_agents: list[str] = []
        conf_sum = 0.0
        n = 0

        for agent_id, agent_data in analyst_signals.items():
            if agent_id in NON_SIGNAL_AGENTS:
                continue
            if not isinstance(agent_data, dict):
                continue

            # Resolve signal data for this ticker
            if ticker in agent_data:
                sig_data = agent_data[ticker]
            elif agent_data.get("ticker") == ticker:
                sig_data = agent_data
            else:
                continue

            if not isinstance(sig_data, dict):
                continue

            signal = str(sig_data.get("signal", "neutral")).lower()
            confidence = float(sig_data.get("confidence", 0.5))

            # Agent weight: from feedback loop DB if available, else 1.0
            weight = 1.0
            if agent_weights:
                w_entry = agent_weights.get(agent_id, {})
                if isinstance(w_entry, dict):
                    weight = float(w_entry.get(ticker, w_entry.get("default", 1.0)))
                elif isinstance(w_entry, (int, float)):
                    weight = float(w_entry)

            # Score: bullish = +1, bearish = -1, neutral = 0; scaled by confidence × weight
            if signal == "bullish":
                score = +1.0 * confidence * weight
                bull_agents.append(agent_id)
            elif signal == "bearish":
                score = -1.0 * confidence * weight
                bear_agents.append(agent_id)
            else:
                score = 0.0

            net_score += score
            total_weight += weight
            conf_sum += confidence
            n += 1

        if n == 0:
            result[ticker] = {
                "net_score": 0.0, "avg_confidence": 0.0,
                "bull_agents": [], "bear_agents": [],
                "net_signal": "neutral", "n_agents": 0,
            }
            continue

        norm_score = net_score / total_weight if total_weight > 0 else 0.0
        avg_conf = conf_sum / n

        if norm_score > 0.1:
            net_signal = "bullish"
        elif norm_score < -0.1:
            net_signal = "bearish"
        else:
            net_signal = "neutral"

        result[ticker] = {
            "net_score": round(norm_score, 4),
            "avg_confidence": round(avg_conf, 3),
            "bull_agents": bull_agents,
            "bear_agents": bear_agents,
            "net_signal": net_signal,
            "n_agents": n,
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Sizing logic (no LLM needed here — pure math)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_sizing(
    ticker: str,
    agg: dict,
    risk_report: dict,
) -> tuple[float, float, str]:
    """
    Returns (sizing_pct, conviction_score, action).
    sizing_pct is BEFORE normalisation across portfolio.
    """
    net_score = agg.get("net_score", 0.0)
    avg_conf = agg.get("avg_confidence", 0.5)
    net_signal = agg.get("net_signal", "neutral")

    # Conviction: |net_score| × avg_confidence → [0, 1]
    conviction = min(1.0, abs(net_score) * avg_conf * 2.0)

    # Base size from conviction tier
    if conviction >= 0.60:
        base_size = CONVICTION_SCALE["high"]
    elif conviction >= 0.35:
        base_size = CONVICTION_SCALE["medium"]
    else:
        base_size = CONVICTION_SCALE["low"]

    # Risk penalty: reduce sizing if ticker has warnings
    ticker_flags = risk_report.get("ticker_flags", {}).get(ticker, [])
    risk_penalty = 0.8 if ticker_flags else 1.0

    # Global VaR penalty
    daily_var = risk_report.get("daily_var_95", 0.0)
    if daily_var > 0.04:
        risk_penalty *= 0.75

    sizing = base_size * risk_penalty

    if net_signal == "bullish":
        action = "BUY"
    elif net_signal == "bearish":
        action = "SELL"
        sizing = min(sizing, 10.0)  # cap short sizing
    else:
        action = "HOLD"
        sizing = 0.0

    sizing = round(min(sizing, MAX_SINGLE_POSITION), 1)
    conviction = round(conviction, 3)

    return sizing, conviction, action


def _normalise_sizing(recommendations: list[dict]) -> list[dict]:
    """
    Scale sizing_pct so that BUY positions sum to at most TOTAL_GROSS_BUDGET.
    Remove positions below MIN_SINGLE_POSITION.
    """
    buys = [r for r in recommendations if r["action"] == "BUY" and r["sizing_pct"] >= MIN_SINGLE_POSITION]
    others = [r for r in recommendations if r not in buys]

    total = sum(r["sizing_pct"] for r in buys)
    if total > TOTAL_GROSS_BUDGET and total > 0:
        scale = TOTAL_GROSS_BUDGET / total
        for r in buys:
            r["sizing_pct"] = round(r["sizing_pct"] * scale, 1)

    # Filter out positions that became too small after scaling
    buys = [r for r in buys if r["sizing_pct"] >= MIN_SINGLE_POSITION]

    return buys + others


# ══════════════════════════════════════════════════════════════════════════════
# LLM reasoning per ticker
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm_prompt(
    tickers: list[str],
    weighted: dict[str, dict],
    risk_report: dict,
    pre_recs: list[dict],
) -> str:
    warnings_text = "\n".join(risk_report.get("warnings", [])) or "None"

    ticker_block = ""
    for t in tickers:
        agg = weighted.get(t, {})
        pre = next((r for r in pre_recs if r["ticker"] == t), {})
        ticker_block += (
            f"\n{t}: net_score={agg.get('net_score', 0):.3f}, "
            f"avg_confidence={agg.get('avg_confidence', 0):.2f}, "
            f"bull_agents={len(agg.get('bull_agents', []))}, "
            f"bear_agents={len(agg.get('bear_agents', []))}, "
            f"action={pre.get('action','?')}, sizing={pre.get('sizing_pct', 0)}%, "
            f"conviction={pre.get('conviction', 0):.2f}"
        )

    return f"""You are the Portfolio Manager of Athanor Alpha, a quantitative AI hedge fund.
You have received aggregated analyst signals and preliminary sizing for {len(tickers)} US equities.

SIGNAL SUMMARY (per ticker):
{ticker_block}

RISK WARNINGS:
{warnings_text}

Portfolio VaR (95%, daily): {risk_report.get('daily_var_95', 0)*100:.1f}%
Historical Max Drawdown estimate: {risk_report.get('max_drawdown_estimate', 0)*100:.1f}%

YOUR TASK:
For each ticker, write a one-sentence reasoning that explains the recommended action and sizing.
Then write a 2-3 sentence portfolio summary and a brief risk note for the human trader.

Respond ONLY with valid JSON matching this schema:
{{
  "recommendations": [
    {{
      "ticker": "AAPL",
      "action": "BUY",
      "sizing_pct": 12.5,
      "conviction": 0.72,
      "reasoning": "..."
    }}
  ],
  "portfolio_summary": "...",
  "risk_notes": "..."
}}

Use exactly the action and sizing_pct values provided above — do not change the numbers.
Write the reasoning and summaries only.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Main agent node
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_manager_agent(state: AgentState) -> dict:
    """
    LangGraph node — Portfolio Manager.
    Reads analyst_signals + risk_report from state.
    Writes portfolio_recommendations to state["data"]["portfolio_recommendations"].
    """
    data: dict[str, Any] = state.get("data", {})
    tickers: list[str] = data.get("tickers", [])
    risk_report: dict = data.get("risk_report", {})

    if not tickers:
        tickers = list(data.get("prefetched_data", {}).keys())

    progress.update_status(AGENT_ID, None, "Aggregating weighted signals")

    # ── 1. Weighted signal aggregation ────────────────────────────────────
    weighted = _weighted_signals(state, tickers)

    # ── 2. Pre-compute sizing (no LLM) ────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Computing position sizing")
    pre_recs: list[dict] = []
    for ticker in tickers:
        agg = weighted.get(ticker, {})
        sizing, conviction, action = _compute_sizing(ticker, agg, risk_report)
        pre_recs.append({
            "ticker": ticker,
            "action": action,
            "sizing_pct": sizing,
            "conviction": conviction,
            "reasoning": "",  # filled by LLM
        })

    # Normalise sizing so BUY positions sum to ≤100%
    pre_recs = _normalise_sizing(pre_recs)

    # ── 3. LLM for reasoning text ─────────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Generating reasoning via LLM")
    prompt = _build_llm_prompt(tickers, weighted, risk_report, pre_recs)

    try:
        llm_result = call_llm(
            prompt=prompt,
            pydantic_model=PortfolioOutput,
            agent_name=AGENT_ID,
            state=state,
        )

        if isinstance(llm_result, PortfolioOutput):
            # Merge LLM reasoning into pre_recs (keep our numbers, use LLM text)
            llm_map = {r.ticker: r for r in llm_result.recommendations}
            for rec in pre_recs:
                llm_rec = llm_map.get(rec["ticker"])
                if llm_rec:
                    rec["reasoning"] = llm_rec.reasoning
            portfolio_summary = llm_result.portfolio_summary
            risk_notes = llm_result.risk_notes
        else:
            portfolio_summary = "LLM output parsing failed — see raw signals."
            risk_notes = "; ".join(risk_report.get("warnings", []))

    except Exception as e:
        portfolio_summary = f"LLM call failed: {e}"
        risk_notes = "; ".join(risk_report.get("warnings", []))
        # Fill reasoning with fallback text
        for rec in pre_recs:
            if not rec["reasoning"]:
                agg = weighted.get(rec["ticker"], {})
                rec["reasoning"] = (
                    f"Net score {agg.get('net_score', 0):.3f} from "
                    f"{agg.get('n_agents', 0)} agents; "
                    f"action {rec['action']} at {rec['sizing_pct']}%."
                )

    # ── 4. Final output ───────────────────────────────────────────────────
    portfolio_recommendations = {
        "recommendations": pre_recs,
        "portfolio_summary": portfolio_summary,
        "risk_notes": risk_notes,
        "total_long_pct": round(
            sum(r["sizing_pct"] for r in pre_recs if r["action"] == "BUY"), 1
        ),
        "n_long": sum(1 for r in pre_recs if r["action"] == "BUY"),
        "n_short": sum(1 for r in pre_recs if r["action"] == "SELL"),
        "n_hold": sum(1 for r in pre_recs if r["action"] == "HOLD"),
    }

    data["portfolio_recommendations"] = portfolio_recommendations
    data["analyst_signals"][AGENT_ID] = {
        "signal": "neutral",
        "confidence": 1.0,
        "reasoning": portfolio_summary,
    }

    show_agent_reasoning(portfolio_recommendations, AGENT_ID)

    summary = (
        f"Portfolio Manager | Long: {portfolio_recommendations['n_long']} tickers "
        f"({portfolio_recommendations['total_long_pct']}%) | "
        f"Short: {portfolio_recommendations['n_short']} | "
        f"Hold: {portfolio_recommendations['n_hold']}"
    )
    progress.update_status(AGENT_ID, None, "Done")

    message = HumanMessage(content=summary, name=AGENT_ID)

    return {
        "messages": [message],
        "data": data,
    }
