# src/agents/fundamentals.py
from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, AgentOutput, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress


# ── Data loader ──────────────────────────────────────────────────────────────

def _load_ticker_data(state: AgentState, ticker: str) -> dict:
    """
    Load financial data for a ticker.
    First tries prefetched_data (if prefetch agent ran).
    Falls back to fetching directly from yfinance.
    """
    prefetched = state.get("data", {}).get("prefetched_data", {}).get(ticker)
    if prefetched:
        return prefetched

    # Fallback: fetch directly
    t = yf.Ticker(ticker)
    return {
        "income_stmt_q":   t.quarterly_income_stmt,
        "balance_sheet_q": t.quarterly_balance_sheet,
        "cash_flow_q":     t.quarterly_cashflow,
        "info":            t.info or {},
    }


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_revenue_growth(incq: pd.DataFrame) -> tuple[float, dict]:
    """Score 0-100 based on QoQ and YoY revenue growth."""
    raw: dict[str, Any] = {}
    try:
        rev = incq.loc["Total Revenue"]
        q0 = float(rev.iloc[0])
        q1 = float(rev.iloc[1])
        q4 = float(rev.iloc[4]) if len(rev) > 4 else None

        qoq = (q0 - q1) / abs(q1) if q1 else None
        yoy = (q0 - q4) / abs(q4) if q4 else None

        raw["revenue_q0_B"]    = round(q0 / 1e9, 2)
        raw["revenue_qoq_pct"] = round(qoq * 100, 2) if qoq is not None else None
        raw["revenue_yoy_pct"] = round(yoy * 100, 2) if yoy is not None else None

        score = 50.0
        if qoq is not None:
            score += max(-50, min(50, qoq * 500))
        if yoy is not None:
            score += max(-50, min(50, yoy * 250))
        score = max(0, min(100, score))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0

    return round(score, 1), raw


def score_operating_margin(incq: pd.DataFrame) -> tuple[float, dict]:
    """Score 0-100 based on operating margin level and trend."""
    raw: dict[str, Any] = {}
    try:
        op_inc = incq.loc["Operating Income"]
        rev    = incq.loc["Total Revenue"]

        margins = []
        for i in range(min(4, len(op_inc))):
            r = float(rev.iloc[i])
            o = float(op_inc.iloc[i])
            margins.append(o / r if r else None)

        margins = [m for m in margins if m is not None]
        current = margins[0] if margins else None
        trend   = margins[0] - margins[-1] if len(margins) >= 2 else 0

        raw["op_margin_current_pct"] = round(current * 100, 2) if current is not None else None
        raw["op_margin_trend_pct"]   = round(trend * 100, 2)
        raw["op_margins_history"]    = [round(m * 100, 2) for m in margins]

        score = 50.0
        if current is not None:
            score += max(-50, min(50, current * 250))
        score += max(-20, min(20, trend * 200))
        score = max(0, min(100, score))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0

    return round(score, 1), raw


def score_fcf_yield(cfq: pd.DataFrame, market_cap: float | None) -> tuple[float, dict]:
    """Score 0-100 based on trailing-twelve-month FCF yield."""
    raw: dict[str, Any] = {}
    try:
        fcf_vals = cfq.loc["Free Cash Flow"]
        ttm_fcf  = sum(float(fcf_vals.iloc[i]) for i in range(min(4, len(fcf_vals))))

        raw["ttm_fcf_B"]    = round(ttm_fcf / 1e9, 2)
        raw["market_cap_B"] = round(market_cap / 1e9, 2) if market_cap else None

        if market_cap and market_cap > 0:
            yield_  = ttm_fcf / market_cap
            raw["fcf_yield_pct"] = round(yield_ * 100, 2)
            score = 50 + yield_ * 1000
            score = max(0, min(100, score))
        else:
            score = 50.0
            raw["fcf_yield_pct"] = None
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0

    return round(score, 1), raw


def score_debt_equity(bsq: pd.DataFrame) -> tuple[float, dict]:
    """Score 0-100: lower and improving D/E is better."""
    raw: dict[str, Any] = {}
    try:
        debt   = bsq.loc["Total Debt"]
        equity = bsq.loc["Common Stock Equity"]

        ratios = []
        for i in range(min(4, len(debt))):
            d = float(debt.iloc[i])
            e = float(equity.iloc[i])
            ratios.append(d / e if e and e != 0 else None)

        ratios  = [r for r in ratios if r is not None]
        current = ratios[0] if ratios else None
        trend   = ratios[0] - ratios[-1] if len(ratios) >= 2 else 0

        raw["debt_equity_current"] = round(current, 3) if current is not None else None
        raw["debt_equity_trend"]   = round(trend, 3)

        if current is not None:
            score = 100 - min(100, current * 25)
        else:
            score = 50.0
        score -= max(-20, min(20, trend * 20))
        score = max(0, min(100, score))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0

    return round(score, 1), raw


def score_insider_activity(info: dict) -> tuple[float, dict]:
    """Score 0-100 from insider ownership (heldPercentInsiders)."""
    raw: dict[str, Any] = {}
    try:
        held = info.get("heldPercentInsiders")
        raw["held_pct_insiders"] = held

        if held is None:
            return 50.0, raw

        score = 50 + min(50, held * 400)
        score = max(0, min(100, score))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0

    return round(score, 1), raw


# ── Main agent ────────────────────────────────────────────────────────────────

def fundamentals_analyst_agent(state: AgentState, agent_id: str = "fundamentals_analyst_agent"):
    """
    Fundamental Analyst Agent – Fase 3.2
    Per ogni ticker:
    1. Legge dati finanziari da prefetched_data o yfinance (fallback)
    2. Calcola 5 score (0-100): revenue growth, op margin, FCF yield, D/E, insider
    3. Passa score + raw data all'LLM per segnale finale
    4. Output strutturato: {direction, expected_return, confidence, reasoning, scores}
    Crypto assets (ticker ending in -USD) are skipped.
    """
    data    = state["data"]
    tickers = data["tickers"]

    fundamental_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Loading financial data")

        # ── Crypto: no fundamental data available ────────────────────────────
        if ticker.endswith("-USD"):
            fundamental_analysis[ticker] = {
                "direction":       "NEUTRAL",
                "expected_return": 0.0,
                "confidence":      0.1,
                "reasoning":       "No fundamental data available for crypto assets.",
                "scores":          {},
            }
            progress.update_status(agent_id, ticker, "Skipped (crypto)")
            continue

        try:
            payload    = _load_ticker_data(state, ticker)
            incq       = payload.get("income_stmt_q")
            bsq        = payload.get("balance_sheet_q")
            cfq        = payload.get("cash_flow_q")
            info       = payload.get("info", {})
            market_cap = info.get("marketCap")

            # ── Compute scores ───────────────────────────────────────────────
            progress.update_status(agent_id, ticker, "Computing fundamental scores")

            s_rev, r_rev = score_revenue_growth(incq)       if incq is not None and not incq.empty else (50.0, {"error": "no data"})
            s_mar, r_mar = score_operating_margin(incq)     if incq is not None and not incq.empty else (50.0, {"error": "no data"})
            s_fcf, r_fcf = score_fcf_yield(cfq, market_cap) if cfq  is not None and not cfq.empty  else (50.0, {"error": "no data"})
            s_de,  r_de  = score_debt_equity(bsq)           if bsq  is not None and not bsq.empty   else (50.0, {"error": "no data"})
            s_ins, r_ins = score_insider_activity(info)

            composite = round((s_rev + s_mar + s_fcf + s_de + s_ins) / 5, 1)

            scores = {
                "revenue_growth":   s_rev,
                "operating_margin": s_mar,
                "fcf_yield":        s_fcf,
                "debt_equity":      s_de,
                "insider_activity": s_ins,
                "composite":        composite,
            }

            raw_data = {
                "revenue_growth":   r_rev,
                "operating_margin": r_mar,
                "fcf_yield":        r_fcf,
                "debt_equity":      r_de,
                "insider_activity": r_ins,
            }

            # ── LLM call ─────────────────────────────────────────────────────
            progress.update_status(agent_id, ticker, "Generating signal via LLM")

            prompt = f"""You are a fundamental analyst evaluating {ticker} for a 3-4 day swing trade.

QUANTITATIVE SCORES (0=worst, 100=best):
- Revenue Growth (QoQ + YoY): {s_rev}/100
- Operating Margin Trend:     {s_mar}/100
- FCF Yield:                  {s_fcf}/100
- Debt/Equity Trajectory:     {s_de}/100
- Insider Ownership:          {s_ins}/100
- COMPOSITE SCORE:            {composite}/100

RAW DATA:
{raw_data}

Respond with:
- direction: LONG if composite > 65, SHORT if composite < 35, NEUTRAL otherwise
- expected_return: your realistic return estimate for a 3-4 day swing trade based on fundamental strength (e.g. 0.02 = +2%). Must reflect a realistic target, not a theoretical maximum. Range: -0.10 to +0.10.
- confidence: your signal quality score from 0.1 to 1.0
- reasoning: 2-3 sentences explaining the key fundamental drivers

Respond in JSON format.
"""

            result: AgentOutput = call_llm(
                prompt=prompt,
                pydantic_model=AgentOutput,
                agent_name=agent_id,
                state=state,
            )

            fundamental_analysis[ticker] = {
                "direction":       result.direction,
                "expected_return": result.expected_return,
                "confidence":      result.confidence,
                "reasoning":       result.reasoning,
                "scores":          scores,
            }

            show_agent_reasoning(
                {"scores": scores, "direction": result.direction, "confidence": result.confidence},
                "Fundamental Analyst",
            )

            progress.update_status(agent_id, ticker, "Done")

        except Exception as e:
            progress.update_status(agent_id, ticker, f"Error: {e}")
            fundamental_analysis[ticker] = {
                "direction":       "NEUTRAL",
                "expected_return": 0.0,
                "confidence":      0.1,
                "reasoning":       f"Error during analysis: {e}",
                "scores":          {},
            }

    # ── Write to state ────────────────────────────────────────────────────────
    msg = HumanMessage(content=str(fundamental_analysis), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(fundamental_analysis, "Fundamental Analyst Agent")

    state["data"]["analyst_signals"][agent_id] = fundamental_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [msg], "data": data}
