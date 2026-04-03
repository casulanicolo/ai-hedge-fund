from __future__ import annotations

from datetime import datetime, timedelta
import json

from src.graph.state import AgentState, AgentOutput, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.tools.api_shim import get_company_news, get_financial_metrics, get_insider_trades, get_market_cap, search_line_items, register_state
from src.utils.llm import call_llm
from src.utils.progress import progress


def michael_burry_agent(state: AgentState, agent_id: str = "michael_burry_agent"):
    """Analyse stocks using Michael Burry's deep-value, contrarian framework."""
    data = state["data"]
    register_state(state)
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data: dict[str, dict] = {}
    burry_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=None)

        progress.update_status(agent_id, ticker, "Fetching line items")
        line_items = search_line_items(
            ticker,
            ["free_cash_flow", "net_income", "total_debt", "cash_and_equivalents",
             "total_assets", "total_liabilities", "outstanding_shares",
             "issuance_or_purchase_of_equity_shares"],
            end_date, api_key=None,
        )

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)

        progress.update_status(agent_id, ticker, "Fetching company news")
        news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=250)

        progress.update_status(agent_id, ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=None)

        progress.update_status(agent_id, ticker, "Analyzing value")
        value_analysis = _analyze_value(metrics, line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet")
        balance_sheet_analysis = _analyze_balance_sheet(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing insider activity")
        insider_analysis = _analyze_insider_activity(insider_trades)

        progress.update_status(agent_id, ticker, "Analyzing contrarian sentiment")
        contrarian_analysis = _analyze_contrarian_sentiment(news)

        total_score = (
            value_analysis["score"] + balance_sheet_analysis["score"] +
            insider_analysis["score"] + contrarian_analysis["score"]
        )
        max_score = (
            value_analysis["max_score"] + balance_sheet_analysis["max_score"] +
            insider_analysis["max_score"] + contrarian_analysis["max_score"]
        )

        if total_score >= 0.7 * max_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "value_analysis": value_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "insider_analysis": insider_analysis,
            "contrarian_analysis": contrarian_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating LLM output")
        burry_output = _generate_burry_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        burry_analysis[ticker] = {
            "direction":       burry_output.direction,
            "expected_return": burry_output.expected_return,
            "confidence":      burry_output.confidence,
            "reasoning":       burry_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=burry_output.reasoning)

    message = HumanMessage(content=json.dumps(burry_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(burry_analysis, "Michael Burry Agent")

    state["data"]["analyst_signals"][agent_id] = burry_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ── Sub-analysis helpers ──────────────────────────────────────────────────────

def _latest_line_item(line_items: list):
    return line_items[0] if line_items else None


def _analyze_value(metrics, line_items, market_cap):
    max_score = 6
    score = 0
    details: list[str] = []

    latest_item = _latest_line_item(line_items)
    fcf = getattr(latest_item, "free_cash_flow", None) if latest_item else None
    if fcf is not None and market_cap:
        fcf_yield = fcf / market_cap
        if fcf_yield >= 0.15:
            score += 4
            details.append(f"Extraordinary FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.12:
            score += 3
            details.append(f"Very high FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.08:
            score += 2
            details.append(f"Respectable FCF yield {fcf_yield:.1%}")
        else:
            details.append(f"Low FCF yield {fcf_yield:.1%}")
    else:
        details.append("FCF data unavailable")

    if metrics:
        ev_ebit = getattr(metrics[0], "ev_to_ebit", None)
        if ev_ebit is not None:
            if ev_ebit < 6:
                score += 2
                details.append(f"EV/EBIT {ev_ebit:.1f} (<6)")
            elif ev_ebit < 10:
                score += 1
                details.append(f"EV/EBIT {ev_ebit:.1f} (<10)")
            else:
                details.append(f"High EV/EBIT {ev_ebit:.1f}")
        else:
            details.append("EV/EBIT data unavailable")
    else:
        details.append("Financial metrics unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def _analyze_balance_sheet(metrics, line_items):
    max_score = 3
    score = 0
    details: list[str] = []

    latest_metrics = metrics[0] if metrics else None
    latest_item = _latest_line_item(line_items)

    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 2
            details.append(f"Low D/E {debt_to_equity:.2f}")
        elif debt_to_equity < 1:
            score += 1
            details.append(f"Moderate D/E {debt_to_equity:.2f}")
        else:
            details.append(f"High leverage D/E {debt_to_equity:.2f}")
    else:
        details.append("Debt-to-equity data unavailable")

    if latest_item is not None:
        cash       = getattr(latest_item, "cash_and_equivalents", None)
        total_debt = getattr(latest_item, "total_debt", None)
        if cash is not None and total_debt is not None:
            if cash > total_debt:
                score += 1
                details.append("Net cash position")
            else:
                details.append("Net debt position")
        else:
            details.append("Cash/debt data unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def _analyze_insider_activity(insider_trades):
    max_score = 2
    score = 0
    details: list[str] = []

    if not insider_trades:
        details.append("No insider trade data")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
    shares_sold   = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
    net = shares_bought - shares_sold

    if net > 0:
        score += 2 if net / max(shares_sold, 1) > 1 else 1
        details.append(f"Net insider buying of {net:,} shares")
    else:
        details.append("Net insider selling")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def _analyze_contrarian_sentiment(news):
    max_score = 1
    score = 0
    details: list[str] = []

    if not news:
        details.append("No recent news")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    negative_count = sum(
        1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"]
    )

    if negative_count >= 5:
        score += 1
        details.append(f"{negative_count} negative headlines (contrarian opportunity)")
    else:
        details.append("Limited negative press")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ── LLM generation ───────────────────────────────────────────────────────────

def _generate_burry_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> AgentOutput:
    """Call the LLM to craft the final trading signal in Burry's voice."""

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are Dr. Michael J. Burry. Your mandate:
- Hunt for deep value using hard numbers (FCF yield, EV/EBIT, balance sheet)
- Be contrarian: negative press can be your friend if fundamentals are solid
- Focus on downside first — avoid leveraged balance sheets
- Look for hard catalysts: insider buying, buybacks, asset sales
- Communicate in Burry's terse, data-driven style

Decision rules:
- direction=LONG:    high FCF yield AND solid balance sheet AND value confirmed
- direction=SHORT:   overvalued OR dangerously leveraged OR fundamental deterioration
- direction=NEUTRAL: mixed or insufficient data

expected_return: realistic % return for a 3-4 day swing trade (e.g. 0.025 = +2.5%).
Positive for LONG, negative for SHORT, ~0 for NEUTRAL. Range: -0.10 to +0.10.
confidence: 0.1 to 1.0.
reasoning: terse, data-driven, max 2 sentences with concrete numbers.""",
        ),
        (
            "human",
            """Analysis Data for {ticker}:
{analysis_data}

Return exactly:
{{
  "direction": "LONG" | "SHORT" | "NEUTRAL",
  "expected_return": float,
  "confidence": float,
  "reasoning": "string"
}}""",
        ),
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker,
    }).to_string()

    def default_factory():
        return AgentOutput(direction="NEUTRAL", expected_return=0.0, confidence=0.1, reasoning="Parsing error — defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=AgentOutput,
        agent_name=agent_id,
        state=state,
        default_factory=default_factory,
    )
