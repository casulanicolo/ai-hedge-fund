from src.graph.state import AgentState, AgentOutput, show_agent_reasoning
from src.tools.api_shim import get_financial_metrics, get_market_cap, search_line_items, register_state
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import json
from src.utils.progress import progress
from src.utils.llm import call_llm


def bill_ackman_agent(state: AgentState, agent_id: str = "bill_ackman_agent"):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    """
    data = state["data"]
    register_state(state)
    end_date = data["end_date"]
    tickers = data["tickers"]
    analysis_data = {}
    ackman_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=None)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            ["revenue", "operating_margin", "debt_to_equity", "free_cash_flow",
             "total_assets", "total_liabilities", "dividends_and_other_cash_distributions",
             "outstanding_shares"],
            end_date, period="annual", limit=5, api_key=None,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=None)

        progress.update_status(agent_id, ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet and capital structure")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing activism potential")
        activism_analysis = analyze_activism_potential(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value & margin of safety")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)

        total_score = (
            quality_analysis["score"] + balance_sheet_analysis["score"] +
            activism_analysis["score"] + valuation_analysis["score"]
        )
        max_possible_score = 20

        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "activism_analysis": activism_analysis,
            "valuation_analysis": valuation_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Bill Ackman analysis")
        ackman_output = generate_ackman_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        ackman_analysis[ticker] = {
            "direction":       ackman_output.direction,
            "expected_return": ackman_output.expected_return,
            "confidence":      ackman_output.confidence,
            "reasoning":       ackman_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=ackman_output.reasoning)

    message = HumanMessage(content=json.dumps(ackman_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")

    state["data"]["analyst_signals"][agent_id] = ackman_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ── Analysis helpers ──────────────────────────────────────────────────────────

def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze business quality"}

    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2:
        initial, final = revenues[-1], revenues[0]
        if initial and final and final > initial:
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:
                score += 2
                details.append(f"Revenue grew {growth_rate*100:.1f}% cumulatively (strong growth).")
            else:
                score += 1
                details.append(f"Revenue growth positive but under 50% ({growth_rate*100:.1f}%).")
        else:
            details.append("Revenue did not grow significantly.")
    else:
        details.append("Not enough revenue data.")

    fcf_vals      = [item.free_cash_flow    for item in financial_line_items if item.free_cash_flow    is not None]
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin  is not None]

    if op_margin_vals:
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins often exceeded 15%.")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data.")

    if fcf_vals:
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive FCF.")
        else:
            details.append("FCF not consistently positive.")

    latest_metrics = metrics[0]
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        details.append(f"High ROE of {latest_metrics.return_on_equity:.1%}.")
    elif latest_metrics.return_on_equity:
        details.append(f"ROE of {latest_metrics.return_on_equity:.1%} is moderate.")
    else:
        details.append("ROE data not available.")

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze financial discipline"}

    debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
            score += 2
            details.append("Debt-to-equity < 1.0 for majority of periods.")
        else:
            details.append("Debt-to-equity >= 1.0 in many periods.")
    else:
        liab_to_assets = [
            item.total_liabilities / item.total_assets
            for item in financial_line_items
            if item.total_liabilities and item.total_assets and item.total_assets > 0
        ]
        if liab_to_assets:
            if sum(1 for r in liab_to_assets if r < 0.5) >= (len(liab_to_assets) // 2 + 1):
                score += 2
                details.append("Liabilities-to-assets < 50% for majority of periods.")
            else:
                details.append("Liabilities-to-assets >= 50% in many periods.")
        else:
            details.append("No consistent leverage ratio data.")

    dividends_list = [item.dividends_and_other_cash_distributions for item in financial_line_items
                      if item.dividends_and_other_cash_distributions is not None]
    if dividends_list:
        if sum(1 for d in dividends_list if d < 0) >= (len(dividends_list) // 2 + 1):
            score += 1
            details.append("History of returning capital via dividends.")
        else:
            details.append("Dividends not consistently paid.")

    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    if len(shares) >= 2:
        if shares[0] < shares[-1]:
            score += 1
            details.append("Outstanding shares decreased over time (buybacks).")
        else:
            details.append("Outstanding shares have not decreased.")

    return {"score": score, "details": "; ".join(details)}


def analyze_activism_potential(financial_line_items: list) -> dict:
    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data for activism potential"}

    revenues   = [item.revenue          for item in financial_line_items if item.revenue          is not None]
    op_margins = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]

    if len(revenues) < 2 or not op_margins:
        return {"score": 0, "details": "Not enough data to assess activism potential."}

    initial, final = revenues[-1], revenues[0]
    revenue_growth = (final - initial) / abs(initial) if initial else 0
    avg_margin = sum(op_margins) / len(op_margins)

    score = 0
    details = []

    if revenue_growth > 0.15 and avg_margin < 0.10:
        score += 2
        details.append(f"Revenue growth {revenue_growth*100:.1f}% with low avg margin {avg_margin*100:.1f}% — activism could unlock value.")
    else:
        details.append("No clear activism opportunity.")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF; FCF = {fcf}", "intrinsic_value": None}

    growth_rate      = 0.06
    discount_rate    = 0.10
    terminal_multiple = 15
    projection_years  = 5

    present_value = sum(
        fcf * (1 + growth_rate) ** y / (1 + discount_rate) ** y
        for y in range(1, projection_years + 1)
    )
    terminal_value = (
        fcf * (1 + growth_rate) ** projection_years * terminal_multiple
    ) / (1 + discount_rate) ** projection_years

    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 3 if margin_of_safety > 0.3 else (1 if margin_of_safety > 0.1 else 0)

    return {
        "score": score,
        "details": f"IV ≈ {intrinsic_value:,.0f}, MktCap ≈ {market_cap:,.0f}, MOS = {margin_of_safety:.2%}",
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety,
    }


# ── LLM output ────────────────────────────────────────────────────────────────

def generate_ackman_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> AgentOutput:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are Bill Ackman. Make investment decisions using your principles:
1. High-quality businesses with durable competitive advantages.
2. Consistent free cash flow and long-term growth.
3. Strong financial discipline (reasonable leverage, efficient capital allocation).
4. Valuation with a margin of safety.
5. Activism potential where operational improvements can unlock value.

Decision rules:
- direction=LONG:    high quality AND fair/cheap valuation AND strong FCF
- direction=SHORT:   poor quality OR overvalued OR weak balance sheet
- direction=NEUTRAL: mixed evidence

expected_return: realistic % return for a 3-4 day swing trade (e.g. 0.02 = +2%).
Positive for LONG, negative for SHORT, ~0 for NEUTRAL. Range: -0.10 to +0.10.
confidence: 0.1 to 1.0.
reasoning: confident, analytic tone, max 2 sentences with key numbers.

Return JSON only.""",
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
        return AgentOutput(direction="NEUTRAL", expected_return=0.0, confidence=0.1, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=AgentOutput,
        agent_name=agent_id,
        state=state,
        default_factory=default_factory,
    )
