from src.graph.state import AgentState, AgentOutput, show_agent_reasoning
from src.tools.api_shim import get_financial_metrics, get_market_cap, search_line_items, register_state
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import json
from src.utils.progress import progress
from src.utils.llm import call_llm


def cathie_wood_agent(state: AgentState, agent_id: str = "cathie_wood_agent"):
    """
    Analyzes stocks using Cathie Wood's investing principles and LLM reasoning.
    """
    data = state["data"]
    register_state(state)
    end_date = data["end_date"]
    tickers = data["tickers"]
    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=None)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            ["revenue", "gross_margin", "operating_margin", "debt_to_equity", "free_cash_flow",
             "total_assets", "total_liabilities", "dividends_and_other_cash_distributions",
             "outstanding_shares", "research_and_development", "capital_expenditure", "operating_expense"],
            end_date, period="annual", limit=5, api_key=None,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=None)

        progress.update_status(agent_id, ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating valuation & high-growth scenario")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap)

        total_score = disruptive_analysis["score"] + innovation_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15

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
            "disruptive_analysis": disruptive_analysis,
            "innovation_analysis": innovation_analysis,
            "valuation_analysis": valuation_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Cathie Wood analysis")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        cw_analysis[ticker] = {
            "direction":       cw_output.direction,
            "expected_return": cw_output.expected_return,
            "confidence":      cw_output.confidence,
            "reasoning":       cw_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=cw_output.reasoning)

    message = HumanMessage(content=json.dumps(cw_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = cw_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ── Analysis helpers ──────────────────────────────────────────────────────────

def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze disruptive potential"}

    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 3:
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i] and revenues[i + 1]:
                growth_rates.append((revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]))

        if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
            score += 2
            details.append(f"Revenue growth accelerating: {growth_rates[0]*100:.1f}% vs {growth_rates[-1]*100:.1f}%")

        latest_growth = growth_rates[0] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional revenue growth: {latest_growth*100:.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong revenue growth: {latest_growth*100:.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate revenue growth: {latest_growth*100:.1f}%")
    else:
        details.append("Insufficient revenue data")

    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, "gross_margin") and item.gross_margin is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0.05:
            score += 2
            details.append(f"Expanding gross margins: +{margin_trend*100:.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{margin_trend*100:.1f}%")
        if gross_margins[0] > 0.50:
            score += 2
            details.append(f"High gross margin: {gross_margins[0]*100:.1f}%")

    operating_expenses = [item.operating_expense for item in financial_line_items if hasattr(item, "operating_expense") and item.operating_expense]
    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth  = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])
        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage: revenue growing faster than expenses")

    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[0] / revenues[0]
        if rd_intensity > 0.15:
            score += 3
            details.append(f"High R&D investment: {rd_intensity*100:.1f}% of revenue")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D investment: {rd_intensity*100:.1f}% of revenue")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Some R&D investment: {rd_intensity*100:.1f}% of revenue")

    normalized_score = (score / 12) * 5
    return {"score": normalized_score, "details": "; ".join(details)}


def analyze_innovation_growth(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze innovation-driven growth"}

    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development]
    revenues    = [item.revenue for item in financial_line_items if item.revenue]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
        if rd_growth > 0.5:
            score += 3
            details.append(f"Strong R&D investment growth: +{rd_growth*100:.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D investment growth: +{rd_growth*100:.1f}%")

        rd_intensity_start = rd_expenses[-1] / revenues[-1]
        rd_intensity_end   = rd_expenses[0]  / revenues[0]
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"Increasing R&D intensity: {rd_intensity_end*100:.1f}% vs {rd_intensity_start*100:.1f}%")

    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    if fcf_vals and len(fcf_vals) >= 2:
        fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("Strong and consistent FCF growth")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("Consistent positive FCF")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("Moderately consistent FCF")

    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if op_margin_vals and len(op_margin_vals) >= 2:
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]
        if op_margin_vals[0] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving operating margin: {op_margin_vals[0]*100:.1f}%")
        elif op_margin_vals[0] > 0.10:
            score += 2
            details.append(f"Healthy operating margin: {op_margin_vals[0]*100:.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving operating efficiency")

    capex = [item.capital_expenditure for item in financial_line_items if hasattr(item, "capital_expenditure") and item.capital_expenditure]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[0]) / revenues[0]
        capex_growth    = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0
        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")

    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if hasattr(item, "dividends_and_other_cash_distributions") and item.dividends_and_other_cash_distributions]
    if dividends and fcf_vals and fcf_vals[0] != 0:
        payout_ratio = dividends[0] / fcf_vals[0]
        if payout_ratio < 0.2:
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")

    normalized_score = (score / 15) * 5
    return {"score": normalized_score, "details": "; ".join(details)}


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float) -> dict:
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data for valuation"}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF; FCF = {fcf}", "intrinsic_value": None}

    growth_rate      = 0.20
    discount_rate    = 0.15
    terminal_multiple = 25
    projection_years  = 5

    present_value = sum(
        fcf * (1 + growth_rate) ** y / (1 + discount_rate) ** y
        for y in range(1, projection_years + 1)
    )
    terminal_value  = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / (1 + discount_rate) ** projection_years
    intrinsic_value = present_value + terminal_value
    mos             = (intrinsic_value - market_cap) / market_cap

    score = 3 if mos > 0.5 else (1 if mos > 0.2 else 0)

    return {
        "score": score,
        "details": f"IV ≈ {intrinsic_value:,.0f}, MktCap ≈ {market_cap:,.0f}, MOS = {mos:.2%}",
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": mos,
    }


# ── LLM output ────────────────────────────────────────────────────────────────

def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "cathie_wood_agent",
) -> AgentOutput:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are Cathie Wood. Make investment decisions using your principles:
1. Seek companies leveraging disruptive innovation with large TAM.
2. Emphasize exponential growth potential and R&D intensity.
3. Accept higher volatility in pursuit of high long-term returns.
4. Growth-biased valuation approach.

Decision rules:
- direction=LONG:    strong disruptive potential AND accelerating growth AND positive MOS
- direction=SHORT:   no disruptive edge OR declining growth OR extreme overvaluation
- direction=NEUTRAL: mixed evidence

expected_return: realistic % return for a 3-4 day swing trade based on innovation momentum (e.g. 0.03 = +3%).
Positive for LONG, negative for SHORT, ~0 for NEUTRAL. Range: -0.10 to +0.10.
confidence: 0.1 to 1.0.
reasoning: future-focused, conviction-driven, max 2 sentences with key metrics.

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
