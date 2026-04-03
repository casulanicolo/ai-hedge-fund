from src.graph.state import AgentState, AgentOutput, show_agent_reasoning
from src.tools.api_shim import get_financial_metrics, get_market_cap, search_line_items, get_insider_trades, get_company_news, register_state
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import json
from src.utils.progress import progress
from src.utils.llm import call_llm


def charlie_munger_agent(state: AgentState, agent_id: str = "charlie_munger_agent"):
    """
    Analyzes stocks using Charlie Munger's investing principles and mental models.
    Focuses on moat strength, management quality, predictability, and valuation.
    """
    data = state["data"]
    register_state(state)
    end_date = data["end_date"]
    tickers = data["tickers"]
    analysis_data = {}
    munger_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10, api_key=None)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            ["revenue", "net_income", "operating_income", "return_on_invested_capital",
             "gross_margin", "operating_margin", "free_cash_flow", "capital_expenditure",
             "cash_and_equivalents", "total_debt", "shareholders_equity", "outstanding_shares",
             "research_and_development", "goodwill_and_intangible_assets"],
            end_date, period="annual", limit=10, api_key=None,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=None)

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, limit=100, api_key=None)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=10, api_key=None)

        progress.update_status(agent_id, ticker, "Analyzing moat strength")
        moat_analysis = analyze_moat_strength(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management quality")
        management_analysis = analyze_management_quality(financial_line_items, insider_trades)

        progress.update_status(agent_id, ticker, "Analyzing business predictability")
        predictability_analysis = analyze_predictability(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating Munger-style valuation")
        valuation_analysis = calculate_munger_valuation(financial_line_items, market_cap)

        total_score = (
            moat_analysis["score"] * 0.35 +
            management_analysis["score"] * 0.25 +
            predictability_analysis["score"] * 0.25 +
            valuation_analysis["score"] * 0.15
        )
        max_possible_score = 10

        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 5.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "moat_analysis": moat_analysis,
            "management_analysis": management_analysis,
            "predictability_analysis": predictability_analysis,
            "valuation_analysis": valuation_analysis,
            "news_sentiment": analyze_news_sentiment(company_news) if company_news else "No news data available",
        }

        progress.update_status(agent_id, ticker, "Generating Charlie Munger analysis")
        munger_output = generate_munger_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
            confidence_hint=compute_confidence(analysis_data[ticker], signal),
        )

        munger_analysis[ticker] = {
            "direction":       munger_output.direction,
            "expected_return": munger_output.expected_return,
            "confidence":      munger_output.confidence,
            "reasoning":       munger_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=munger_output.reasoning)

    message = HumanMessage(content=json.dumps(munger_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(munger_analysis, "Charlie Munger Agent")

    progress.update_status(agent_id, None, "Done")
    state["data"]["analyst_signals"][agent_id] = munger_analysis

    return {"messages": [message], "data": state["data"]}


# ── Analysis helpers (unchanged logic) ───────────────────────────────────────

def analyze_moat_strength(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze moat strength"}

    roic_values = [item.return_on_invested_capital for item in financial_line_items
                   if hasattr(item, 'return_on_invested_capital') and item.return_on_invested_capital is not None]

    if roic_values:
        high_roic_count = sum(1 for r in roic_values if r > 0.15)
        if high_roic_count >= len(roic_values) * 0.8:
            score += 3
            details.append(f"Excellent ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count >= len(roic_values) * 0.5:
            score += 2
            details.append(f"Good ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count > 0:
            score += 1
            details.append(f"Mixed ROIC: >15% in only {high_roic_count}/{len(roic_values)} periods")
        else:
            details.append("Poor ROIC: Never exceeds 15% threshold")
    else:
        details.append("No ROIC data available")

    gross_margins = [item.gross_margin for item in financial_line_items
                     if hasattr(item, 'gross_margin') and item.gross_margin is not None]

    if gross_margins and len(gross_margins) >= 3:
        margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i] >= gross_margins[i-1])
        if margin_trend >= len(gross_margins) * 0.7:
            score += 2
            details.append("Strong pricing power: Gross margins consistently improving")
        elif sum(gross_margins) / len(gross_margins) > 0.3:
            score += 1
            details.append(f"Good pricing power: Average gross margin {sum(gross_margins)/len(gross_margins):.1%}")
        else:
            details.append("Limited pricing power: Low or declining gross margins")
    else:
        details.append("Insufficient gross margin data")

    if len(financial_line_items) >= 3:
        capex_to_revenue = []
        for item in financial_line_items:
            if (hasattr(item, 'capital_expenditure') and item.capital_expenditure is not None and
                    hasattr(item, 'revenue') and item.revenue is not None and item.revenue > 0):
                capex_to_revenue.append(abs(item.capital_expenditure) / item.revenue)

        if capex_to_revenue:
            avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
            if avg_capex_ratio < 0.05:
                score += 2
                details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            elif avg_capex_ratio < 0.10:
                score += 1
                details.append(f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            else:
                details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")

    r_and_d = [item.research_and_development for item in financial_line_items
               if hasattr(item, 'research_and_development') and item.research_and_development is not None]
    goodwill = [item.goodwill_and_intangible_assets for item in financial_line_items
                if hasattr(item, 'goodwill_and_intangible_assets') and item.goodwill_and_intangible_assets is not None]

    if r_and_d and sum(r_and_d) > 0:
        score += 1
        details.append("Invests in R&D, building intellectual property")
    if goodwill:
        score += 1
        details.append("Significant goodwill/intangible assets")

    final_score = min(10, score * 10 / 9)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_management_quality(financial_line_items: list, insider_trades: list) -> dict:
    score = 0
    details = []

    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze management quality"}

    fcf_values = [item.free_cash_flow for item in financial_line_items
                  if hasattr(item, 'free_cash_flow') and item.free_cash_flow is not None]
    net_income_values = [item.net_income for item in financial_line_items
                         if hasattr(item, 'net_income') and item.net_income is not None]

    if fcf_values and net_income_values and len(fcf_values) == len(net_income_values):
        fcf_to_ni_ratios = [fcf_values[i] / net_income_values[i]
                            for i in range(len(fcf_values))
                            if net_income_values[i] and net_income_values[i] > 0]
        if fcf_to_ni_ratios:
            avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
            if avg_ratio > 1.1:
                score += 3
                details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.9:
                score += 2
                details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.7:
                score += 1
                details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            else:
                details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")

    debt_values   = [item.total_debt for item in financial_line_items if hasattr(item, 'total_debt') and item.total_debt is not None]
    equity_values = [item.shareholders_equity for item in financial_line_items if hasattr(item, 'shareholders_equity') and item.shareholders_equity is not None]
    recent_de_ratio = None

    if debt_values and equity_values and len(debt_values) == len(equity_values):
        recent_de_ratio = debt_values[0] / equity_values[0] if equity_values[0] > 0 else float('inf')
        if recent_de_ratio < 0.3:
            score += 3
            details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 0.7:
            score += 2
            details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 1.5:
            score += 1
            details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")

    cash_values    = [item.cash_and_equivalents for item in financial_line_items if hasattr(item, 'cash_and_equivalents') and item.cash_and_equivalents is not None]
    revenue_values = [item.revenue for item in financial_line_items if hasattr(item, 'revenue') and item.revenue is not None]
    cash_to_revenue = None

    if cash_values and revenue_values and revenue_values[0] and revenue_values[0] > 0:
        cash_to_revenue = cash_values[0] / revenue_values[0]
        if 0.1 <= cash_to_revenue <= 0.25:
            score += 2
            details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
            score += 1
            details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif cash_to_revenue > 0.4:
            details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")

    insider_buy_ratio = None
    if insider_trades and len(insider_trades) > 0:
        buys  = sum(1 for t in insider_trades if getattr(t, 'transaction_type', None) and t.transaction_type.lower() in ['buy', 'purchase'])
        sells = sum(1 for t in insider_trades if getattr(t, 'transaction_type', None) and t.transaction_type.lower() in ['sell', 'sale'])
        total = buys + sells
        if total > 0:
            insider_buy_ratio = buys / total
            if insider_buy_ratio > 0.7:
                score += 2
                details.append(f"Strong insider buying: {buys}/{total} transactions are purchases")
            elif insider_buy_ratio > 0.4:
                score += 1
                details.append(f"Balanced insider trading: {buys}/{total} transactions are purchases")
            elif insider_buy_ratio < 0.1 and sells > 5:
                score -= 1
                details.append(f"Concerning insider selling: {sells}/{total} transactions are sales")

    share_counts = [item.outstanding_shares for item in financial_line_items if hasattr(item, 'outstanding_shares') and item.outstanding_shares is not None]
    share_count_trend = "unknown"

    if share_counts and len(share_counts) >= 3:
        if share_counts[0] < share_counts[-1] * 0.95:
            score += 2
            share_count_trend = "decreasing"
            details.append("Shareholder-friendly: Reducing share count over time")
        elif share_counts[0] < share_counts[-1] * 1.05:
            score += 1
            share_count_trend = "stable"
            details.append("Stable share count: Limited dilution")
        elif share_counts[0] > share_counts[-1] * 1.2:
            score -= 1
            share_count_trend = "increasing"
            details.append("Concerning dilution: Share count increased significantly")
        else:
            share_count_trend = "increasing"
            details.append("Moderate share count increase over time")

    final_score = max(0, min(10, score * 10 / 12))
    return {
        "score": final_score,
        "details": "; ".join(details),
        "insider_buy_ratio": insider_buy_ratio,
        "recent_de_ratio": recent_de_ratio,
        "cash_to_revenue": cash_to_revenue,
        "share_count_trend": share_count_trend,
    }


def analyze_predictability(financial_line_items: list) -> dict:
    score = 0
    details = []

    if not financial_line_items or len(financial_line_items) < 5:
        return {"score": 0, "details": "Insufficient data (need 5+ years)"}

    revenues = [item.revenue for item in financial_line_items if hasattr(item, 'revenue') and item.revenue is not None]
    if revenues and len(revenues) >= 5:
        growth_rates = [(revenues[i] / revenues[i+1] - 1) for i in range(len(revenues)-1) if revenues[i+1] != 0]
        if growth_rates:
            avg_growth = sum(growth_rates) / len(growth_rates)
            volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
            if avg_growth > 0.05 and volatility < 0.1:
                score += 3
                details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth, low volatility")
            elif avg_growth > 0 and volatility < 0.2:
                score += 2
                details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth")
            elif avg_growth > 0:
                score += 1
                details.append(f"Growing but volatile revenue: {avg_growth:.1%} avg growth")
            else:
                details.append(f"Declining or unpredictable revenue: {avg_growth:.1%} avg growth")

    op_income = [item.operating_income for item in financial_line_items if hasattr(item, 'operating_income') and item.operating_income is not None]
    if op_income and len(op_income) >= 5:
        positive_periods = sum(1 for x in op_income if x > 0)
        if positive_periods == len(op_income):
            score += 3
            details.append("Operating income positive in all periods")
        elif positive_periods >= len(op_income) * 0.8:
            score += 2
            details.append(f"Operating income positive in {positive_periods}/{len(op_income)} periods")
        elif positive_periods >= len(op_income) * 0.6:
            score += 1
            details.append(f"Operating income positive in {positive_periods}/{len(op_income)} periods")

    op_margins = [item.operating_margin for item in financial_line_items if hasattr(item, 'operating_margin') and item.operating_margin is not None]
    if op_margins and len(op_margins) >= 5:
        avg_margin = sum(op_margins) / len(op_margins)
        volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
        if volatility < 0.03:
            score += 2
            details.append(f"Highly stable margins: {avg_margin:.1%} avg")
        elif volatility < 0.07:
            score += 1
            details.append(f"Moderately stable margins: {avg_margin:.1%} avg")

    fcf_values = [item.free_cash_flow for item in financial_line_items if hasattr(item, 'free_cash_flow') and item.free_cash_flow is not None]
    if fcf_values and len(fcf_values) >= 5:
        positive_fcf = sum(1 for x in fcf_values if x > 0)
        if positive_fcf == len(fcf_values):
            score += 2
            details.append("Positive FCF in all periods")
        elif positive_fcf >= len(fcf_values) * 0.8:
            score += 1
            details.append(f"Positive FCF in {positive_fcf}/{len(fcf_values)} periods")

    final_score = min(10, score * 10 / 10)
    return {"score": final_score, "details": "; ".join(details)}


def calculate_munger_valuation(financial_line_items: list, market_cap: float) -> dict:
    score = 0
    details = []

    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    fcf_values = [item.free_cash_flow for item in financial_line_items if hasattr(item, 'free_cash_flow') and item.free_cash_flow is not None]

    if not fcf_values or len(fcf_values) < 3:
        return {"score": 0, "details": "Insufficient FCF data for valuation"}

    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))

    if normalized_fcf <= 0:
        return {"score": 0, "details": f"Negative normalized FCF ({normalized_fcf})", "intrinsic_value": None}

    if market_cap <= 0:
        return {"score": 0, "details": f"Invalid market cap ({market_cap})"}

    fcf_yield = normalized_fcf / market_cap

    if fcf_yield > 0.08:
        score += 4
        details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.05:
        score += 3
        details.append(f"Good value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.03:
        score += 1
        details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
    else:
        details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")

    conservative_value = normalized_fcf * 10
    reasonable_value   = normalized_fcf * 15
    optimistic_value   = normalized_fcf * 20

    mos = (reasonable_value - market_cap) / market_cap

    if mos > 0.3:
        score += 3
        details.append(f"Large margin of safety: {mos:.1%} upside")
    elif mos > 0.1:
        score += 2
        details.append(f"Moderate margin of safety: {mos:.1%} upside")
    elif mos > -0.1:
        score += 1
        details.append(f"Fair price: within 10% of reasonable value")
    else:
        details.append(f"Expensive: {-mos:.1%} premium to reasonable value")

    if len(fcf_values) >= 3:
        recent_avg = sum(fcf_values[:3]) / 3
        older_avg  = fcf_values[-1] if len(fcf_values) < 6 else sum(fcf_values[-3:]) / 3
        if recent_avg > older_avg * 1.2:
            score += 3
            details.append("Growing FCF trend")
        elif recent_avg > older_avg:
            score += 2
            details.append("Stable to growing FCF")
        else:
            details.append("Declining FCF trend")

    final_score = min(10, score * 10 / 10)
    return {
        "score": final_score,
        "details": "; ".join(details),
        "intrinsic_value_range": {"conservative": conservative_value, "reasonable": reasonable_value, "optimistic": optimistic_value},
        "fcf_yield": fcf_yield,
        "normalized_fcf": normalized_fcf,
        "margin_of_safety_vs_fair_value": mos,
    }


def analyze_news_sentiment(news_items: list) -> str:
    if not news_items:
        return "No news data available"
    return f"Qualitative review of {len(news_items)} recent news items would be needed"


def _r(x, n=3):
    try:
        return round(float(x), n)
    except Exception:
        return None


def make_munger_facts_bundle(analysis: dict) -> dict:
    moat = analysis.get("moat_analysis") or {}
    mgmt = analysis.get("management_analysis") or {}
    pred = analysis.get("predictability_analysis") or {}
    val  = analysis.get("valuation_analysis") or {}
    ivr  = val.get("intrinsic_value_range") or {}

    return {
        "pre_signal":            analysis.get("signal"),
        "score":                 _r(analysis.get("score"), 2),
        "max_score":             _r(analysis.get("max_score"), 2),
        "moat_score":            _r(moat.get("score"), 2),
        "mgmt_score":            _r(mgmt.get("score"), 2),
        "predictability_score":  _r(pred.get("score"), 2),
        "valuation_score":       _r(val.get("score"), 2),
        "fcf_yield":             _r(val.get("fcf_yield"), 4),
        "normalized_fcf":        _r(val.get("normalized_fcf"), 0),
        "reasonable_value":      _r(ivr.get("reasonable"), 0),
        "margin_of_safety":      _r(val.get("margin_of_safety_vs_fair_value"), 3),
        "insider_buy_ratio":     _r(mgmt.get("insider_buy_ratio"), 2),
        "recent_de_ratio":       _r(mgmt.get("recent_de_ratio"), 2),
        "cash_to_revenue":       _r(mgmt.get("cash_to_revenue"), 2),
        "share_count_trend":     mgmt.get("share_count_trend"),
        "notes": {
            "moat":           (moat.get("details") or "")[:120],
            "mgmt":           (mgmt.get("details") or "")[:120],
            "predictability": (pred.get("details") or "")[:120],
            "valuation":      (val.get("details") or "")[:120],
        },
    }


def compute_confidence(analysis: dict, signal: str) -> float:
    moat = float((analysis.get("moat_analysis") or {}).get("score") or 0)
    mgmt = float((analysis.get("management_analysis") or {}).get("score") or 0)
    pred = float((analysis.get("predictability_analysis") or {}).get("score") or 0)
    val  = float((analysis.get("valuation_analysis") or {}).get("score") or 0)

    quality     = 0.35 * moat + 0.25 * mgmt + 0.25 * pred
    quality_pct = quality / 8.5 if quality > 0 else 0  # 0..1

    mos     = float((analysis.get("valuation_analysis") or {}).get("margin_of_safety_vs_fair_value") or 0)
    val_adj = max(-0.10, min(0.10, mos / 3.0))

    base = 0.85 * quality_pct + 0.15 * (val / 10) + val_adj
    base = max(0.1, min(1.0, base))
    return round(base, 2)


def generate_munger_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    confidence_hint: float,
) -> AgentOutput:
    facts_bundle = make_munger_facts_bundle(analysis_data)

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are Charlie Munger. Decide bullish, bearish, or neutral using only the facts. "
            "Return JSON only. Keep reasoning under 120 characters.\n\n"
            "Decision rules:\n"
            "- direction=LONG:    moat strong AND predictable AND fair/cheap valuation\n"
            "- direction=SHORT:   poor moat OR unpredictable OR expensive\n"
            "- direction=NEUTRAL: mixed evidence\n\n"
            "expected_return: realistic % return for a 3-4 day swing trade (e.g. 0.02 = +2%). "
            "Positive for LONG, negative for SHORT, ~0 for NEUTRAL. Range: -0.10 to +0.10.\n"
            "confidence: use the provided hint value exactly."
        ),
        (
            "human",
            "Ticker: {ticker}\n"
            "Facts:\n{facts}\n"
            "Confidence hint: {confidence}\n\n"
            "Return exactly:\n"
            "{{\n"
            '  "direction": "LONG" | "SHORT" | "NEUTRAL",\n'
            '  "expected_return": float,\n'
            '  "confidence": float,\n'
            '  "reasoning": "short justification"\n'
            "}}"
        ),
    ])

    prompt = template.invoke({
        "ticker":     ticker,
        "facts":      json.dumps(facts_bundle, separators=(",", ":"), ensure_ascii=False),
        "confidence": confidence_hint,
    }).to_string()

    def _default():
        return AgentOutput(direction="NEUTRAL", expected_return=0.0, confidence=0.1, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=AgentOutput,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )
