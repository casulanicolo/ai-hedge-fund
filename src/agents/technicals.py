"""
src/agents/technicals.py
Technical Analyst Agent – Fase 3.1
Legge da prefetched_data (nessuna API diretta), calcola indicatori avanzati,
rileva il regime di mercato e produce un segnale strutturato via LLM.
"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, AgentOutput, show_agent_reasoning
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.data.state_reader import get_ohlcv_daily, get_ohlcv_5m
from src.indicators.technical_indicators import compute_indicator_snapshot
from src.indicators.regime_detector import detect_regime


##### Technical Analyst Agent #####
def technical_analyst_agent(state: AgentState, agent_id: str = "technical_analyst_agent"):
    """
    Agente di analisi tecnica avanzata.

    Per ogni ticker:
    1. Legge OHLCV da prefetched_data (no API)
    2. Calcola RSI, MACD, Bollinger, ATR, VWAP, OBV, Ichimoku, Fibonacci
    3. Analisi multi-timeframe (daily + intraday 5m se disponibile)
    4. Rileva il regime: TRENDING / RANGING / VOLATILE
    5. Passa snapshot + regime all'LLM per generare il segnale finale
    6. Output strutturato: {direction, expected_return, confidence, reasoning}
    """
    data = state["data"]
    tickers = data["tickers"]

    technical_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Reading OHLCV from prefetched data")

        # --- 1. Leggi DataFrame dallo state ---
        daily_df    = get_ohlcv_daily(state, ticker)
        intraday_df = get_ohlcv_5m(state, ticker)

        if daily_df is not None:
            daily_df = _normalize_df(daily_df)
        if intraday_df is not None:
            intraday_df = _normalize_df(intraday_df)

        # --- 2. Calcola indicatori ---
        progress.update_status(agent_id, ticker, "Computing indicators")
        snapshot = compute_indicator_snapshot(daily_df, intraday_df)

        if not snapshot:
            progress.update_status(agent_id, ticker, "Failed: insufficient price data")
            technical_analysis[ticker] = {
                "direction":       "NEUTRAL",
                "expected_return": 0.0,
                "confidence":      0.1,
                "indicators":      {},
                "regime":          "UNKNOWN",
                "regime_direction": "FLAT",
                "reasoning":       "Insufficient price data for technical analysis.",
            }
            continue

        snapshot["has_intraday"] = intraday_df is not None
        snapshot["daily_bars"]   = len(daily_df) if daily_df is not None else 0

        # --- 3. Rileva regime ---
        progress.update_status(agent_id, ticker, "Detecting market regime")
        regime_info = detect_regime(snapshot)

        # --- 4. Genera segnale via LLM ---
        progress.update_status(agent_id, ticker, "Generating LLM signal")
        signal_result = _generate_signal_via_llm(state, ticker, snapshot, regime_info, agent_id)

        # --- 5. Assembla output ---
        technical_analysis[ticker] = {
            "direction":        signal_result.direction,
            "expected_return":  signal_result.expected_return,
            "confidence":       signal_result.confidence,
            "regime":           regime_info["regime"],
            "regime_direction": regime_info["direction"],
            "indicators":       _build_indicators_summary(snapshot),
            "reasoning":        signal_result.reasoning,
        }

        progress.update_status(
            agent_id, ticker, "Done",
            analysis=json.dumps(technical_analysis[ticker], indent=2)
        )

    # --- Messaggio LangGraph ---
    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    state["data"]["analyst_signals"][agent_id] = technical_analysis
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


# ---------------------------------------------------------------------------
# Helper: normalizza colonne DataFrame
# ---------------------------------------------------------------------------

def _normalize_df(df):
    import pandas as pd
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LLM signal generation
# ---------------------------------------------------------------------------

def _generate_signal_via_llm(
    state: AgentState,
    ticker: str,
    snapshot: dict,
    regime_info: dict,
    agent_id: str,
) -> AgentOutput:

    regime    = regime_info["regime"]
    direction = regime_info["direction"]
    reg_conf  = round(regime_info["confidence"] * 100)
    reg_reasons = " ".join(regime_info["reasons"])

    price        = snapshot.get("price", 0)
    rsi_14       = snapshot.get("rsi_14", 50)
    macd_hist    = snapshot.get("macd_histogram", 0)
    macd_bullish = snapshot.get("macd_bullish", False)
    bb_pct       = snapshot.get("bb_pct", 0.5)
    atr_pct      = snapshot.get("atr_pct", 0)
    adx          = snapshot.get("adx", 20)
    strong_trend = snapshot.get("strong_trend", False)
    plus_di      = snapshot.get("plus_di", 20)
    minus_di     = snapshot.get("minus_di", 20)
    price_vwap   = snapshot.get("price_vs_vwap", 0)
    obv_trend    = snapshot.get("obv_trend", "flat")
    above_cloud  = snapshot.get("above_cloud", False)
    below_cloud  = snapshot.get("below_cloud", False)
    ema_8        = snapshot.get("ema_8", 0)
    ema_21       = snapshot.get("ema_21", 0)
    ema_55       = snapshot.get("ema_55", 0)
    nearest_sup  = snapshot.get("nearest_support", "N/A")
    hurst        = snapshot.get("hurst", 0.5)
    intra_rsi    = snapshot.get("intraday_rsi_14")
    intra_macd   = snapshot.get("intraday_macd_hist")

    ichimoku_pos = (
        "ABOVE cloud" if above_cloud
        else "BELOW cloud" if below_cloud
        else "INSIDE cloud"
    )

    intraday_line = "- Intraday data: not available"
    if intra_rsi is not None:
        intraday_line = f"- RSI 14 intraday 5m: {intra_rsi:.1f}"
        if intra_macd is not None:
            intraday_line += f" | MACD hist 5m: {intra_macd:.4f}"

    regime_instruction = {
        "TRENDING":  "Follow the trend. Weight ADX, EMA alignment, Ichimoku position, OBV direction.",
        "RANGING":   "Use mean-reversion logic. Weight RSI extremes, BB %B, VWAP, Fibonacci support/resistance.",
        "VOLATILE":  "Be cautious. Lower confidence. Note ATR% and BB width as primary risk factors.",
    }.get(regime, "")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a quantitative technical analyst. Analyze the indicators and produce a structured trading signal."
        ),
        (
            "human",
            f"""Analyze the following technical indicators for {ticker} and produce a trading signal.

MARKET REGIME: {regime} ({direction}) – confidence {reg_conf}%
Regime reasons: {reg_reasons}
Regime instruction: {regime_instruction}

TECHNICAL INDICATORS (daily):
- Price:               {price:.2f}
- RSI 14:              {rsi_14:.1f}
- MACD histogram:      {macd_hist:.4f} ({'positive' if macd_bullish else 'negative'})
- BB %B:               {bb_pct:.2f} (0=lower band, 1=upper band)
- ATR%:                {atr_pct:.2%}
- ADX:                 {adx:.1f} ({'strong trend' if strong_trend else 'weak trend'})
- +DI / -DI:           {plus_di:.1f} / {minus_di:.1f}
- Price vs VWAP:       {price_vwap:.2%}
- OBV trend:           {obv_trend}
- Ichimoku position:   {ichimoku_pos}
- EMA 8 / 21 / 55:     {ema_8:.2f} / {ema_21:.2f} / {ema_55:.2f}
- Nearest Fib support: {nearest_sup}
- Hurst exponent:      {hurst:.2f}
{intraday_line}

Respond with:
- direction: LONG, SHORT, or NEUTRAL
- expected_return: your realistic return estimate for a 3-4 day swing trade (e.g. 0.025 = +2.5%). Must reflect a realistic target, not a theoretical maximum. Range: -0.10 to +0.10.
- confidence: your signal quality score from 0.1 to 1.0
- reasoning: max 3 sentences explaining the signal
"""
        ),
    ])

    def default_factory():
        return AgentOutput(
            direction="NEUTRAL",
            expected_return=0.0,
            confidence=0.1,
            reasoning=f"LLM unavailable. Rule-based fallback: regime {regime}, RSI {rsi_14:.1f}.",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=AgentOutput,
        agent_name=agent_id,
        state=state,
        default_factory=default_factory,
    )


# ---------------------------------------------------------------------------
# Helper: summary indicatori per il report
# ---------------------------------------------------------------------------

def _build_indicators_summary(snapshot: dict) -> dict:
    keys = [
        "price", "rsi_14", "rsi_28",
        "macd", "macd_signal", "macd_histogram", "macd_bullish",
        "bb_upper", "bb_lower", "bb_pct", "bb_width",
        "atr", "atr_pct",
        "adx", "plus_di", "minus_di", "strong_trend",
        "vwap", "price_vs_vwap",
        "obv_trend",
        "tenkan", "kijun", "above_cloud", "below_cloud", "in_cloud",
        "nearest_support",
        "ema_8", "ema_21", "ema_55",
        "hurst",
        "intraday_rsi_14", "intraday_macd_hist",
        "has_intraday", "daily_bars",
    ]
    return {k: snapshot[k] for k in keys if k in snapshot}
