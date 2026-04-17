"""
src/agents/macro_agent.py
─────────────────────────
Athanor Alpha — Macro Environment Agent (FIX C2).

Dimension: MACRO
Data source: state["data"]["macro_data"] (populated by data_prefetch_agent)
Fallback: direct fetch via MacroFetcher if macro_data absent

Signals consumed (all from yfinance, zero external APIs):
  - VIX level and 10-day trend
  - Yield spread (10Y - 3M): positive = healthy, negative = inverted/recession risk
  - 20-day rate trend (is 10Y rising or falling?)

Architecture:
  1. Read macro_data from state
  2. Compute deterministic macro scores (0-100 scale)
  3. ONE global LLM call → macro direction + reasoning
  4. Apply per-ticker sensitivity multiplier:
       crypto/high-beta: amplified confidence (more macro-sensitive)
       defensives: dampened confidence (less macro-sensitive)

Output written to state["data"]["analyst_signals"]["macro_agent"]:
  {ticker: {direction, expected_return, confidence, reasoning,
            macro_regime, vix, yield_spread}}
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, AgentOutput, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.trade_levels import compute_trade_levels

logger = logging.getLogger(__name__)

AGENT_ID = "macro_agent"

# ── Per-ticker macro sensitivity ──────────────────────────────────────────────
# Multiplier applied to confidence (and expected_return) for each ticker.
# Reflects how much a ticker's price is driven by macro conditions.
# Default = 1.0 for tickers not listed.

TICKER_SENSITIVITY: dict[str, float] = {
    # Crypto — most macro-reactive (risk-on/risk-off plays)
    "BTC-USD": 2.0,
    "ETH-USD": 2.0,
    "SOL-USD": 2.0,
    "BNB-USD": 2.0,
    # Crypto-proxy equities
    "MSTR": 1.8,
    "COIN": 1.8,
    # High-beta growth
    "TSLA": 1.5,
    "SMCI": 1.5,
    "NVDA": 1.4,
    "MELI": 1.3,
    "META": 1.2,
    "AMZN": 1.1,
    # Broad market
    "AAPL": 0.9,
    "MSFT": 0.9,
    "GOOGL": 1.0,
    # Defensive / financials (less macro-reactive for short swings)
    "JPM":  0.8,
    "V":    0.8,
    "UNH":  0.6,
}


# ── Macro score computation ───────────────────────────────────────────────────

def _compute_macro_scores(macro_data: dict) -> dict:
    """
    Deterministic scoring of macro indicators (0-100 scale, higher = more bullish macro).

    Weights:
      VIX score:       45%  (primary fear gauge)
      Yield curve:     40%  (leading recession indicator)
      Rate trend:      15%  (momentum of rates)
    """
    vix             = macro_data.get("vix")
    yield_spread    = macro_data.get("yield_spread_10y3m")
    yield_10y       = macro_data.get("yield_10y")
    yield_3m        = macro_data.get("yield_3m")
    rate_trend_20d  = macro_data.get("rate_trend_20d")
    vix_trend_10d   = macro_data.get("vix_trend_10d")

    # ── VIX score ─────────────────────────────────────────────────────────────
    if vix is not None:
        if vix < 15:
            vix_score = 90
            vix_regime = "CALM"
        elif vix < 18:
            vix_score = 80
            vix_regime = "CALM"
        elif vix < 25:
            vix_score = 60
            vix_regime = "NORMAL"
        elif vix < 35:
            vix_score = 30
            vix_regime = "ELEVATED"
        else:
            vix_score = 10
            vix_regime = "CRISIS"
    else:
        vix_score  = 50
        vix_regime = "UNKNOWN"

    # ── Yield curve score ─────────────────────────────────────────────────────
    # Deeply inverted = recession risk = bearish for risk assets
    if yield_spread is not None:
        if yield_spread > 1.5:
            yc_score = 90     # very steep — strong growth expectations
        elif yield_spread > 0.5:
            yc_score = 72
        elif yield_spread > 0.0:
            yc_score = 55     # mildly positive — neutral
        elif yield_spread > -0.5:
            yc_score = 38     # flat to slightly inverted — caution
        elif yield_spread > -1.0:
            yc_score = 22     # clearly inverted — recession risk
        else:
            yc_score = 10     # deeply inverted
    else:
        yc_score = 50

    # ── Rate trend score ──────────────────────────────────────────────────────
    # Rapidly rising rates = equity headwind; falling rates = tailwind
    if rate_trend_20d is not None:
        if rate_trend_20d < -0.50:       # rates falling fast → bullish
            rate_trend_score = 78
        elif rate_trend_20d < -0.20:
            rate_trend_score = 65
        elif rate_trend_20d <= 0.20:     # flat
            rate_trend_score = 52
        elif rate_trend_20d <= 0.50:     # rising moderately
            rate_trend_score = 38
        else:                            # rising fast → equity headwind
            rate_trend_score = 25
    else:
        rate_trend_score = 50

    # ── Composite ────────────────────────────────────────────────────────────
    composite = round(
        vix_score * 0.45 +
        yc_score  * 0.40 +
        rate_trend_score * 0.15,
        1,
    )

    # ── Regime classification ─────────────────────────────────────────────────
    if composite >= 62:
        regime = "RISK_ON"
    elif composite <= 38:
        regime = "RISK_OFF"
    else:
        regime = "CAUTION"

    return {
        "composite":         composite,
        "vix_score":         round(vix_score, 1),
        "yc_score":          round(yc_score, 1),
        "rate_trend_score":  round(rate_trend_score, 1),
        "regime":            regime,
        "vix_regime":        vix_regime,
        "vix":               vix,
        "yield_10y":         yield_10y,
        "yield_3m":          yield_3m,
        "yield_spread":      yield_spread,
        "rate_trend_20d":    rate_trend_20d,
        "vix_trend_10d":     vix_trend_10d,
    }


# ── LLM: global macro direction ──────────────────────────────────────────────

def _get_macro_direction(scores: dict, state: AgentState) -> AgentOutput:
    """One LLM call for the global macro assessment (not per-ticker)."""

    regime = scores["regime"]
    vix_str = f"{scores['vix']:.1f}" if scores["vix"] is not None else "N/A"
    spread_str = f"{scores['yield_spread']:.3f}" if scores["yield_spread"] is not None else "N/A"
    tnx_str = f"{scores['yield_10y']:.2f}%" if scores["yield_10y"] is not None else "N/A"
    irx_str = f"{scores['yield_3m']:.2f}%" if scores["yield_3m"] is not None else "N/A"
    trend_str = f"{scores['rate_trend_20d']:+.3f}pp" if scores["rate_trend_20d"] is not None else "N/A"

    prompt = f"""You are a macro economist evaluating the global investment environment for 3-4 day equity swing trades.

MACRO INDICATORS (from yfinance):
- VIX:                 {vix_str}  (regime: {scores['vix_regime']})
- 10Y Treasury Yield:  {tnx_str}
- 3M T-Bill Rate:      {irx_str}
- Yield Spread 10Y-3M: {spread_str} pp  (positive = normal curve, negative = inverted)
- 10Y Rate Trend 20d:  {trend_str}  (positive = rising rates)

MACRO SCORES (0-100, higher = more bullish macro):
- VIX score:         {scores['vix_score']}/100  (weight 45%)
- Yield curve score: {scores['yc_score']}/100   (weight 40%)
- Rate trend score:  {scores['rate_trend_score']}/100  (weight 15%)
- COMPOSITE:         {scores['composite']}/100
- MACRO REGIME:      {regime}

DECISION RULES:
- LONG  (RISK_ON ≥ 62):  calm VIX + normal/steep yield curve → equities favored
- SHORT (RISK_OFF ≤ 38): high VIX OR deeply inverted yield curve → risk assets under pressure
- NEUTRAL (CAUTION):     mixed signals, unclear macro direction

Respond in JSON:
- direction: "LONG" | "SHORT" | "NEUTRAL"
- expected_return: realistic macro-driven return for 3-4 day swing (-0.03 to +0.03).
  Macro tailwinds/headwinds are real but modest at this horizon — keep magnitude realistic.
- confidence: 0.1-1.0 (higher = clearer macro signal)
- reasoning: 2-3 sentences describing the macro environment and why it favors LONG/SHORT/NEUTRAL
"""

    def _default_factory() -> AgentOutput:
        if regime == "RISK_ON":
            return AgentOutput(
                direction="LONG", expected_return=0.010, confidence=0.4,
                reasoning=f"LLM unavailable. Macro regime RISK_ON (composite={scores['composite']:.0f}).",
            )
        if regime == "RISK_OFF":
            return AgentOutput(
                direction="SHORT", expected_return=-0.010, confidence=0.4,
                reasoning=f"LLM unavailable. Macro regime RISK_OFF (composite={scores['composite']:.0f}).",
            )
        return AgentOutput(
            direction="NEUTRAL", expected_return=0.0, confidence=0.3,
            reasoning=f"LLM unavailable. Macro regime CAUTION (composite={scores['composite']:.0f}).",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=AgentOutput,
        agent_name=AGENT_ID,
        state=state,
        default_factory=_default_factory,
    )


# ── Per-ticker sensitivity application ───────────────────────────────────────

def _apply_sensitivity(
    direction: str,
    expected_return: float,
    confidence: float,
    sensitivity: float,
) -> tuple[str, float, float]:
    """
    Scale confidence and expected_return by ticker macro sensitivity.
    Sensitivity > 1.0 amplifies the macro signal (crypto, high-beta).
    Sensitivity < 1.0 dampens it (defensives).
    Caps: confidence ≤ 1.0, |expected_return| ≤ 0.05.
    """
    scaled_confidence = min(1.0, confidence * sensitivity)
    scaled_er = max(-0.05, min(0.05, expected_return * sensitivity))
    return direction, round(scaled_er, 4), round(scaled_confidence, 3)


# ── Critical Alert ───────────────────────────────────────────────────────────

def _check_macro_alert(scores: dict, macro_data: dict) -> None:
    """
    Print a visible ASCII banner to stdout and log at CRITICAL level when
    macro_agent detects RISK_OFF regime (VIX spike OR deeply inverted curve).
    Called once per run, immediately after scores are computed.
    """
    if scores.get("regime") != "RISK_OFF":
        return

    vix        = scores.get("vix")
    spread     = scores.get("yield_spread")
    composite  = scores.get("composite", 0.0)
    vix_str    = f"{vix:.1f}" if vix is not None else "N/A"
    spread_str = f"{spread:+.3f}pp" if spread is not None else "N/A"

    banner = (
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║          ⚠  MACRO CRITICAL ALERT — RISK_OFF REGIME  ⚠       ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        f"║  Composite score : {composite:5.1f}/100  (threshold ≤38)             ║\n"
        f"║  VIX             : {vix_str:<10s}  (regime: {scores.get('vix_regime','?'):<10s})     ║\n"
        f"║  Yield spread    : {spread_str:<10s}  (10Y - 3M)                  ║\n"
        "║                                                              ║\n"
        "║  ACTION: All position caps hard-limited to 5%.              ║\n"
        "║          Review open positions for immediate risk reduction.  ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

    print(banner, flush=True)
    logger.critical(
        "MACRO RISK_OFF: composite=%.1f VIX=%s spread=%s — position cap=5%%",
        composite, vix_str, spread_str,
    )


# ── Main agent node ───────────────────────────────────────────────────────────

def macro_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node — Macro Environment Agent.

    Reads global macro data from state["data"]["macro_data"].
    Produces per-ticker macro signals based on VIX + yield curve analysis.
    """
    data:    dict[str, Any] = state.get("data", {})
    tickers: list[str]      = data.get("tickers", [])

    # ── 1. Read macro data (fetched by data_prefetch_agent) ──────────────────
    macro_data: dict = data.get("macro_data", {})

    if not macro_data:
        # Graceful fallback: try a direct fetch if data_prefetch skipped it
        try:
            from src.data.macro_fetcher import MacroFetcher
            macro_data = MacroFetcher.fetch()
            logger.info("macro_agent: direct fetch fallback succeeded")
        except Exception as exc:
            logger.warning(
                "macro_agent: macro_data absent and direct fetch failed (%s) — "
                "defaulting all tickers to NEUTRAL.",
                exc,
            )
            macro_data = {}

    # ── 2. Compute macro scores ───────────────────────────────────────────────
    scores = _compute_macro_scores(macro_data)

    progress.update_status(
        AGENT_ID, None,
        f"Regime={scores['regime']} composite={scores['composite']:.0f} "
        f"VIX={scores.get('vix') or 'N/A'} spread={scores.get('yield_spread') or 'N/A'}pp",
    )

    # ── Critical alert (RISK_OFF banner) ─────────────────────────────────────
    _check_macro_alert(scores, macro_data)

    # ── 3. One LLM call for global macro direction ───────────────────────────
    if macro_data:
        macro_signal = _get_macro_direction(scores, state)
    else:
        # No data at all → neutral with minimal confidence
        macro_signal = AgentOutput(
            direction="NEUTRAL",
            expected_return=0.0,
            confidence=0.1,
            reasoning="No macro data available — defaulting to NEUTRAL.",
        )

    logger.info(
        "macro_agent: global signal=%s er=%+.3f conf=%.2f",
        macro_signal.direction, macro_signal.expected_return, macro_signal.confidence,
    )

    # ── 4. Apply per-ticker sensitivity ──────────────────────────────────────
    signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        sensitivity = TICKER_SENSITIVITY.get(ticker, 1.0)

        direction, scaled_er, scaled_conf = _apply_sensitivity(
            macro_signal.direction,
            macro_signal.expected_return,
            macro_signal.confidence,
            sensitivity,
        )

        sensitivity_note = f" [macro sensitivity {sensitivity:.1f}x]" if sensitivity != 1.0 else ""
        reasoning = macro_signal.reasoning + sensitivity_note

        levels = compute_trade_levels(direction, state, ticker)

        signals[ticker] = {
            "direction":       direction,
            "expected_return": scaled_er,
            "confidence":      scaled_conf,
            "reasoning":       reasoning,
            "macro_regime":    scores["regime"],
            "vix":             scores.get("vix"),
            "yield_spread":    scores.get("yield_spread"),
            "macro_composite": scores["composite"],
            **levels,
        }

        progress.update_status(
            AGENT_ID, ticker,
            f"{direction} er={scaled_er:+.3f} conf={scaled_conf:.2f} "
            f"({scores['regime']} sensitivity={sensitivity:.1f}x)",
        )

    # ── 5. Write to state ─────────────────────────────────────────────────────
    data["analyst_signals"][AGENT_ID] = signals

    show_agent_reasoning(
        {
            "regime": scores["regime"],
            "composite": scores["composite"],
            "vix": scores.get("vix"),
            "yield_spread": scores.get("yield_spread"),
            "global_signal": macro_signal.direction,
            "tickers": {t: s["direction"] for t, s in signals.items()},
        },
        "Macro Agent",
    )

    progress.update_status(AGENT_ID, None, "Done")

    summary = (
        f"Macro Agent | {scores['regime']} composite={scores['composite']:.0f} | "
        f"VIX={scores.get('vix') or 'N/A'} | "
        f"spread={scores.get('yield_spread') or 'N/A'}pp | "
        f"global={macro_signal.direction}"
    )
    msg = HumanMessage(content=summary, name=AGENT_ID)

    return {"messages": [msg], "data": data}
