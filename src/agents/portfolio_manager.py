"""
portfolio_manager.py – Athanor Alpha Phase 3.5
Reads analyst_signals + risk_report from state.
Produces per-ticker recommendations:
  {ticker, action, sizing_pct, conviction, weighted_conviction, reasoning,
   entry_price, stop_loss, take_profit, size_usd, rr_ratio, consensus}
No execution – output only. Writes to state["data"]["portfolio_recommendations"].
"""

from __future__ import annotations

import json
import os
from math import floor
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress

AGENT_ID = "portfolio_manager"

# ── Sizing parameters ────────────────────────────────────────────────────────
MAX_SINGLE_POSITION   = 20.0
MIN_SINGLE_POSITION   = 2.0
TOTAL_GROSS_BUDGET    = 100.0
KELLY_FRACTION        = 0.25
CORR_PENALTY_THRESHOLD = 0.70
CORR_PENALTY_FACTOR   = 0.70

# ── Signal quality thresholds ─────────────────────────────────────────────────
NET_SCORE_THRESHOLD     = 0.25
MIN_CONSENSUS_RATIO     = 0.65
MIN_CONVICTION_TO_TRADE = 0.30
WC_THRESHOLD            = 0.008   # Weighted Conviction minima per entrare (0.8%)

# ── Dollar-risk parameters ────────────────────────────────────────────────────
PORTFOLIO_SIZE_USD = float(os.getenv("PORTFOLIO_SIZE_USD", "10000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.005"))
MAX_ACTIVE_TRADES  = int(os.getenv("MAX_ACTIVE_TRADES", "3"))

NON_SIGNAL_AGENTS = {"risk_manager", "data_prefetch"}


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic output model
# ══════════════════════════════════════════════════════════════════════════════

class TickerRecommendation(BaseModel):
    ticker:     str   = Field(description="Equity ticker symbol")
    action:     str   = Field(description="BUY | SELL | HOLD")
    sizing_pct: float = Field(description="Recommended portfolio weight in percent (0-100)")
    conviction: float = Field(description="Conviction score 0.0-1.0")
    reasoning:  str   = Field(description="One-sentence rationale")


class PortfolioOutput(BaseModel):
    recommendations:   list[TickerRecommendation] = Field(description="One recommendation per ticker")
    portfolio_summary: str = Field(description="2-3 sentence summary of the overall portfolio stance")
    risk_notes:        str = Field(description="Any risk warnings the human should be aware of")


# ══════════════════════════════════════════════════════════════════════════════
# Weighted Conviction (Step 1.5)
# ══════════════════════════════════════════════════════════════════════════════

def compute_weighted_conviction(signals: list[dict]) -> float:
    """
    WC = sum(expected_return_i * confidence_i) / n_non_neutral

    signals: list of dicts with keys 'direction', 'expected_return', 'confidence'.
    Returns WC as a float. Positive = net bullish, negative = net bearish.
    If no non-neutral agents, returns 0.0.
    """
    non_neutral = [s for s in signals if s.get("direction", "NEUTRAL") != "NEUTRAL"]
    if not non_neutral:
        return 0.0

    wc = sum(
        s.get("expected_return", 0.0) * s.get("confidence", 0.0)
        for s in non_neutral
    ) / len(non_neutral)

    return round(wc, 6)


def _collect_signals_for_ticker(state: AgentState, ticker: str) -> list[dict]:
    """
    Collect all agent signals for a given ticker that have the new
    direction/expected_return/confidence format.
    """
    analyst_signals = state.get("data", {}).get("analyst_signals", {})
    signals = []

    for agent_id, agent_data in analyst_signals.items():
        if agent_id in NON_SIGNAL_AGENTS:
            continue
        if not isinstance(agent_data, dict):
            continue

        if ticker in agent_data:
            sig = agent_data[ticker]
        elif agent_data.get("ticker") == ticker:
            sig = agent_data
        else:
            continue

        if not isinstance(sig, dict):
            continue

        # Only include signals with the new format
        if "direction" in sig and "expected_return" in sig and "confidence" in sig:
            signals.append({
                "agent_id":        agent_id,
                "direction":       sig.get("direction", "NEUTRAL"),
                "expected_return": float(sig.get("expected_return", 0.0)),
                "confidence":      float(sig.get("confidence", 0.5)),
            })

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# Legge posizioni aperte da SQLite
# ══════════════════════════════════════════════════════════════════════════════

def _load_open_positions() -> list[dict]:
    try:
        from src.portfolio.manager import get_open_positions
        return get_open_positions()
    except Exception:
        return []


def _format_positions_for_prompt(positions: list[dict]) -> str:
    if not positions:
        return "Nessuna posizione aperta. Portafoglio in cash."

    lines = []
    for p in positions:
        ticker    = p["ticker"]
        direction = p["direction"]
        entry     = p["entry_price"]
        sl        = p["stop_loss"]
        tp        = p["take_profit"]
        size      = p["size_usd"]
        opened    = p["opened_at"][:10]
        note      = f" [{p['note']}]" if p.get("note") else ""
        lines.append(
            f"  {ticker}: {direction} | entry={entry:.2f} | SL={sl:.2f} | TP={tp:.2f} "
            f"| size=${size:,.0f} | aperta il {opened}{note}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Signal aggregation with agent weights
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_signals(state: AgentState, tickers: list[str]) -> dict[str, dict]:
    """
    For each ticker compute a weighted net score across all agents.
    Also computes Weighted Conviction (WC) using the new AgentOutput format.
    Returns {ticker: {net_score, avg_confidence, bull_agents, bear_agents,
                       net_signal, weighted_conviction}}
    """
    analyst_signals: dict[str, Any] = state.get("data", {}).get("analyst_signals", {})
    agent_weights:   dict[str, Any] = state.get("data", {}).get("agent_weights", {})

    result: dict[str, dict] = {}

    for ticker in tickers:
        net_score    = 0.0
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

            if ticker in agent_data:
                sig_data = agent_data[ticker]
            elif agent_data.get("ticker") == ticker:
                sig_data = agent_data
            else:
                continue

            if not isinstance(sig_data, dict):
                continue

            # Support both old format (signal) and new format (direction)
            direction  = sig_data.get("direction")
            signal_raw = sig_data.get("signal", "neutral")

            if direction:
                signal = "bullish" if direction == "LONG" else ("bearish" if direction == "SHORT" else "neutral")
            else:
                signal = str(signal_raw).lower()

            if signal == "skip":
                continue

            confidence = float(sig_data.get("confidence", 0.5))
            if confidence > 1.0:
                confidence /= 100.0
            confidence = max(0.0, min(1.0, confidence))

            weight = 1.0
            if agent_weights:
                w_entry = agent_weights.get(agent_id, {})
                if isinstance(w_entry, dict):
                    weight = float(w_entry.get(ticker, w_entry.get("default", 1.0)))
                elif isinstance(w_entry, (int, float)):
                    weight = float(w_entry)

            if signal == "bullish":
                score = +1.0 * confidence * weight
                bull_agents.append(agent_id)
            elif signal == "bearish":
                score = -1.0 * confidence * weight
                bear_agents.append(agent_id)
            else:
                score = 0.0

            net_score    += score
            total_weight += weight
            conf_sum     += confidence
            n            += 1

        # Compute Weighted Conviction using new AgentOutput format
        ticker_signals = _collect_signals_for_ticker(state, ticker)
        wc = compute_weighted_conviction(ticker_signals)

        if n == 0:
            result[ticker] = {
                "net_score": 0.0, "avg_confidence": 0.0,
                "bull_agents": [], "bear_agents": [],
                "net_signal": "neutral", "n_agents": 0,
                "weighted_conviction": wc,
            }
            continue

        norm_score = net_score / total_weight if total_weight > 0 else 0.0
        avg_conf   = conf_sum / n

        if norm_score > NET_SCORE_THRESHOLD:
            net_signal = "bullish"
        elif norm_score < -NET_SCORE_THRESHOLD:
            net_signal = "bearish"
        else:
            net_signal = "neutral"

        result[ticker] = {
            "net_score":           round(norm_score, 4),
            "avg_confidence":      round(avg_conf, 3),
            "bull_agents":         bull_agents,
            "bear_agents":         bear_agents,
            "net_signal":          net_signal,
            "n_agents":            n,
            "weighted_conviction": wc,
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Sizing logic
# ══════════════════════════════════════════════════════════════════════════════

def _load_risk_params() -> dict:
    """
    Carica i parametri di rischio da config/risk_params.yaml.
    Se il file non esiste o c'è un errore, usa valori di default conservativi.
    """
    defaults = {
        "target_daily_vol": 0.01,
        "kelly_fraction":   0.5,
        "max_position_pct": 0.15,
        "min_position_pct": 0.02,
        "wc_threshold":     0.008,
        "sector_cap":       0.30,
    }
    try:
        import yaml
        with open("config/risk_params.yaml", "r") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            defaults.update(loaded)
    except Exception:
        pass
    return defaults


def compute_vol_adjusted_size(atr_pct: float, params: dict) -> float:
    """
    Calcola la dimensione della posizione basandosi sulla volatilità del titolo
    (Volatility Targeting).

    Logica:
      raw_size = target_daily_vol / atr_pct

    Esempio:
      - target_vol=1%, ATR%=2%  → raw_size=50%  → clippato a max 15%
      - target_vol=1%, ATR%=0.5% → raw_size=200% → clippato a max 15%
      - target_vol=1%, ATR%=5%  → raw_size=20%  → clippato a max 15%

    Restituisce una percentuale del portafoglio come float (es. 0.08 = 8%).
    """
    target_vol    = float(params.get("target_daily_vol", 0.01))
    max_pos       = float(params.get("max_position_pct", 0.15))
    min_pos       = float(params.get("min_position_pct", 0.02))

    if atr_pct <= 0:
        return min_pos

    raw_size = target_vol / atr_pct

    # Clippa tra min e max
    sized = max(min_pos, min(raw_size, max_pos))
    return round(sized, 4)


def compute_half_kelly(p_win: float, reward_ratio: float, params: dict) -> float:
    """
    Calcola la dimensione della posizione con la formula Half-Kelly.

    Kelly intero: f = (p_win * (reward_ratio + 1) - 1) / reward_ratio
    Half-Kelly:   half_f = f * kelly_fraction  (default 0.5)

    Parametri:
      p_win         – probabilità stimata di vincere (usa avg_confidence degli agenti LONG)
      reward_ratio  – expected_return / stop_loss stimato (es. target 2%, SL 1% → ratio=2.0)
      params        – dizionario da risk_params.yaml

    Esempi:
      p_win=0.6, reward_ratio=2.0 → Kelly=0.40 → Half=0.20 → clippato a max 0.15
      p_win=0.4, reward_ratio=2.0 → Kelly negativo → ritorna 0.0 (non entrare)
      p_win=0.55, reward_ratio=1.5 → Kelly=0.175 → Half=0.0875

    Restituisce una percentuale del portafoglio come float (es. 0.10 = 10%).
    Kelly negativo o zero significa segnale debole: ritorna 0.0.
    """
    kelly_fraction = float(params.get("kelly_fraction", 0.5))
    max_pos        = float(params.get("max_position_pct", 0.15))

    if reward_ratio <= 0 or p_win <= 0:
        return 0.0

    # Formula Kelly intera
    kelly_f = (p_win * (reward_ratio + 1) - 1) / reward_ratio

    # Kelly negativo = expected value negativo = non entrare
    if kelly_f <= 0:
        return 0.0

    # Applica la frazione (Half-Kelly di default)
    half_f = kelly_f * kelly_fraction

    # Clippa a [0, max_position_pct]
    sized = min(half_f, max_pos)
    return round(sized, 4)


def _kelly_sizing(edge: float, rr_ratio: float) -> float:
    if rr_ratio <= 0 or edge <= 0:
        return 0.0
    kelly_f = (edge * rr_ratio - (1.0 - edge)) / rr_ratio
    kelly_f = max(0.0, kelly_f)
    return round(kelly_f * KELLY_FRACTION * 100.0, 1)


def _compute_sizing(
    ticker: str,
    agg: dict,
    risk_report: dict,
    state: dict | None = None,
) -> tuple[float, float, str, float]:
    """
    Returns (sizing_pct, conviction_score, action, consensus_ratio).

    Fase 2: sizing dinamico con Volatility Targeting + Half-Kelly.
    Il sizing finale è il minimo tra:
      - vol_adjusted_size: basato su ATR del titolo (quanto è volatile)
      - half_kelly_size:   basato su conviction e reward/risk (qualità segnale)
    Entrambi i valori sono in percentuale del portafoglio (0-100).
    """
    import logging
    log = logging.getLogger(__name__)

    net_score  = agg.get("net_score", 0.0)
    avg_conf   = agg.get("avg_confidence", 0.5)
    net_signal = agg.get("net_signal", "neutral")
    n_agents   = agg.get("n_agents", 0)
    wc         = agg.get("weighted_conviction", 0.0)

    conviction = min(1.0, abs(net_score) * avg_conf * 2.0)

    if net_signal == "bullish":
        action = "BUY"
        direction_agents = agg.get("bull_agents", [])
    elif net_signal == "bearish":
        action = "SELL"
        direction_agents = agg.get("bear_agents", [])
    else:
        return 0.0, round(conviction, 3), "HOLD", 0.0

    consensus_ratio = len(direction_agents) / n_agents if n_agents > 0 else 0.0

    if consensus_ratio < MIN_CONSENSUS_RATIO:
        return 0.0, round(conviction, 3), "HOLD", consensus_ratio

    if conviction < MIN_CONVICTION_TO_TRADE:
        return 0.0, round(conviction, 3), "HOLD", consensus_ratio

    # WC filter: if absolute WC is below threshold, skip the trade
    if abs(wc) < WC_THRESHOLD:
        return 0.0, round(conviction, 3), "HOLD", consensus_ratio

    trade_levels = risk_report.get("trade_levels", {}).get(ticker, {})
    rr_ratio = float(trade_levels.get("rr_ratio", 2.0))

    # ── Fase 2: Volatility Targeting + Half-Kelly ─────────────────────────
    params = _load_risk_params()

    # 1. Vol-adjusted size: usa ATR se abbiamo lo state, altrimenti default
    if state is not None:
        from src.data.state_reader import get_atr
        atr_pct = get_atr(state, ticker)
    else:
        atr_pct = params.get("target_daily_vol", 0.01)  # fallback conservativo

    vol_size = compute_vol_adjusted_size(atr_pct, params)  # es. 0.08 = 8%

    # 2. Half-Kelly size: usa avg_confidence come p_win
    #    reward_ratio = rr_ratio dal risk manager (già disponibile)
    p_win = avg_conf
    kelly_size = compute_half_kelly(p_win, rr_ratio, params)  # es. 0.10 = 10%

    # 3. Sizing finale = minimo tra i due (approccio conservativo)
    #    Converte da [0,1] a [0,100] per coerenza con il resto del PM
    dynamic_sizing_pct = min(vol_size, kelly_size) * 100.0

    log.info(
        "_compute_sizing %s: ATR%%=%.3f vol_size=%.1f%% kelly_size=%.1f%% → final=%.1f%%",
        ticker, atr_pct, vol_size * 100, kelly_size * 100, dynamic_sizing_pct,
    )

    # Fallback: se entrambi sono zero usa il sizing precedente (legacy)
    if dynamic_sizing_pct <= 0:
        edge = conviction
        dynamic_sizing_pct = _kelly_sizing(edge, rr_ratio)
        if dynamic_sizing_pct < MIN_SINGLE_POSITION and conviction >= 0.15:
            dynamic_sizing_pct = MIN_SINGLE_POSITION

    sizing = dynamic_sizing_pct
    # ── Fine Fase 2 ───────────────────────────────────────────────────────

    # Applica penalità di rischio (invariate dalla logica precedente)
    ticker_flags = risk_report.get("ticker_flags", {}).get(ticker, [])
    risk_penalty = 0.8 if ticker_flags else 1.0

    daily_var = risk_report.get("daily_var_95", 0.0)
    if daily_var > 0.04:
        risk_penalty *= 0.75

    macro = risk_report.get("macro_regime", {})
    vix_multiplier = float(macro.get("sizing_multiplier", 1.0))
    risk_penalty *= vix_multiplier

    sizing = sizing * risk_penalty

    if action == "SELL":
        sizing = min(sizing, 10.0)

    sizing = round(min(sizing, MAX_SINGLE_POSITION), 1)
    return sizing, round(conviction, 3), action, consensus_ratio


def _apply_correlation_penalty(recommendations: list[dict], risk_report: dict) -> list[dict]:
    corr_matrix = risk_report.get("correlation_matrix", {})
    if not corr_matrix:
        return recommendations

    buys = sorted(
        [r for r in recommendations if r["action"] == "BUY"],
        key=lambda r: r["conviction"], reverse=True,
    )
    penalized: set[str] = set()

    for i, r1 in enumerate(buys):
        for r2 in buys[i + 1:]:
            if r2["ticker"] in penalized:
                continue
            c = corr_matrix.get(r1["ticker"], {}).get(r2["ticker"], 0.0)
            if abs(c) > CORR_PENALTY_THRESHOLD:
                r2["sizing_pct"] = round(r2["sizing_pct"] * CORR_PENALTY_FACTOR, 1)
                penalized.add(r2["ticker"])

    return recommendations


def _normalise_sizing(recommendations: list[dict], risk_report: dict | None = None) -> list[dict]:
    if risk_report:
        recommendations = _apply_correlation_penalty(recommendations, risk_report)

    buys   = [r for r in recommendations if r["action"] == "BUY" and r["sizing_pct"] >= MIN_SINGLE_POSITION]
    others = [r for r in recommendations if r not in buys]

    total = sum(r["sizing_pct"] for r in buys)
    if total > TOTAL_GROSS_BUDGET and total > 0:
        scale = TOTAL_GROSS_BUDGET / total
        for r in buys:
            r["sizing_pct"] = round(r["sizing_pct"] * scale, 1)

    buys = [r for r in buys if r["sizing_pct"] >= MIN_SINGLE_POSITION]
    return buys + others


# ══════════════════════════════════════════════════════════════════════════════
# Trade-level enrichment and trade selection
# ══════════════════════════════════════════════════════════════════════════════

def _select_top_trades(pre_recs: list[dict]) -> list[dict]:
    active  = [r for r in pre_recs if r["action"] in ("BUY", "SELL")]
    hold    = [r for r in pre_recs if r["action"] == "HOLD"]
    active.sort(key=lambda r: r["conviction"] * r.get("consensus_ratio", 0.5), reverse=True)
    top     = active[:MAX_ACTIVE_TRADES]
    demoted = active[MAX_ACTIVE_TRADES:]

    for r in demoted:
        r["action"]     = "HOLD"
        r["sizing_pct"] = 0.0

    return top + demoted + hold


def _enrich_with_trade_levels(
    pre_recs: list[dict],
    risk_report: dict,
    weighted: dict[str, dict],
) -> list[dict]:
    trade_levels:    dict[str, dict] = risk_report.get("trade_levels", {})
    max_position_usd = PORTFOLIO_SIZE_USD * MAX_SINGLE_POSITION / 100.0

    for rec in pre_recs:
        ticker = rec["ticker"]
        action = rec["action"]
        agg    = weighted.get(ticker, {})
        n      = agg.get("n_agents", 0)
        wc     = agg.get("weighted_conviction", 0.0)

        rec["weighted_conviction"] = wc

        if action == "BUY":
            votes = agg.get("bull_agents", [])
            rec["consensus"] = f"{len(votes)}/{n} bullish" if n else "—"
        elif action == "SELL":
            votes = agg.get("bear_agents", [])
            rec["consensus"] = f"{len(votes)}/{n} bearish" if n else "—"
        else:
            rec["consensus"] = f"{n} agents / mixed"

        levels = trade_levels.get(ticker) if action in ("BUY", "SELL") else None

        if levels:
            entry = levels["entry_price"]
            sl    = levels["stop_loss"]
            tp    = levels["take_profit"]
            rr    = levels["rr_ratio"]

            risk_per_share = abs(entry - sl)
            risk_amount    = PORTFOLIO_SIZE_USD * RISK_PER_TRADE_PCT

            if risk_per_share > 0:
                shares   = floor(risk_amount / risk_per_share)
                size_usd = round(min(shares * entry, max_position_usd), 0)
            else:
                size_usd = 0.0

            rec["entry_price"] = entry
            rec["stop_loss"]   = sl
            rec["take_profit"] = tp
            rec["size_usd"]    = size_usd
            rec["rr_ratio"]    = rr
        else:
            rec["entry_price"] = None
            rec["stop_loss"]   = None
            rec["take_profit"] = None
            rec["size_usd"]    = None
            rec["rr_ratio"]    = None

    return pre_recs


# ══════════════════════════════════════════════════════════════════════════════
# LLM prompt
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm_prompt(
    tickers: list[str],
    weighted: dict[str, dict],
    risk_report: dict,
    pre_recs: list[dict],
    open_positions: list[dict],
) -> str:
    warnings_text  = "\n".join(risk_report.get("warnings", [])) or "None"
    positions_text = _format_positions_for_prompt(open_positions)
    open_pos_map   = {p["ticker"]: p for p in open_positions}

    ticker_block = ""
    for t in tickers:
        agg   = weighted.get(t, {})
        pre   = next((r for r in pre_recs if r["ticker"] == t), {})
        entry = pre.get("entry_price")
        sl    = pre.get("stop_loss")
        tp    = pre.get("take_profit")
        sz    = pre.get("size_usd")
        wc    = pre.get("weighted_conviction", agg.get("weighted_conviction", 0.0))

        levels_str = (
            f"entry={entry}, SL={sl}, TP={tp}, size=${sz:.0f}"
            if entry is not None else "no trade levels"
        )

        pos_info = ""
        if t in open_pos_map:
            p = open_pos_map[t]
            pos_info = (
                f" | POSIZIONE APERTA: {p['direction']} entry={p['entry_price']:.2f} "
                f"SL={p['stop_loss']:.2f} TP={p['take_profit']:.2f} size=${p['size_usd']:,.0f}"
            )

        ticker_block += (
            f"\n{t}: net_score={agg.get('net_score', 0):.3f}, "
            f"avg_confidence={agg.get('avg_confidence', 0):.2f}, "
            f"WC={wc:+.4f}, "
            f"consensus={pre.get('consensus','—')}, "
            f"action={pre.get('action','?')}, sizing={pre.get('sizing_pct', 0)}%, "
            f"conviction={pre.get('conviction', 0):.2f}, "
            f"{levels_str}{pos_info}"
        )

    return f"""You are the Portfolio Manager of Athanor Alpha, a quantitative AI hedge fund.
You have received aggregated analyst signals, preliminary sizing, and the current open positions.

SIGNAL SUMMARY (per ticker):
{ticker_block}

CURRENT OPEN POSITIONS:
{positions_text}

RISK WARNINGS:
{warnings_text}

Portfolio VaR (95%, daily): {risk_report.get('daily_var_95', 0)*100:.1f}%
Historical Max Drawdown estimate: {risk_report.get('max_drawdown_estimate', 0)*100:.1f}%

YOUR TASK:
For each ticker, write a one-sentence reasoning that explains the recommended action and sizing.
Include the Weighted Conviction (WC) value in the reasoning (e.g. "WC=+0.017").
If a position is already open for that ticker, explicitly address whether to MAINTAIN, ADD TO,
REDUCE, or CLOSE the position based on the current signal.
Then write a 2-3 sentence portfolio summary and a brief risk note.

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

Use exactly the action and sizing_pct values provided above – do not change the numbers.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Main agent node
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_manager_agent(state: AgentState) -> dict:
    """LangGraph node – Portfolio Manager."""
    data:        dict[str, Any] = state.get("data", {})
    tickers:     list[str]      = data.get("tickers", [])
    risk_report: dict           = data.get("risk_report", {})

    if not tickers:
        tickers = list(data.get("prefetched_data", {}).keys())

    progress.update_status(AGENT_ID, None, "Aggregating weighted signals")

    open_positions = _load_open_positions()
    if open_positions:
        progress.update_status(AGENT_ID, None, f"Trovate {len(open_positions)} posizioni aperte")

    # 1. Weighted signal aggregation (includes WC)
    weighted = _weighted_signals(state, tickers)

    # Log WC per ticker
    for ticker, agg in weighted.items():
        wc = agg.get("weighted_conviction", 0.0)
        progress.update_status(AGENT_ID, ticker, f"WC={wc:+.4f}")

    # 2. Pre-compute sizing
    progress.update_status(AGENT_ID, None, "Computing position sizing")
    pre_recs: list[dict] = []
    for ticker in tickers:
        agg = weighted.get(ticker, {})
        sizing, conviction, action, consensus_ratio = _compute_sizing(ticker, agg, risk_report, state=state)
        pre_recs.append({
            "ticker":              ticker,
            "action":              action,
            "sizing_pct":          sizing,
            "conviction":          conviction,
            "consensus_ratio":     consensus_ratio,
            "weighted_conviction": agg.get("weighted_conviction", 0.0),
            "reasoning":           "",
        })

    pre_recs = _normalise_sizing(pre_recs, risk_report)

    # 3. Select top trades and enrich
    progress.update_status(AGENT_ID, None, "Selecting top trades and enriching with ATR levels")
    pre_recs = _select_top_trades(pre_recs)
    pre_recs = _enrich_with_trade_levels(pre_recs, risk_report, weighted)

    # 4. LLM for reasoning
    progress.update_status(AGENT_ID, None, "Generating reasoning via LLM")
    prompt = _build_llm_prompt(tickers, weighted, risk_report, pre_recs, open_positions)

    try:
        llm_result = call_llm(
            prompt=prompt,
            pydantic_model=PortfolioOutput,
            agent_name=AGENT_ID,
            state=state,
        )

        if isinstance(llm_result, PortfolioOutput):
            llm_map = {r.ticker: r for r in llm_result.recommendations}
            for rec in pre_recs:
                llm_rec = llm_map.get(rec["ticker"])
                if llm_rec:
                    rec["reasoning"] = llm_rec.reasoning
            portfolio_summary = llm_result.portfolio_summary
            risk_notes        = llm_result.risk_notes
        else:
            portfolio_summary = "LLM output parsing failed – see raw signals."
            risk_notes        = "; ".join(risk_report.get("warnings", []))

    except Exception as e:
        portfolio_summary = f"LLM call failed: {e}"
        risk_notes        = "; ".join(risk_report.get("warnings", []))
        for rec in pre_recs:
            if not rec["reasoning"]:
                agg = weighted.get(rec["ticker"], {})
                wc  = agg.get("weighted_conviction", 0.0)
                rec["reasoning"] = (
                    f"Net score {agg.get('net_score', 0):.3f} from "
                    f"{agg.get('n_agents', 0)} agents; WC={wc:+.4f}; "
                    f"action {rec['action']} at {rec['sizing_pct']}%."
                )

    # 5. Build output
    portfolio_recommendations = {
        "recommendations":   pre_recs,
        "portfolio_summary": portfolio_summary,
        "risk_notes":        risk_notes,
        "open_positions":    open_positions,
        "total_long_pct": round(
            sum(r["sizing_pct"] for r in pre_recs if r["action"] == "BUY"), 1
        ),
        "n_long":  sum(1 for r in pre_recs if r["action"] == "BUY"),
        "n_short": sum(1 for r in pre_recs if r["action"] == "SELL"),
        "n_hold":  sum(1 for r in pre_recs if r["action"] == "HOLD"),
    }

    data["portfolio_recommendations"] = portfolio_recommendations
    data["analyst_signals"][AGENT_ID] = {
        "signal":     "neutral",
        "confidence": 1.0,
        "reasoning":  portfolio_summary,
    }

    show_agent_reasoning(portfolio_recommendations, AGENT_ID)

    summary = (
        f"Portfolio Manager | Long: {portfolio_recommendations['n_long']} tickers "
        f"({portfolio_recommendations['total_long_pct']}%) | "
        f"Short: {portfolio_recommendations['n_short']} | "
        f"Hold: {portfolio_recommendations['n_hold']} | "
        f"Posizioni aperte: {len(open_positions)}"
    )
    progress.update_status(AGENT_ID, None, "Done")

    return {
        "messages": [HumanMessage(content=summary, name=AGENT_ID)],
        "data": data,
    }
