"""
portfolio_manager.py — Athanor Alpha Phase 3.5
Reads analyst_signals + risk_report from state.
Produces per-ticker recommendations:
  {ticker, action, sizing_pct, conviction, reasoning,
   entry_price, stop_loss, take_profit, size_usd, rr_ratio, consensus}
No execution — output only. Writes to state["data"]["portfolio_recommendations"].
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

# ── Sizing parameters ──────────────────────────────────────────────────────
MAX_SINGLE_POSITION = 20.0    # % of portfolio, hard cap per ticker
MIN_SINGLE_POSITION = 2.0     # % below this → skip (not worth it)
TOTAL_GROSS_BUDGET = 100.0    # we size to 100% gross, human decides leverage
CONVICTION_SCALE = {          # how conviction maps to base sizing
    "high":   15.0,
    "medium": 9.0,
    "low":    4.0,
}

# ── Dollar-risk parameters (read from .env, with sensible defaults) ────────
PORTFOLIO_SIZE_USD = float(os.getenv("PORTFOLIO_SIZE_USD", "10000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))   # 1% per trade
MAX_ACTIVE_TRADES  = int(os.getenv("MAX_ACTIVE_TRADES", "5"))          # max open positions

# Agents excluded from signal aggregation (they don't emit trading signals)
NON_SIGNAL_AGENTS = {"risk_manager", "data_prefetch"}


# ══════════════════════════════════════════════════════════════════════════
# Pydantic output model
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# Legge posizioni aperte da SQLite
# ══════════════════════════════════════════════════════════════════════════

def _load_open_positions() -> list[dict]:
    """
    Carica le posizioni aperte da SQLite tramite portfolio manager.
    Ritorna lista vuota se la tabella non esiste ancora.
    """
    try:
        from src.portfolio.manager import get_open_positions
        return get_open_positions()
    except Exception:
        return []


def _format_positions_for_prompt(positions: list[dict]) -> str:
    """
    Formatta le posizioni aperte in testo leggibile per il prompt LLM.
    """
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


# ══════════════════════════════════════════════════════════════════════════
# Signal aggregation with agent weights
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# Sizing logic (no LLM needed here — pure math)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# Trade-level enrichment and trade selection
# ══════════════════════════════════════════════════════════════════════════

def _select_top_trades(pre_recs: list[dict]) -> list[dict]:
    """
    Keep at most MAX_ACTIVE_TRADES active positions (BUY + SELL combined).
    Candidates are ranked by conviction descending.
    Excess positions are demoted to HOLD (sizing_pct = 0).
    """
    active = [r for r in pre_recs if r["action"] in ("BUY", "SELL")]
    hold   = [r for r in pre_recs if r["action"] == "HOLD"]

    active.sort(key=lambda r: r["conviction"], reverse=True)
    top    = active[:MAX_ACTIVE_TRADES]
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
    """
    For each BUY/SELL recommendation, attach operational trade levels from
    risk_report["trade_levels"] and compute dollar position size.
    """
    trade_levels: dict[str, dict] = risk_report.get("trade_levels", {})
    max_position_usd = PORTFOLIO_SIZE_USD * MAX_SINGLE_POSITION / 100.0

    for rec in pre_recs:
        ticker = rec["ticker"]
        action = rec["action"]
        agg    = weighted.get(ticker, {})
        n      = agg.get("n_agents", 0)

        # Consensus string
        if action == "BUY":
            votes = agg.get("bull_agents", [])
            rec["consensus"] = f"{len(votes)}/{n} bullish" if n else "—"
        elif action == "SELL":
            votes = agg.get("bear_agents", [])
            rec["consensus"] = f"{len(votes)}/{n} bearish" if n else "—"
        else:
            rec["consensus"] = f"{n} agents / mixed"

        # Trade levels (only for actionable tickers)
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


# ══════════════════════════════════════════════════════════════════════════
# LLM reasoning per ticker — include posizioni aperte nel prompt
# ══════════════════════════════════════════════════════════════════════════

def _build_llm_prompt(
    tickers: list[str],
    weighted: dict[str, dict],
    risk_report: dict,
    pre_recs: list[dict],
    open_positions: list[dict],
) -> str:
    warnings_text = "\n".join(risk_report.get("warnings", [])) or "None"
    positions_text = _format_positions_for_prompt(open_positions)

    # Mappa ticker → posizione aperta per riferimento rapido
    open_pos_map = {p["ticker"]: p for p in open_positions}

    ticker_block = ""
    for t in tickers:
        agg   = weighted.get(t, {})
        pre   = next((r for r in pre_recs if r["ticker"] == t), {})
        entry = pre.get("entry_price")
        sl    = pre.get("stop_loss")
        tp    = pre.get("take_profit")
        sz    = pre.get("size_usd")
        levels_str = (
            f"entry={entry}, SL={sl}, TP={tp}, size=${sz:.0f}"
            if entry is not None else "no trade levels"
        )

        # Aggiungi info posizione aperta se esiste
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
If a position is already open for that ticker, explicitly address whether to MAINTAIN, ADD TO,
REDUCE, or CLOSE the position based on the current signal — do not ignore existing positions.
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
Write the reasoning and summaries only. Be specific and complete — do not truncate.
"""


# ══════════════════════════════════════════════════════════════════════════
# Main agent node
# ══════════════════════════════════════════════════════════════════════════

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

    # ── 0. Carica posizioni aperte da SQLite ────────────────────────────
    open_positions = _load_open_positions()
    if open_positions:
        progress.update_status(AGENT_ID, None, f"Trovate {len(open_positions)} posizioni aperte")

    # ── 1. Weighted signal aggregation ──────────────────────────────────
    weighted = _weighted_signals(state, tickers)

    # ── 2. Pre-compute sizing (no LLM) ──────────────────────────────────
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

    # ── 3. Select top trades and enrich with operational levels ─────────
    progress.update_status(AGENT_ID, None, "Selecting top trades and enriching with ATR levels")
    pre_recs = _select_top_trades(pre_recs)
    pre_recs = _enrich_with_trade_levels(pre_recs, risk_report, weighted)

    # ── 4. LLM for reasoning text (con posizioni aperte nel contesto) ───
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
        for rec in pre_recs:
            if not rec["reasoning"]:
                agg = weighted.get(rec["ticker"], {})
                rec["reasoning"] = (
                    f"Net score {agg.get('net_score', 0):.3f} from "
                    f"{agg.get('n_agents', 0)} agents; "
                    f"action {rec['action']} at {rec['sizing_pct']}%."
                )

    # ── 5. Salva posizioni aperte nell'output per il report ─────────────
    portfolio_recommendations = {
        "recommendations": pre_recs,
        "portfolio_summary": portfolio_summary,
        "risk_notes": risk_notes,
        "open_positions": open_positions,   # ← incluse per il report
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
        f"Hold: {portfolio_recommendations['n_hold']} | "
        f"Posizioni aperte: {len(open_positions)}"
    )
    progress.update_status(AGENT_ID, None, "Done")

    message = HumanMessage(content=summary, name=AGENT_ID)

    return {
        "messages": [message],
        "data": data,
    }
