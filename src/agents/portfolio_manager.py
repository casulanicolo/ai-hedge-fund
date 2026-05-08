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
import logging
import os
import pathlib
from datetime import datetime, timezone
from math import floor
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.execution.orders import TradeOrder, order_to_dict
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress

# Fase 7 — Meta-learner (optional import; falls back gracefully if not installed)
try:
    from src.ml.meta_learner import load_current_learner as _load_meta_learner
    from src.db.init_db import get_connection as _get_db_conn
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False

AGENT_ID = "portfolio_manager"
logger = logging.getLogger(__name__)

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
ENABLE_REBALANCE   = os.getenv("ENABLE_REBALANCE", "false").lower() == "true"  # FIX: rebalancing capital-aware (default off)

NON_SIGNAL_AGENTS = {"risk_manager", "data_prefetch"}

# ── Macro regime position caps (FIX C2) ──────────────────────────────────────
# Hard ceiling on position size (%) based on macro_agent's combined regime
# (VIX + yield curve). Applied AFTER Kelly/vol sizing and VIX multiplier.
# RISK_OFF drastic reduction protects capital during macro stress.
MACRO_REGIME_POSITION_CAPS: dict[str, float] = {
    "RISK_ON":  20.0,   # normal max (same as MAX_SINGLE_POSITION)
    "CAUTION":  12.0,   # reduced — macro uncertainty
    "RISK_OFF":  5.0,   # drastic — VIX spike OR inverted yield curve
    "UNKNOWN":  12.0,   # treat unknown as caution
}

# ── Dimension map (FIX C2) ────────────────────────────────────────────────────
# Maps agent_id keys (as written into analyst_signals) to one of the 4 canonical
# signal dimensions. Agents in the same dimension are averaged before cross-
# dimension aggregation, preventing FUNDAMENTALS (6 agents) from dominating.
# Agents NOT listed here are bucketed under "OTHER" and still contribute but
# do not inflate any canonical dimension.

AGENT_DIMENSION_MAP: dict[str, str] = {
    # ── FUNDAMENTALS ──────────────────────────────────────────────────────────
    "fundamentals_analyst_agent": "FUNDAMENTALS",
    "warren_buffett_agent":       "FUNDAMENTALS",
    "ben_graham_agent":           "FUNDAMENTALS",
    "charlie_munger_agent":       "FUNDAMENTALS",
    "michael_burry_agent":        "FUNDAMENTALS",
    "bill_ackman_agent":          "FUNDAMENTALS",
    "cathie_wood_agent":          "FUNDAMENTALS",
    "phil_fisher_agent":          "FUNDAMENTALS",
    "mohnish_pabrai_agent":       "FUNDAMENTALS",
    "peter_lynch_agent":          "FUNDAMENTALS",
    "rakesh_jhunjhunwala_agent":  "FUNDAMENTALS",
    "aswath_damodaran_agent":     "FUNDAMENTALS",
    # ── TECHNICAL ─────────────────────────────────────────────────────────────
    "technical_analyst_agent":    "TECHNICAL",
    "breakout_momentum":          "TECHNICAL",
    # ── SENTIMENT ─────────────────────────────────────────────────────────────
    "sentiment_agent":            "SENTIMENT",
    "news_sentiment_agent":       "SENTIMENT",
    # ── MACRO ─────────────────────────────────────────────────────────────────
    "macro_agent":                "MACRO",
}

# ── Threshold per livelli informativi su HOLD ─────────────────────────────────
INFO_NET_SCORE_THRESHOLD = 0.10   # |net_score| minimo per mostrare livelli info su HOLD


# ── Fase 7: EWA weight + meta-learner helpers ─────────────────────────────────

def _load_ewa_weights() -> dict[tuple[str, str], float]:
    """Load all (agent_id, ticker) → weight from agent_weights table. Empty dict on failure."""
    if not _ML_AVAILABLE:
        return {}
    try:
        conn  = _get_db_conn()
        rows  = conn.execute("SELECT agent_id, ticker, weight FROM agent_weights").fetchall()
        conn.close()
        return {(r["agent_id"], r["ticker"]): float(r["weight"]) for r in rows}
    except Exception:
        return {}


def _extract_ml_context(state: AgentState, ticker: str) -> dict:
    """
    Build meta-learner context dict from live pipeline state.
    Keys: regime, vix_at_prediction, realized_vol_20d, sector, month, day_of_week.
    """
    from datetime import datetime
    from src.ml.feature_extractor import SECTOR_MAP

    macro_sig = (
        state.get("data", {})
             .get("analyst_signals", {})
             .get("macro_agent", {})
    )
    first_macro = next(iter(macro_sig.values()), {}) if isinstance(macro_sig, dict) else {}

    regime = first_macro.get("macro_regime", "CAUTION")
    vix    = first_macro.get("vix", 20.0) or 20.0

    now = datetime.now()
    return {
        "regime":            regime,
        "vix_at_prediction": float(vix),
        "realized_vol_20d":  0.0,   # not tracked live; model handles 0 gracefully
        "sector":            SECTOR_MAP.get(ticker, "UNKNOWN"),
        "month":             now.month,
        "day_of_week":       now.weekday(),
    }


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
    Dimension-balanced signal aggregation (FIX C2) with Fase 7 meta-learner weights.

    Architecture:
      1. Collect all agent signals, group by AGENT_DIMENSION_MAP dimension.
         Apply effective_weight = EWA_weight × meta_learner_weight to each agent score.
      2. Per dimension: compute average score and confidence across agents in that dim.
      3. Cross-dimension: average the dimension scores with equal weight per dimension.
         → Prevents 6 FUNDAMENTALS agents from dominating 2 TECHNICAL + 1 SENTIMENT.
      4. Consensus ratio = number of dimensions agreeing / number of active dimensions.
      5. Weighted Conviction = dimension-averaged WC (not per-agent WC).

    Returns {ticker: {net_score, avg_confidence, bull_agents, bear_agents,
                       net_signal, n_agents, n_dims, dim_scores,
                       weighted_conviction, consensus_ratio}}
    """
    analyst_signals: dict[str, Any] = state.get("data", {}).get("analyst_signals", {})

    # Fase 7: load EWA weights + meta-learner once for all tickers
    ewa_weights  = _load_ewa_weights()
    meta_learner = _load_meta_learner() if _ML_AVAILABLE else None

    result: dict[str, dict] = {}

    for ticker in tickers:

        # ── Step 1: collect per-agent data, grouped by dimension ───────────
        dim_buckets: dict[str, list[dict]] = {}
        all_bull_agents: list[str] = []
        all_bear_agents: list[str] = []
        n_agents_total = 0

        # Fase 7: extract meta-learner context once per ticker
        ml_context = _extract_ml_context(state, ticker) if meta_learner else {}

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

            # Normalise direction: supports both old (signal) and new (direction) formats
            direction  = sig_data.get("direction")
            signal_raw = sig_data.get("signal", "neutral")

            if direction:
                signal = "bullish" if direction == "LONG" else (
                    "bearish" if direction == "SHORT" else "neutral"
                )
            else:
                signal = str(signal_raw).lower()

            if signal == "skip":
                continue

            confidence = float(sig_data.get("confidence", 0.5))
            if confidence > 1.0:
                confidence /= 100.0
            confidence = max(0.0, min(1.0, confidence))

            expected_return = float(sig_data.get("expected_return", 0.0))

            # Score: ±confidence for directional signals, 0 for neutral
            if signal == "bullish":
                score = +confidence
                all_bull_agents.append(agent_id)
            elif signal == "bearish":
                score = -confidence
                all_bear_agents.append(agent_id)
            else:
                score = 0.0

            # Fase 7: effective_weight = EWA_weight × meta_learner_weight
            if score != 0.0:
                ewa_w  = ewa_weights.get((agent_id, ticker), 1.0)
                signal_raw_str = "BUY" if signal == "bullish" else ("SELL" if signal == "bearish" else "HOLD")
                ml_ctx = {**ml_context, "ewa_weight": ewa_w}
                if meta_learner:
                    ml_w = meta_learner.predict_weight(
                        agent_id, ticker, signal_raw_str, confidence, ml_ctx,
                    )
                else:
                    ml_w = 1.0
                effective_w = max(0.05, min(2.0, ewa_w * ml_w))
                score = max(-1.0, min(1.0, score * effective_w))

            dim = AGENT_DIMENSION_MAP.get(agent_id, "OTHER")
            dim_buckets.setdefault(dim, []).append({
                "agent_id":       agent_id,
                "signal":         signal,
                "score":          score,
                "confidence":     confidence,
                "expected_return": expected_return,
            })
            n_agents_total += 1

        if n_agents_total == 0:
            result[ticker] = {
                "net_score": 0.0, "avg_confidence": 0.0,
                "bull_agents": [], "bear_agents": [],
                "net_signal": "neutral", "n_agents": 0,
                "n_dims": 0, "dim_scores": {},
                "weighted_conviction": 0.0, "consensus_ratio": 0.0,
            }
            continue

        # ── Step 2: per-dimension aggregation ─────────────────────────────
        dim_scores_map: dict[str, dict] = {}

        for dim, agents_list in dim_buckets.items():
            n_dim      = len(agents_list)
            dim_score  = sum(a["score"] for a in agents_list) / n_dim
            dim_conf   = sum(a["confidence"] for a in agents_list) / n_dim

            non_neutral = [a for a in agents_list if a["signal"] != "neutral"]
            dim_wc = (
                sum(a["expected_return"] * a["confidence"] for a in non_neutral)
                / len(non_neutral)
                if non_neutral else 0.0
            )

            dim_scores_map[dim] = {
                "avg_score": dim_score,
                "avg_conf":  dim_conf,
                "dim_wc":    dim_wc,
                "n":         n_dim,
            }

        # ── Step 3: cross-dimension aggregation (equal weight per dim) ────
        # Each of the 4 canonical dimensions (FUNDAMENTALS, TECHNICAL,
        # SENTIMENT, MACRO) contributes exactly 1/n_dims to the final score.
        # With all 4 active: each dimension = 25%.
        # If a dimension is absent (agent inactive): remaining dims share 100%
        # equally. e.g. 3 active dims → each ~33%.
        n_dims     = len(dim_scores_map)
        final_score = sum(d["avg_score"] for d in dim_scores_map.values()) / n_dims
        final_conf  = sum(d["avg_conf"]  for d in dim_scores_map.values()) / n_dims

        # Dimension-weighted WC: average dim_wc across all active dimensions
        # (denominator = n_dims to correctly dilute inactive/neutral dims)
        dim_wc_final = sum(d["dim_wc"] for d in dim_scores_map.values()) / n_dims

        # ── Step 4: net signal ────────────────────────────────────────────
        _net_thr = (0.05 if state.get("metadata", {}).get("backtest") else NET_SCORE_THRESHOLD)
        if final_score > _net_thr:
            net_signal = "bullish"
            n_agreeing = sum(
                1 for d in dim_scores_map.values() if d["avg_score"] > 0.03
            )
        elif final_score < -_net_thr:
            net_signal = "bearish"
            n_agreeing = sum(
                1 for d in dim_scores_map.values() if d["avg_score"] < -0.03
            )
        else:
            net_signal = "neutral"
            n_agreeing = 0

        consensus_ratio = n_agreeing / n_dims if n_dims > 0 else 0.0

        result[ticker] = {
            "net_score":           round(final_score, 4),
            "avg_confidence":      round(final_conf, 3),
            "bull_agents":         all_bull_agents,
            "bear_agents":         all_bear_agents,
            "net_signal":          net_signal,
            "n_agents":            n_agents_total,
            "n_dims":              n_dims,
            "dim_scores":          {k: round(v["avg_score"], 4) for k, v in dim_scores_map.items()},
            "weighted_conviction": round(dim_wc_final, 6),
            "consensus_ratio":     round(consensus_ratio, 3),
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Sizing logic
# ══════════════════════════════════════════════════════════════════════════════

def _get_macro_regime(state: dict | None) -> str:
    """
    Read the macro_agent's regime from state["data"]["analyst_signals"]["macro_agent"].
    Returns the regime string for the first ticker found (regime is global).
    Falls back to "CAUTION" if macro_agent has not run or data is absent.
    """
    if state is None:
        return "CAUTION"
    try:
        macro_signals = (
            state.get("data", {})
                 .get("analyst_signals", {})
                 .get("macro_agent", {})
        )
        if not macro_signals:
            return "CAUTION"
        first = next(iter(macro_signals.values()), {})
        return first.get("macro_regime", "CAUTION")
    except Exception:
        return "CAUTION"


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
    except FileNotFoundError:
        # File assente è normale in ambienti senza config personalizzata —
        # log a DEBUG per non inquinare l'output del run.
        logger.debug(
            "[portfolio_manager] config/risk_params.yaml non trovato — uso valori di default."
        )
    except Exception as e:
        # Errori inattesi (YAML malformato, permessi, ecc.) vengono loggati
        # come WARNING in modo che il problema sia visibile nei log del VPS.
        logger.warning(
            "[portfolio_manager] Errore nel caricamento di risk_params.yaml: %s — uso valori di default.", e
        )
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
    net_score  = agg.get("net_score", 0.0)
    avg_conf   = agg.get("avg_confidence", 0.5)
    net_signal = agg.get("net_signal", "neutral")
    n_agents   = agg.get("n_agents", 0)
    wc         = agg.get("weighted_conviction", 0.0)

    _conviction_mult = 3.5 if bool((state or {}).get("metadata", {}).get("backtest")) else 2.0
    conviction = min(1.0, abs(net_score) * avg_conf * _conviction_mult)

    # In backtest mode use relaxed thresholds so seeded deterministic signals
    # can generate real trades. Live thresholds are unchanged.
    is_backtest = bool((state or {}).get("metadata", {}).get("backtest"))
    _min_consensus  = 0.0  if is_backtest else MIN_CONSENSUS_RATIO
    _min_conviction = 0.04 if is_backtest else MIN_CONVICTION_TO_TRADE
    _wc_threshold   = 0.0  if is_backtest else WC_THRESHOLD
    _net_threshold  = 0.05 if is_backtest else NET_SCORE_THRESHOLD

    if net_signal == "bullish":
        action = "BUY"
    elif net_signal == "bearish":
        action = "SELL"
    else:
        return 0.0, round(conviction, 3), "HOLD", 0.0

    # Consensus is now dimension-based (pre-computed in _weighted_signals).
    # e.g. 3 out of 4 active dimensions agree → 0.75
    consensus_ratio = agg.get("consensus_ratio", 0.0)

    if consensus_ratio < _min_consensus:
        return 0.0, round(conviction, 3), "HOLD", consensus_ratio

    if conviction < _min_conviction:
        return 0.0, round(conviction, 3), "HOLD", consensus_ratio

    # WC filter: if absolute WC is below threshold, skip the trade
    if abs(wc) < _wc_threshold:
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

    logger.info(
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

    # ── Macro regime hard cap (FIX C2) ────────────────────────────────────
    # Applied LAST — overrides all other sizing when macro_agent signals
    # RISK_OFF (deep yield-curve inversion OR VIX crisis).
    macro_regime = _get_macro_regime(state)
    macro_cap = MACRO_REGIME_POSITION_CAPS.get(macro_regime, 12.0)
    if sizing > macro_cap:
        logger.info(
            "_compute_sizing %s: macro cap %.1f%% → %.1f%% (regime=%s)",
            ticker, sizing, macro_cap, macro_regime,
        )
        sizing = round(macro_cap, 1)

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
    effective_portfolio_usd: float = PORTFOLIO_SIZE_USD,
) -> list[dict]:
    trade_levels:    dict[str, dict] = risk_report.get("trade_levels", {})
    max_position_usd = effective_portfolio_usd * MAX_SINGLE_POSITION / 100.0

    for rec in pre_recs:
        ticker = rec["ticker"]
        action = rec["action"]
        agg    = weighted.get(ticker, {})
        n      = agg.get("n_agents", 0)
        wc     = agg.get("weighted_conviction", 0.0)

        rec["weighted_conviction"] = wc

        n_dims    = agg.get("n_dims", 0)
        dim_scores = agg.get("dim_scores", {})
        dim_str   = " | ".join(
            f"{k}:{v:+.3f}" for k, v in sorted(dim_scores.items())
        ) if dim_scores else ""

        if action == "BUY":
            votes = agg.get("bull_agents", [])
            base  = f"{len(votes)}/{n} agents bullish"
        elif action == "SELL":
            votes = agg.get("bear_agents", [])
            base  = f"{len(votes)}/{n} agents bearish"
        else:
            base  = f"{n} agents / mixed"

        rec["consensus"] = f"{base} [{n_dims}D: {dim_str}]" if dim_str else base

        # ── Livelli operativi per BUY/SELL ────────────────────────────────
        levels = trade_levels.get(ticker) if action in ("BUY", "SELL") else None

        if levels:
            entry = levels["entry_price"]
            sl    = levels["stop_loss"]
            tp    = levels["take_profit"]
            rr    = levels["rr_ratio"]

            risk_per_share = abs(entry - sl)
            risk_amount    = effective_portfolio_usd * RISK_PER_TRADE_PCT

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
            # Nessun livello informativo per trade attivi
            rec["info_entry"]     = None
            rec["info_sl"]        = None
            rec["info_tp"]        = None
            rec["info_size_usd"]  = None
            rec["info_rr_ratio"]  = None
            rec["info_direction"] = None
            rec["devil_vetoed"]   = False

        else:
            # ── Livelli operativi: nessuno (HOLD o vetato) ────────────────
            rec["entry_price"] = None
            rec["stop_loss"]   = None
            rec["take_profit"] = None
            rec["size_usd"]    = None
            rec["rr_ratio"]    = None

            # ── Livelli informativi: calcolati se segnale abbastanza forte ─
            net_score     = agg.get("net_score", 0.0)
            devil_vetoed  = rec.get("reasoning", "").startswith("HOLD — vetoed")

            rec["devil_vetoed"] = devil_vetoed

            # Cerca i livelli nel risk_report anche per ticker HOLD/vetati:
            # il risk_manager calcola trade_levels solo per bullish_approved + bearish,
            # ma se net_signal è bullish/bearish il ticker potrebbe averli.
            # In alternativa li calcoliamo direttamente dai dati nel risk_report signal_summary.
            info_levels = trade_levels.get(ticker)  # potrebbe esserci se era bullish/bearish

            if info_levels and abs(net_score) >= INFO_NET_SCORE_THRESHOLD:
                i_entry = info_levels["entry_price"]
                i_sl    = info_levels["stop_loss"]
                i_tp    = info_levels["take_profit"]
                i_rr    = info_levels["rr_ratio"]
                i_dir   = info_levels["direction"]

                risk_per_share = abs(i_entry - i_sl)
                risk_amount    = effective_portfolio_usd * RISK_PER_TRADE_PCT

                if risk_per_share > 0:
                    shares       = floor(risk_amount / risk_per_share)
                    i_size_usd   = round(min(shares * i_entry, max_position_usd), 0)
                else:
                    i_size_usd = 0.0

                rec["info_entry"]     = i_entry
                rec["info_sl"]        = i_sl
                rec["info_tp"]        = i_tp
                rec["info_size_usd"]  = i_size_usd
                rec["info_rr_ratio"]  = i_rr
                rec["info_direction"] = i_dir
            elif abs(net_score) >= INFO_NET_SCORE_THRESHOLD:
                # net_score significativo ma nessun trade_level nel risk_report
                # (es. ticker escluso dal sector cap prima del calcolo ATR).
                # Proviamo a ricavare la direzione dal net_score.
                rec["info_entry"]     = None
                rec["info_sl"]        = None
                rec["info_tp"]        = None
                rec["info_size_usd"]  = None
                rec["info_rr_ratio"]  = None
                rec["info_direction"] = "long" if net_score > 0 else "short"
            else:
                # Segnale troppo debole: nessun livello informativo
                rec["info_entry"]     = None
                rec["info_sl"]        = None
                rec["info_tp"]        = None
                rec["info_size_usd"]  = None
                rec["info_rr_ratio"]  = None
                rec["info_direction"] = None

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
    da_output: dict | None = None,
) -> str:
    da_output = da_output or {}
    da_vix     = da_output.get("vix_level", 0.0)
    da_regime  = da_output.get("vix_regime", "UNKNOWN")
    da_mult    = da_output.get("size_multiplier", 1.0)
    da_vetoed  = da_output.get("vetoed_tickers", [])
    da_block = (
        f"VIX={da_vix:.1f} ({da_regime}), size_multiplier={da_mult:.2f}"
        + (f", vetoed={da_vetoed}" if da_vetoed else "")
    )
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

        dim_scores = agg.get("dim_scores", {})
        dim_str = " | ".join(
            f"{k}:{v:+.3f}" for k, v in sorted(dim_scores.items())
        ) if dim_scores else "—"

        ticker_block += (
            f"\n{t}: net_score={agg.get('net_score', 0):.3f}, "
            f"dims=[{dim_str}], "
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

DEVIL'S ADVOCATE:
{da_block}

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
# TradeOrder builder
# ══════════════════════════════════════════════════════════════════════════════

_REGIME_NORM: dict[str, str] = {
    "RISK_ON":  "RISK_ON",
    "CAUTION":  "CAUTION",
    "RISK_OFF": "RISK_OFF",
}

_ACTION_MAP: dict[str, str] = {
    "BUY":  "OPEN_LONG",
    "SELL": "OPEN_SHORT",
    "HOLD": "HOLD",
}


def _cb_opens_halted() -> bool:
    """CB gate: return True if CB1/CB3/CB5 are active (no new OPEN orders)."""
    try:
        from src.risk.circuit_breakers import is_cb_active
        return any(is_cb_active(cb) for cb in ("cb1", "cb3", "cb5"))
    except Exception:
        return False


def _build_trade_orders(
    pre_recs: list[dict],
    macro_regime: str,
    run_id: str,
    state: AgentState,
    effective_portfolio_usd: float = PORTFOLIO_SIZE_USD,
) -> list[TradeOrder]:
    """
    Convert pre_recs (internal dicts) to structured TradeOrder objects.
    Existing sizing logic (Kelly/vol) and SL/TP levels are preserved as-is;
    this function only maps them into the typed contract.
    """
    regime: str = _REGIME_NORM.get(macro_regime, "CAUTION")
    ts = datetime.now(timezone.utc)
    orders: list[TradeOrder] = []

    # CB gate (Fase 8): if CB1/CB3/CB5 active → only CLOSE/HOLD allowed
    cb_halted = _cb_opens_halted()
    if cb_halted:
        logger.warning("[pm] CB gate active — all OPEN orders downgraded to HOLD")

    for rec in pre_recs:
        ticker     = rec["ticker"]
        action_raw = rec.get("action", "HOLD")
        action     = _ACTION_MAP.get(action_raw, "HOLD")

        # CB gate: convert OPEN_LONG / OPEN_SHORT → HOLD
        if cb_halted and action in ("OPEN_LONG", "OPEN_SHORT"):
            action = "HOLD"

        entry_price  = rec.get("entry_price")
        size_usd     = rec.get("size_usd")

        # quantity from dollar amount and entry price (mirrors _enrich_with_trade_levels logic)
        quantity: int | None = None
        notional: float | None = None
        if action != "HOLD":
            notional = float(size_usd) if size_usd is not None else None
            # Fallback: trade_levels missing (e.g. ticker dropped by sector cap) but
            # portfolio_manager approved it anyway — compute notional from sizing_pct.
            if notional is None and action in ("OPEN_LONG", "OPEN_SHORT"):
                sizing_pct = rec.get("sizing_pct") or 0.0
                if sizing_pct > 0:
                    notional = round(sizing_pct / 100.0 * effective_portfolio_usd, 0)
            if entry_price and notional and entry_price > 0:
                quantity = floor(notional / entry_price) or None

        # per-agent signed scores for attribution
        signals = _collect_signals_for_ticker(state, ticker)
        agent_contributions: dict[str, float] = {}
        for sig in signals:
            aid  = sig["agent_id"]
            dirn = sig.get("direction", "NEUTRAL")
            conf = float(sig.get("confidence", 0.0))
            if dirn == "LONG":
                agent_contributions[aid] = round(+conf, 4)
            elif dirn == "SHORT":
                agent_contributions[aid] = round(-conf, 4)
            else:
                agent_contributions[aid] = 0.0

        reasoning = (rec.get("reasoning") or "")[:500]

        order = TradeOrder(
            ticker=ticker,
            action=action,
            quantity=quantity,
            notional_usd=notional,
            order_type="MARKET",
            limit_price=None,
            stop_loss=rec.get("stop_loss") if action != "HOLD" else None,
            take_profit=rec.get("take_profit") if action != "HOLD" else None,
            time_in_force="DAY",
            conviction=float(rec.get("conviction", 0.0)),
            weighted_conviction=float(rec.get("weighted_conviction", 0.0)),
            regime_at_decision=regime,
            reasoning=reasoning,
            agent_contributions=agent_contributions,
            created_at=ts,
            run_id=run_id,
        )
        orders.append(order)

    return orders


# ══════════════════════════════════════════════════════════════════════════════
# Run log persistence
# ══════════════════════════════════════════════════════════════════════════════

_RUNS_LOG_PATH = pathlib.Path("logs/runs.jsonl")


def _write_run_log(
    state: AgentState,
    weighted: dict[str, dict],
    pre_recs: list[dict],
    portfolio_summary: str,
    risk_notes: str,
    macro_regime: str,
    trade_orders: list[TradeOrder] | None = None,
) -> None:
    """
    Append one JSON line to logs/runs.jsonl for ex-post analysis.

    Schema (all fields JSON-serialisable):
      run_id, timestamp, tickers,
      macro_regime,
      per_ticker: {ticker: {net_score, dim_scores, weighted_conviction, consensus_ratio}},
      recommendations: [{ticker, action, sizing_pct, conviction, reasoning}],
      portfolio_summary, risk_notes
    """
    try:
        data = state.get("data", {})
        run_id = data.get("run_id") or state.get("metadata", {}).get("run_id", "unknown")
        tickers = data.get("tickers", [])
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        per_ticker: dict = {}
        for ticker, agg in weighted.items():
            per_ticker[ticker] = {
                "net_score":           agg.get("net_score", 0.0),
                "dim_scores":          agg.get("dim_scores", {}),
                "weighted_conviction": agg.get("weighted_conviction", 0.0),
                "consensus_ratio":     agg.get("consensus_ratio", 0.0),
                "n_dims":              agg.get("n_dims", 0),
                "n_agents":            agg.get("n_agents", 0),
            }

        recs_log = [
            {
                "ticker":     r.get("ticker"),
                "action":     r.get("action"),
                "sizing_pct": r.get("sizing_pct"),
                "conviction": r.get("conviction"),
                "wc":         r.get("weighted_conviction"),
                "reasoning":  r.get("reasoning", ""),
            }
            for r in pre_recs
        ]

        orders_log = [order_to_dict(o) for o in (trade_orders or [])]

        _LEGACY_FIELDS = (
            "ticker", "action", "sizing_pct", "conviction", "consensus_ratio",
            "weighted_conviction", "reasoning", "consensus",
            "entry_price", "stop_loss", "take_profit", "size_usd", "rr_ratio",
            "devil_vetoed",
            "info_entry", "info_sl", "info_tp", "info_size_usd",
            "info_rr_ratio", "info_direction",
        )
        portfolio_recommendations_log = [
            {k: r.get(k) for k in _LEGACY_FIELDS}
            for r in pre_recs
        ]

        entry = {
            "run_id":                    run_id,
            "timestamp":                 ts,
            "tickers":                   tickers,
            "macro_regime":              macro_regime,
            "per_ticker":                per_ticker,
            "recommendations":           recs_log,
            "portfolio_recommendations": portfolio_recommendations_log,
            "trade_orders":              orders_log,
            "portfolio_summary":         portfolio_summary,
            "risk_notes":                risk_notes,
        }

        _RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _RUNS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            "[portfolio_manager] run log appended to %s (run_id=%s)",
            _RUNS_LOG_PATH, run_id,
        )

    except Exception as exc:
        logger.warning("[portfolio_manager] _write_run_log failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# FIX: Rebalancing layer — libera cash tramite PARTIAL_EXIT se insufficiente
# ══════════════════════════════════════════════════════════════════════════════

def _rebalance_for_new_buys(
    trade_orders: "list[TradeOrder]",
    open_positions: list,
    adapter,
    run_id: str,
    weighted: dict,
) -> "list[TradeOrder]":
    """
    Se cash disponibile < somma notional nuovi BUY, propone PARTIAL_EXIT (50%)
    sulle posizioni con weighted_conviction più bassa fino a liberare abbastanza cash.
    Attivo solo se ENABLE_REBALANCE=true.
    """
    if not ENABLE_REBALANCE:
        return trade_orders

    try:
        account = adapter.get_account()
        _pv     = float(account.portfolio_value)
        available_cash = float(getattr(account, "cash", 0) or 0)
    except Exception as exc:
        logger.warning("[rebalance] get_account failed: %s", exc)
        return trade_orders

    new_buys = [o for o in trade_orders if o.action in ("OPEN_LONG", "OPEN_SHORT") and o.notional_usd]
    total_new_notional = sum(float(o.notional_usd) for o in new_buys)

    if total_new_notional == 0 or available_cash >= total_new_notional:
        return trade_orders

    cash_gap = total_new_notional - available_cash
    logger.info(
        "[rebalance] cash=%.0f < total_new_notional=%.0f → cash_gap=%.0f",
        available_cash, total_new_notional, cash_gap,
    )

    # Ordina posizioni aperte per weighted_conviction crescente (esci prima da quelle con meno conviction)
    conviction_map = {t: agg.get("weighted_conviction", 0.0) for t, agg in weighted.items()}
    sortable = []
    for pos in open_positions:
        t    = pos["ticker"]
        conv = conviction_map.get(t, 0.5)
        size = float(pos.get("size_usd", 0))
        sortable.append((conv, t, pos, size))
    sortable.sort(key=lambda x: x[0])

    freed = 0.0
    ts    = datetime.now(timezone.utc)
    extra_orders: list = []

    for conv, ticker, pos, pos_size in sortable:
        if freed >= cash_gap or pos_size <= 0:
            continue
        exit_amount = round(pos_size * 0.50, 0)
        freed += exit_amount
        logger.info(
            "[rebalance] Partial exit %s 50%% to free $%.0f for new positions",
            ticker, exit_amount,
        )
        extra_orders.append(TradeOrder(
            ticker=ticker,
            action="SCALE_OUT",
            quantity=None,
            notional_usd=exit_amount,
            order_type="MARKET",
            limit_price=None,
            stop_loss=None,
            take_profit=None,
            time_in_force="DAY",
            conviction=conv,
            weighted_conviction=conv,
            regime_at_decision="CAUTION",
            reasoning=f"[rebalance] Partial exit 50% to free ${exit_amount:,.0f} for new positions",
            agent_contributions={},
            created_at=ts,
            run_id=run_id,
        ))

    if extra_orders:
        logger.info("[rebalance] Added %d partial exits freeing ~$%.0f", len(extra_orders), freed)

    return trade_orders + extra_orders


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

    # -- Fase 3: leggi output Devil's Advocate --------------------------------
    da_output: dict      = data.get("devils_advocate_output", {})
    da_vetoed: list[str] = da_output.get("vetoed_tickers", [])
    da_multiplier: float = float(da_output.get("size_multiplier", 1.0))
    da_vix: float        = float(da_output.get("vix_level", 0.0))
    da_regime: str       = da_output.get("vix_regime", "UNKNOWN")
    da_veto_reasons: dict = da_output.get("veto_reasons", {})

    if da_vetoed:
        progress.update_status(AGENT_ID, None,
            f"Devil's Advocate: {len(da_vetoed)} ticker vetati, multiplier={da_multiplier:.2f}")
    else:
        progress.update_status(AGENT_ID, None,
            f"Devil's Advocate: VIX={da_vix:.1f} ({da_regime}), multiplier={da_multiplier:.2f}")
    # -- Fine lettura Devil's Advocate ----------------------------------------

    # 1. Weighted signal aggregation (includes WC)
    weighted = _weighted_signals(state, tickers)

    # Log macro regime detected by macro_agent
    macro_regime_detected = _get_macro_regime(state)
    macro_cap = MACRO_REGIME_POSITION_CAPS.get(macro_regime_detected, 12.0)
    logger.info(
        "[portfolio_manager] macro_regime=%s → position_cap=%.0f%%",
        macro_regime_detected, macro_cap,
    )
    progress.update_status(
        AGENT_ID, None,
        f"macro_regime={macro_regime_detected} cap={macro_cap:.0f}%",
    )

    # Log dimension scores + WC per ticker
    for ticker, agg in weighted.items():
        wc         = agg.get("weighted_conviction", 0.0)
        n_dims     = agg.get("n_dims", 0)
        dim_scores = agg.get("dim_scores", {})
        dim_log    = " | ".join(
            f"{k}:{v:+.3f}" for k, v in sorted(dim_scores.items())
        )
        logger.info(
            "[portfolio_manager] %s: net_score=%+.4f n_dims=%d WC=%+.4f | %s",
            ticker,
            agg.get("net_score", 0.0),
            n_dims,
            wc,
            dim_log if dim_log else "no dims",
        )
        progress.update_status(
            AGENT_ID, ticker,
            f"net={agg.get('net_score',0):+.3f} WC={wc:+.4f} dims=[{dim_log}]",
        )

    # 2a. Fetch live buying_power as effective portfolio size
    try:
        from src.execution.alpaca_adapter import AlpacaBrokerAdapter
        _acct = AlpacaBrokerAdapter().get_account()
        effective_portfolio_usd = float(_acct.buying_power)
    except Exception:
        effective_portfolio_usd = PORTFOLIO_SIZE_USD
    logger.info("[portfolio_manager] effective_portfolio_usd=%.0f", effective_portfolio_usd)

    # 2. Pre-compute sizing
    progress.update_status(AGENT_ID, None, "Computing position sizing")
    pre_recs: list[dict] = []
    for ticker in tickers:
        agg = weighted.get(ticker, {})

        # -- Fase 3: applica veto ---------------------------------------------
        if ticker in da_vetoed:
            veto_reason = da_veto_reasons.get(ticker, "vetoed by Devil's Advocate")
            logger.warning(
                "[portfolio_manager] %s: action overridden to HOLD (%s)", ticker, veto_reason
            )
            pre_recs.append({
                "ticker":              ticker,
                "action":              "HOLD",
                "sizing_pct":          0.0,
                "conviction":          0.0,
                "consensus_ratio":     0.0,
                "weighted_conviction": agg.get("weighted_conviction", 0.0),
                "reasoning":           f"HOLD — vetoed by Devil's Advocate: {veto_reason}",
            })
            continue
        # -- Fine veto --------------------------------------------------------

        sizing, conviction, action, consensus_ratio = _compute_sizing(ticker, agg, risk_report, state=state)

        # -- Fase 3: applica size_multiplier ----------------------------------
        if action in ("BUY", "SELL") and da_multiplier < 1.0:
            sizing_before = sizing
            sizing = round(sizing * da_multiplier, 1)
            logger.info(
                "[portfolio_manager] %s: sizing %.1f%% -> %.1f%% (DA multiplier=%.2f, VIX=%.1f %s)",
                ticker, sizing_before, sizing, da_multiplier, da_vix, da_regime,
            )
        # -- Fine multiplier --------------------------------------------------

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
    pre_recs = _enrich_with_trade_levels(pre_recs, risk_report, weighted, effective_portfolio_usd)

    # 4. LLM for reasoning
    progress.update_status(AGENT_ID, None, "Generating reasoning via LLM")
    prompt = _build_llm_prompt(tickers, weighted, risk_report, pre_recs, open_positions, da_output=da_output)

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
            # Fill in auto-reasoning for records the LLM stub left blank
            for rec in pre_recs:
                if not rec.get("reasoning"):
                    agg = weighted.get(rec["ticker"], {})
                    wc  = agg.get("weighted_conviction", 0.0)
                    rec["reasoning"] = (
                        f"net_score={agg.get('net_score',0):+.3f} "
                        f"avg_conf={agg.get('avg_confidence',0):.2f} "
                        f"WC={wc:+.4f} → {rec['action']} {rec['sizing_pct']}%"
                    )
            # In backtest mode llm_result is a default stub with no real summary
            is_backtest_run = bool(state.get("metadata", {}).get("backtest"))
            if is_backtest_run or llm_result.portfolio_summary.startswith("Error"):
                buys  = sum(1 for r in pre_recs if r["action"] == "BUY")
                sells = sum(1 for r in pre_recs if r["action"] == "SELL")
                holds = sum(1 for r in pre_recs if r["action"] == "HOLD")
                portfolio_summary = (
                    f"Backtest run: {buys} BUY / {sells} SELL / {holds} HOLD "
                    f"from {len(pre_recs)} tickers (quant-only, no LLM)."
                )
            else:
                portfolio_summary = llm_result.portfolio_summary
            risk_notes = llm_result.risk_notes or "; ".join(risk_report.get("warnings", []))
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

    # 5. Salva decisioni in SQLite (portfolio_decisions)
    try:
        from src.db.init_db import get_connection, insert_portfolio_decision
        from datetime import datetime, timezone
        _ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _rid  = data.get("run_id") or state.get("metadata", {}).get("run_id", "unknown")
        _conn = get_connection()
        for rec in pre_recs:
            insert_portfolio_decision(_conn, _rid, _ts, rec, weighted)
        _conn.close()
        progress.update_status(AGENT_ID, None, f"Saved {len(pre_recs)} decisions to DB")
    except Exception as _e:
        logger.warning("[portfolio_manager] Failed to save portfolio_decisions: %s", _e)

    # 5b. Build structured TradeOrders (Fase 4 – Alpaca prerequisite)
    _run_id = data.get("run_id") or state.get("metadata", {}).get("run_id", "unknown")
    trade_orders = _build_trade_orders(pre_recs, macro_regime_detected, _run_id, state, effective_portfolio_usd)

    # FIX: rebalancing layer — aggiunge PARTIAL_EXIT se cash insufficiente per nuovi BUY
    if ENABLE_REBALANCE:
        try:
            from src.execution.alpaca_adapter import AlpacaBrokerAdapter
            _rb_adapter = AlpacaBrokerAdapter()
            trade_orders = _rebalance_for_new_buys(
                trade_orders, open_positions, _rb_adapter, _run_id, weighted,
            )
        except Exception as _rb_exc:
            logger.warning("[rebalance] _rebalance_for_new_buys failed: %s", _rb_exc)

    data["trade_orders"] = trade_orders

    progress.update_status(AGENT_ID, None, f"Built {len(trade_orders)} TradeOrders")

    # 6. Persist run log
    _write_run_log(
        state=state,
        weighted=weighted,
        pre_recs=pre_recs,
        portfolio_summary=portfolio_summary,
        risk_notes=risk_notes,
        macro_regime=macro_regime_detected,
        trade_orders=trade_orders,
    )

    # 7. Build output
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
