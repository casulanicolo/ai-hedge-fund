"""
src/agents/devils_advocate.py
─────────────────────────────
Athanor Alpha – Devil's Advocate Agent (Fase 3)

Agente avversariale che trova ragioni per NON entrare nel mercato.
Si esegue dopo tutti gli analyst agent e prima del Risk Manager.

Responsabilità:
  1. VIX Check      – classifica il regime di volatilità di mercato e
                      produce un size_multiplier globale (1.0 → 0.40).
  2. Signal Coherence Veto – veta i ticker in cui meno del X% degli
                      agenti concorda sulla stessa direction.
                      La soglia è dinamica in base al livello VIX:
                        VIX < 18  (LOW):      0.45
                        VIX 18-25 (NORMAL):   0.55
                        VIX 25-35 (ELEVATED): 0.65
                        VIX > 35  (CRISIS):   0.75

Output scritto in state["data"]["devils_advocate_output"]:
  {
    "vix_level":             float,
    "vix_regime":            "LOW" | "ELEVATED" | "HIGH" | "EXTREME",
    "macro_risk":            "LOW" | "MEDIUM" | "HIGH",
    "size_multiplier":       float,   # 0.40 – 1.0
    "coherence_threshold":   float,   # threshold usata in questa run
    "vetoed_tickers":        list[str],
    "veto_reasons":          dict[str, str],   # ticker -> motivo
    "reasoning":             str,
  }
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress

logger = logging.getLogger(__name__)
AGENT_ID = "devils_advocate"

# Agenti che non producono segnali direzionali (da escludere dal conteggio)
NON_SIGNAL_AGENTS = {"risk_manager", "data_prefetch", "portfolio_manager", "prediction_log"}


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic output model
# ══════════════════════════════════════════════════════════════════════════════

class DevilsAdvocateOutput(BaseModel):
    """Output strutturato del Devil's Advocate."""

    vix_level:             float = Field(description="Livello VIX corrente")
    vix_regime:            Literal["LOW", "ELEVATED", "HIGH", "EXTREME"] = Field(
                               description="Regime di volatilità classificato"
                           )
    macro_risk:            Literal["LOW", "MEDIUM", "HIGH"] = Field(
                               description="Livello di rischio macro complessivo"
                           )
    size_multiplier:       float = Field(
                               description="Moltiplicatore globale delle size (0.40–1.0)"
                           )
    coherence_threshold:   float = Field(
                               description="Soglia di coerenza dinamica usata in questa run"
                           )
    vetoed_tickers:        list[str] = Field(
                               default_factory=list,
                               description="Ticker vetati per incoerenza segnali"
                           )
    veto_reasons:          dict[str, str] = Field(
                               default_factory=dict,
                               description="Motivo del veto per ogni ticker vetato"
                           )
    reasoning:             str = Field(
                               description="Spiegazione sintetica delle decisioni"
                           )

    @field_validator("size_multiplier")
    @classmethod
    def clamp_multiplier(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ══════════════════════════════════════════════════════════════════════════════
# Step 3.2 – VIX Check
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_vix(state: AgentState) -> float:
    """
    Tenta di leggere il VIX dal prefetch state.
    Se non disponibile, lo fetcha direttamente con yfinance.
    Ritorna 20.0 come fallback conservativo in caso di errore.
    """
    DEFAULT_VIX = 20.0

    # Prova prima dallo state (se il data_prefetch lo ha già scaricato)
    prefetched = state.get("data", {}).get("prefetched_data", {})
    vix_payload = prefetched.get("^VIX") or prefetched.get("VIX")
    if vix_payload:
        try:
            df = vix_payload.get("ohlcv_daily")
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                vix_val = float(df["close"].iloc[-1])
                logger.info("[%s] VIX letto dallo state: %.2f", AGENT_ID, vix_val)
                return vix_val
        except Exception as e:
            logger.warning("[%s] Errore lettura VIX da state: %s", AGENT_ID, e)

    # Fallback: fetch diretto con yfinance
    try:
        import yfinance as yf
        vix_ticker = yf.Ticker("^VIX")
        vix_val = float(vix_ticker.fast_info["last_price"])
        logger.info("[%s] VIX fetchato da yfinance: %.2f", AGENT_ID, vix_val)
        return vix_val
    except Exception as e:
        logger.warning("[%s] Impossibile fetchare VIX: %s — uso default %.1f",
                       AGENT_ID, e, DEFAULT_VIX)
        return DEFAULT_VIX


def _load_vix_thresholds() -> dict:
    """
    Carica le soglie VIX canoniche da config/risk_params.yaml.
    Fallback ai valori canonici se il file non è disponibile.
    """
    defaults = {"low": 18, "elevated": 25, "high": 35, "extreme": 40}
    try:
        import yaml
        with open("config/risk_params.yaml", "r") as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict) and "vix_thresholds" in loaded:
            defaults.update(loaded["vix_thresholds"])
    except Exception:
        pass
    return defaults


def _classify_vix(vix_level: float) -> tuple[str, float, str]:
    """
    Classifica il VIX in un regime e restituisce:
      (vix_regime, size_multiplier, macro_risk)

    Soglie canoniche da config/risk_params.yaml (vix_thresholds):
      VIX < low      (18)  → LOW      → multiplier=1.00 → macro_risk=LOW
      VIX low-elevated(25) → ELEVATED → multiplier=0.85 → macro_risk=LOW
      VIX elevated-high(35)→ HIGH     → multiplier=0.70 → macro_risk=MEDIUM
      VIX > high     (35)  → EXTREME  → multiplier=0.40 → macro_risk=HIGH
    """
    t = _load_vix_thresholds()
    if vix_level < t["low"]:
        return "LOW", 1.00, "LOW"
    elif vix_level < t["elevated"]:
        return "ELEVATED", 0.85, "LOW"
    elif vix_level < t["high"]:
        return "HIGH", 0.70, "MEDIUM"
    else:
        return "EXTREME", 0.40, "HIGH"


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic coherence threshold
# ══════════════════════════════════════════════════════════════════════════════

def _get_coherence_threshold(vix_level: float) -> float:
    """
    Restituisce la soglia di coerenza dinamica in base al livello VIX.
    Soglie canoniche da config/risk_params.yaml (vix_thresholds):
      VIX < low      (18)  → 0.45  – mercato calmo, accettiamo meno consensus
      VIX low-elevated(25) → 0.55  – soglia standard
      VIX elevated-high(35)→ 0.60  – volatilità elevata, richiediamo più agreement
      VIX > high     (35)  → 0.65  – crisi, standard molto alto

    Impatto atteso: riduce false rejection in regime LOW,
    aumenta protezione in regime ELEVATED/CRISIS.
    """
    t = _load_vix_thresholds()
    if vix_level < t["low"]:
        return 0.45
    elif vix_level < t["elevated"]:
        return 0.55
    elif vix_level < t["high"]:
        return 0.60
    else:
        return 0.65


# ══════════════════════════════════════════════════════════════════════════════
# Step 3.3 – Signal Coherence Veto
# ══════════════════════════════════════════════════════════════════════════════

def _compute_signal_coherence(
    state: AgentState,
    tickers: list[str],
    vix_regime: str,
    vix_level: float,
) -> tuple[list[str], dict[str, str], float]:
    """
    Per ogni ticker calcola la percentuale di agenti che concordano
    sulla direction di maggioranza.

    La soglia di coerenza è dinamica in base al vix_level:
      VIX < 18  → 0.45
      VIX 18-25 → 0.55
      VIX 25-35 → 0.65
      VIX > 35  → 0.75

    Se agreement < threshold → ticker vetato.
    In regime EXTREME → tutti i ticker vengono vetati indipendentemente.

    Ritorna (vetoed_tickers, veto_reasons, coherence_threshold_used).
    """
    analyst_signals: dict[str, Any] = state.get("data", {}).get("analyst_signals", {})
    vetoed: list[str] = []
    reasons: dict[str, str] = {}

    # Calcola soglia dinamica
    coherence_threshold = _get_coherence_threshold(vix_level)
    logger.info(
        "[%s] Coherence threshold dinamica: %.2f (VIX=%.1f)",
        AGENT_ID, coherence_threshold, vix_level,
    )

    # Regime EXTREME: veta tutto
    if vix_regime == "EXTREME":
        for ticker in tickers:
            vetoed.append(ticker)
            reasons[ticker] = "VIX regime EXTREME — nessuna operatività consentita"
        logger.warning("[%s] Regime EXTREME: tutti i ticker vetati", AGENT_ID)
        return vetoed, reasons, coherence_threshold

    for ticker in tickers:
        counts = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        total = 0

        for agent_id, agent_data in analyst_signals.items():
            if agent_id in NON_SIGNAL_AGENTS:
                continue
            if not isinstance(agent_data, dict):
                continue

            # Legge il segnale per questo ticker
            if ticker in agent_data:
                sig = agent_data[ticker]
            elif agent_data.get("ticker") == ticker:
                sig = agent_data
            else:
                continue

            if not isinstance(sig, dict):
                continue

            # Supporta sia formato nuovo (direction) che vecchio (signal)
            direction = sig.get("direction")
            if direction in ("LONG", "SHORT", "NEUTRAL"):
                counts[direction] += 1
                total += 1
            else:
                signal_raw = str(sig.get("signal", "neutral")).lower()
                if signal_raw == "bullish":
                    counts["LONG"] += 1
                    total += 1
                elif signal_raw == "bearish":
                    counts["SHORT"] += 1
                    total += 1
                elif signal_raw in ("neutral", "hold"):
                    counts["NEUTRAL"] += 1
                    total += 1

        if total == 0:
            logger.warning("[%s] %s: nessun segnale trovato — skip veto check", AGENT_ID, ticker)
            continue

        # Trova la direction di maggioranza
        majority_dir = max(counts, key=lambda d: counts[d])
        majority_count = counts[majority_dir]
        agreement_pct = majority_count / total

        logger.info(
            "[%s] %s: LONG=%d SHORT=%d NEUTRAL=%d total=%d agreement=%.0f%% (majority=%s, threshold=%.0f%%)",
            AGENT_ID, ticker,
            counts["LONG"], counts["SHORT"], counts["NEUTRAL"],
            total, agreement_pct * 100, majority_dir, coherence_threshold * 100,
        )

        if agreement_pct < coherence_threshold:
            vetoed.append(ticker)
            reason = (
                f"Solo {agreement_pct:.0%} di agreement "
                f"({counts['LONG']} LONG vs {counts['SHORT']} SHORT "
                f"vs {counts['NEUTRAL']} NEUTRAL su {total} agenti) "
                f"— sotto soglia {coherence_threshold:.0%} (VIX={vix_level:.1f})"
            )
            reasons[ticker] = reason
            logger.warning("[%s] %s vetato: %s", AGENT_ID, ticker, reason)
        else:
            logger.info("[%s] %s: agreement %.0f%% — OK (threshold %.0f%%)",
                        AGENT_ID, ticker, agreement_pct * 100, coherence_threshold * 100)

    return vetoed, reasons, coherence_threshold


# ══════════════════════════════════════════════════════════════════════════════
# Main agent node
# ══════════════════════════════════════════════════════════════════════════════

def devils_advocate_agent(state: AgentState) -> dict:
    """LangGraph node – Devil's Advocate."""
    data: dict[str, Any] = state.get("data", {})
    tickers: list[str] = data.get("tickers", [])

    if not tickers:
        tickers = list(data.get("prefetched_data", {}).keys())

    progress.update_status(AGENT_ID, None, "starting...")
    logger.info("[%s] Avvio analisi su %d ticker: %s", AGENT_ID, len(tickers), tickers)

    # ── Step 3.2: VIX Check ──────────────────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Fetching VIX...")
    vix_level = _fetch_vix(state)
    vix_regime, size_multiplier, macro_risk = _classify_vix(vix_level)

    logger.info(
        "[%s] VIX=%.2f, regime=%s, size_multiplier=%.2f, macro_risk=%s",
        AGENT_ID, vix_level, vix_regime, size_multiplier, macro_risk,
    )
    progress.update_status(
        AGENT_ID, None,
        f"VIX={vix_level:.1f} ({vix_regime}), multiplier={size_multiplier:.2f}"
    )

    # ── Step 3.3: Signal Coherence Veto ─────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Checking signal coherence...")
    vetoed_tickers, veto_reasons, coherence_threshold = _compute_signal_coherence(
        state, tickers, vix_regime, vix_level
    )

    if vetoed_tickers:
        logger.warning("[%s] Ticker vetati: %s", AGENT_ID, vetoed_tickers)
    else:
        logger.info("[%s] Nessun ticker vetato — segnali coerenti", AGENT_ID)

    # ── Build reasoning ──────────────────────────────────────────────────────
    veto_str = ""
    if vetoed_tickers:
        veto_lines = [f"  - {t}: {veto_reasons[t]}" for t in vetoed_tickers]
        veto_str = "Ticker vetati:\n" + "\n".join(veto_lines)
    else:
        veto_str = "Nessun ticker vetato — tutti i segnali mostrano sufficiente coerenza."

    reasoning = (
        f"VIX={vix_level:.1f} → regime {vix_regime} → size_multiplier={size_multiplier:.2f}. "
        f"Macro risk: {macro_risk}. "
        f"Coherence threshold dinamica: {coherence_threshold:.0%} (VIX={vix_level:.1f}). "
        f"{veto_str}"
    )

    # ── Build output ─────────────────────────────────────────────────────────
    output = DevilsAdvocateOutput(
        vix_level=round(vix_level, 2),
        vix_regime=vix_regime,
        macro_risk=macro_risk,
        size_multiplier=size_multiplier,
        coherence_threshold=coherence_threshold,
        vetoed_tickers=vetoed_tickers,
        veto_reasons=veto_reasons,
        reasoning=reasoning,
    )

    output_dict = output.model_dump()
    data["devils_advocate_output"] = output_dict

    # Scrive anche in analyst_signals per compatibilità con il grafo
    data.setdefault("analyst_signals", {})[AGENT_ID] = {
        "signal":     "neutral",
        "confidence": 1.0,
        "reasoning":  reasoning,
    }

    show_agent_reasoning(output_dict, AGENT_ID)
    progress.update_status(AGENT_ID, None, "completed")
    logger.info("[%s] completed — vetoed=%s multiplier=%.2f threshold=%.2f",
                AGENT_ID, vetoed_tickers, size_multiplier, coherence_threshold)

    summary = (
        f"Devil's Advocate | VIX={vix_level:.1f} ({vix_regime}) | "
        f"size_multiplier={size_multiplier:.2f} | "
        f"coherence_threshold={coherence_threshold:.0%} | "
        f"vetoed={vetoed_tickers if vetoed_tickers else 'none'}"
    )

    return {
        "messages": [HumanMessage(content=summary, name=AGENT_ID)],
        "data": data,
    }
