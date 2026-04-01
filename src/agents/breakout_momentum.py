"""
src/agents/breakout_momentum.py
Breakout & Volume Surge Agent — Athanor Alpha

Agente dedicato al rilevamento di breakout tecnici confermati da volume anomalo.
Non si basa su fondamentali o valutazioni — guarda esclusivamente struttura
del prezzo e volume per identificare momentum nascente o breakdown in corso.

Metriche calcolate per ogni ticker:
  1. Volume Surge Ratio      — volume corrente vs media 20gg (>2x = anomalo)
  2. ATR Expansion           — ATR recente vs ATR 20gg fa (espansione = momentum)
  3. 52-Week High Proximity  — distanza % dal massimo annuale
  4. Resistance Breakout     — prezzo sopra il massimo degli ultimi 20gg (con volume)
  5. Weekly Trend Confirm    — trend weekly (EMA8 vs EMA21 su weekly candles)
  6. Momentum Score          — ROC 5gg e 10gg

Segnale finale via LLM con contesto quantitativo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.data.state_reader import get_ohlcv_daily, get_ohlcv_weekly, get_ohlcv_4h


AGENT_ID = "breakout_momentum"

# ── Parametri ─────────────────────────────────────────────────────────────────
VOLUME_SURGE_THRESHOLD = 2.0    # Volume > 2x media 20gg = anomalo
ATR_EXPANSION_THRESHOLD = 1.3   # ATR recente > 1.3x ATR 20gg fa = espansione
BREAKOUT_LOOKBACK = 20          # gg per calcolare la resistenza
HIGH_PROXIMITY_PCT = 0.05       # Entro il 5% dal 52w high = zona breakout


# ── Output strutturato ─────────────────────────────────────────────────────────

class BreakoutSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = Field(
        description="Trading signal based on breakout/momentum analysis"
    )
    confidence: float = Field(
        description="Confidence level between 0 and 100"
    )
    reasoning: str = Field(
        description="Concise explanation of the signal, max 3 sentences"
    )


# ── Metriche ───────────────────────────────────────────────────────────────────

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level columns if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _volume_surge_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Rapporto tra volume dell'ultima barra e media volume degli ultimi N periodi.
    Valori > 2 indicano volume anomalo (breakout o panic sell).
    """
    if df is None or len(df) < lookback + 1:
        return 1.0
    try:
        vol = df["Volume"].dropna()
        if len(vol) < lookback + 1:
            return 1.0
        avg_vol = vol.iloc[-(lookback + 1):-1].mean()
        last_vol = vol.iloc[-1]
        if avg_vol <= 0:
            return 1.0
        return round(float(last_vol / avg_vol), 2)
    except Exception:
        return 1.0


def _atr_expansion(df: pd.DataFrame, period: int = 14, lookback: int = 20) -> float:
    """
    Rapporto tra ATR corrente e ATR di 20 giorni fa.
    Valori > 1.3 indicano espansione della volatilità (momentum nascente).
    """
    if df is None or len(df) < period + lookback + 5:
        return 1.0
    try:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr = pd.concat(
            [high - low,
             (high - close.shift(1)).abs(),
             (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        current_atr = float(atr.iloc[-1])
        past_atr = float(atr.iloc[-(lookback + 1)])
        if past_atr <= 0:
            return 1.0
        return round(current_atr / past_atr, 2)
    except Exception:
        return 1.0


def _high_proximity(df: pd.DataFrame) -> float:
    """
    Distanza percentuale del prezzo corrente dal massimo degli ultimi 252 giorni.
    0.0 = al massimo, 0.10 = 10% sotto il massimo.
    """
    if df is None or len(df) < 2:
        return 1.0
    try:
        close = df["Close"].dropna()
        current = float(close.iloc[-1])
        high_52w = float(close.rolling(min(252, len(close))).max().iloc[-1])
        if high_52w <= 0:
            return 1.0
        return round((high_52w - current) / high_52w, 4)
    except Exception:
        return 1.0


def _resistance_breakout(df: pd.DataFrame, lookback: int = BREAKOUT_LOOKBACK) -> dict:
    """
    Rileva se il prezzo ha rotto sopra/sotto il massimo/minimo degli ultimi N gg
    con volume anomalo (conferma del breakout).

    Returns:
        {
          "breakout_up":   bool  — prezzo > max degli ultimi N gg
          "breakdown":     bool  — prezzo < min degli ultimi N gg
          "distance_pct":  float — distanza % dal livello di resistenza
        }
    """
    if df is None or len(df) < lookback + 2:
        return {"breakout_up": False, "breakdown": False, "distance_pct": 0.0}
    try:
        close = df["Close"].dropna()
        current = float(close.iloc[-1])
        prev_high = float(close.iloc[-(lookback + 1):-1].max())
        prev_low = float(close.iloc[-(lookback + 1):-1].min())

        breakout_up = current > prev_high
        breakdown = current < prev_low
        distance_pct = round((current - prev_high) / prev_high, 4) if prev_high > 0 else 0.0

        return {
            "breakout_up": breakout_up,
            "breakdown": breakdown,
            "distance_pct": distance_pct,
        }
    except Exception:
        return {"breakout_up": False, "breakdown": False, "distance_pct": 0.0}


def _weekly_trend(df_weekly: pd.DataFrame | None) -> str:
    """
    Trend settimanale basato su EMA8 vs EMA21 sui candles weekly.
    Returns: "UP" | "DOWN" | "FLAT"
    """
    if df_weekly is None or len(df_weekly) < 22:
        return "FLAT"
    try:
        close = df_weekly["Close"].dropna()
        ema8 = close.ewm(span=8, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        if float(ema8.iloc[-1]) > float(ema21.iloc[-1]):
            return "UP"
        elif float(ema8.iloc[-1]) < float(ema21.iloc[-1]):
            return "DOWN"
        return "FLAT"
    except Exception:
        return "FLAT"


def _momentum_roc(df: pd.DataFrame) -> dict:
    """
    Rate of Change (ROC) a 5 e 10 giorni.
    ROC% = (close_now - close_N) / close_N * 100
    """
    if df is None or len(df) < 11:
        return {"roc_5d": 0.0, "roc_10d": 0.0}
    try:
        close = df["Close"].dropna()
        roc5 = round((float(close.iloc[-1]) - float(close.iloc[-6])) / float(close.iloc[-6]) * 100, 2)
        roc10 = round((float(close.iloc[-1]) - float(close.iloc[-11])) / float(close.iloc[-11]) * 100, 2)
        return {"roc_5d": roc5, "roc_10d": roc10}
    except Exception:
        return {"roc_5d": 0.0, "roc_10d": 0.0}


def _compute_breakout_score(metrics: dict) -> float:
    """
    Score composito 0-1 che riflette la forza del segnale breakout.
    Usato come pre-filtro (se score < 0.3 → neutral diretto senza LLM).
    """
    score = 0.0

    # Volume anomalo (+0.3)
    if metrics["volume_surge_ratio"] >= VOLUME_SURGE_THRESHOLD:
        score += 0.3
    elif metrics["volume_surge_ratio"] >= 1.5:
        score += 0.15

    # ATR expansion (+0.2)
    if metrics["atr_expansion"] >= ATR_EXPANSION_THRESHOLD:
        score += 0.2
    elif metrics["atr_expansion"] >= 1.1:
        score += 0.1

    # Resistenza rotta con volume (+0.25)
    if metrics["breakout_up"] and metrics["volume_surge_ratio"] >= 1.5:
        score += 0.25
    elif metrics["breakdown"] and metrics["volume_surge_ratio"] >= 1.5:
        score += 0.25

    # 52w high proximity (+0.15)
    if metrics["high_proximity"] <= HIGH_PROXIMITY_PCT:
        score += 0.15

    # Momentum ROC (+0.1)
    if abs(metrics["roc_5d"]) >= 3.0:
        score += 0.1

    return round(min(score, 1.0), 3)


# ── Main agent node ────────────────────────────────────────────────────────────

def breakout_momentum_agent(state: AgentState) -> dict:
    """
    LangGraph node — Breakout & Volume Surge Agent.
    Analizza rotture di resistenza con volume anomalo e ATR expansion.
    Produce segnale bullish/bearish/neutral per ogni ticker.
    """
    data = state["data"]
    tickers: list[str] = data.get("tickers", [])

    signals: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(AGENT_ID, ticker, "Computing breakout metrics")

        daily_df = get_ohlcv_daily(state, ticker)
        weekly_df = get_ohlcv_weekly(state, ticker)

        if daily_df is not None:
            daily_df = _normalize_df(daily_df)
        if weekly_df is not None:
            weekly_df = _normalize_df(weekly_df)

        if daily_df is None or len(daily_df) < 25:
            progress.update_status(AGENT_ID, ticker, "Insufficient data")
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": "Dati OHLCV insufficienti per l'analisi breakout.",
            }
            continue

        # ── Calcolo metriche ──────────────────────────────────────────────
        vol_surge = _volume_surge_ratio(daily_df)
        atr_exp = _atr_expansion(daily_df)
        proximity = _high_proximity(daily_df)
        breakout = _resistance_breakout(daily_df)
        weekly_trend = _weekly_trend(weekly_df)
        roc = _momentum_roc(daily_df)

        metrics = {
            "volume_surge_ratio": vol_surge,
            "atr_expansion": atr_exp,
            "high_proximity": proximity,
            "breakout_up": breakout["breakout_up"],
            "breakdown": breakout["breakdown"],
            "distance_pct": breakout["distance_pct"],
            "weekly_trend": weekly_trend,
            "roc_5d": roc["roc_5d"],
            "roc_10d": roc["roc_10d"],
        }

        composite_score = _compute_breakout_score(metrics)

        # ── Segnale diretto se score molto basso (senza sprecare LLM call) ─
        if composite_score < 0.20:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 10,
                "reasoning": f"Nessun breakout rilevato. Volume surge: {vol_surge:.1f}x, ATR exp: {atr_exp:.2f}x.",
            }
            progress.update_status(AGENT_ID, ticker, "No breakout — neutral (fast path)")
            continue

        # ── LLM reasoning ────────────────────────────────────────────────
        progress.update_status(AGENT_ID, ticker, "Calling LLM for breakout reasoning")

        try:
            close_price = float(daily_df["Close"].dropna().iloc[-1])
        except Exception:
            close_price = 0.0

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a quantitative momentum and breakout trader.
You analyze technical breakout signals and volume patterns to identify high-probability momentum entries.
Your edge is detecting when institutional money moves into a ticker via price-volume confirmation.
You ONLY analyze price action and volume — no fundamentals, no valuations.

Respond ONLY with valid JSON matching the schema provided."""
            ),
            (
                "human",
                """Analyze the following breakout metrics for {ticker} (current price: {price:.2f}):

BREAKOUT METRICS:
- Volume Surge Ratio: {volume_surge_ratio:.2f}x  (>2.0 = strong, >1.5 = moderate)
- ATR Expansion:      {atr_expansion:.2f}x  (>1.3 = volatility expanding = momentum building)
- 52-Week High Dist:  {high_proximity:.1%}  (0% = at high, 5% = near breakout zone)
- Resistance Breakout UP:   {breakout_up}
- Resistance Breakdown:     {breakdown}
- Distance from Resistance: {distance_pct:+.1%}
- Weekly Trend:       {weekly_trend}
- ROC 5d:             {roc_5d:+.1f}%
- ROC 10d:            {roc_10d:+.1f}%
- Composite Score:    {composite_score:.2f}/1.0

INTERPRETATION GUIDE:
- bullish: breakout_up=True + volume_surge_ratio > 1.5 + weekly_trend=UP → momentum long setup
- bearish: breakdown=True + volume_surge_ratio > 1.5 + weekly_trend=DOWN → momentum short setup
- neutral: no confirmed breakout or conflicting signals

Return JSON with signal, confidence (0-100), and reasoning."""
            ),
        ])

        try:
            result = call_llm(
                prompt=prompt_template.invoke({
                    "ticker": ticker,
                    "price": close_price,
                    "volume_surge_ratio": metrics["volume_surge_ratio"],
                    "atr_expansion": metrics["atr_expansion"],
                    "high_proximity": metrics["high_proximity"],
                    "breakout_up": metrics["breakout_up"],
                    "breakdown": metrics["breakdown"],
                    "distance_pct": metrics["distance_pct"],
                    "weekly_trend": metrics["weekly_trend"],
                    "roc_5d": metrics["roc_5d"],
                    "roc_10d": metrics["roc_10d"],
                    "composite_score": composite_score,
                }).to_string(),
                pydantic_model=BreakoutSignal,
                agent_name=AGENT_ID,
                state=state,
            )

            if isinstance(result, BreakoutSignal):
                signals[ticker] = {
                    "signal": result.signal,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "metrics": metrics,
                    "composite_score": composite_score,
                }
            else:
                raise ValueError("LLM did not return BreakoutSignal")

        except Exception as e:
            # Fallback deterministico se LLM fallisce
            if composite_score >= 0.5 and (metrics["breakout_up"] or metrics["roc_5d"] > 3):
                signal = "bullish"
                confidence = min(90, int(composite_score * 100))
            elif composite_score >= 0.5 and (metrics["breakdown"] or metrics["roc_5d"] < -3):
                signal = "bearish"
                confidence = min(90, int(composite_score * 100))
            else:
                signal = "neutral"
                confidence = 20

            signals[ticker] = {
                "signal": signal,
                "confidence": confidence,
                "reasoning": (
                    f"Breakout score {composite_score:.2f}. Vol surge {metrics['volume_surge_ratio']:.1f}x. "
                    f"ATR exp {metrics['atr_expansion']:.2f}x. Weekly trend {metrics['weekly_trend']}. "
                    f"(LLM fallback: {e})"
                ),
                "metrics": metrics,
                "composite_score": composite_score,
            }

        progress.update_status(AGENT_ID, ticker, f"Done — {signals[ticker]['signal']}")

    # ── Aggiorna state ────────────────────────────────────────────────────────
    data["analyst_signals"][AGENT_ID] = signals

    show_agent_reasoning(signals, AGENT_ID)

    n_bull = sum(1 for s in signals.values() if s.get("signal") == "bullish")
    n_bear = sum(1 for s in signals.values() if s.get("signal") == "bearish")
    summary = (
        f"Breakout Agent | {n_bull} bullish breakout / {n_bear} breakdown / "
        f"{len(signals) - n_bull - n_bear} neutral"
    )
    progress.update_status(AGENT_ID, None, "Done")

    message = HumanMessage(content=summary, name=AGENT_ID)

    return {
        "messages": [message],
        "data": data,
    }
