"""
risk_manager.py – Athanor Alpha Phase 3.5 + Fase 6 (Sector Cap Enforcement)
Reads all analyst signals from state, computes:
  - Correlation matrix (10 tickers)
  - Sector concentration
  - Sector cap enforcement (max 30% per sector, from risk_params.yaml)
  - Parametric VaR
  - Max drawdown limit check
  - ATR-based trade levels (entry, stop loss, take profit) for BUY/SELL tickers
Writes risk_report to state. Does NOT block signals – only annotates.
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import yaml
from typing import Any

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.indicators.regime_detector import detect_macro_regime

AGENT_ID = "risk_manager"

# ── Sector map (hardcoded for the 10-ticker universe) ─────────────────────────
SECTOR_MAP: dict[str, str] = {
    # Large-cap US
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Technology",
    "META": "Technology",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "V": "Financials",
    "UNH": "Healthcare",
    # High-beta stocks
    "MSTR": "Crypto-Proxy",
    "COIN": "Crypto-Proxy",
    "SMCI": "Technology",
    "MELI": "Consumer Discretionary",
    # Crypto
    "BTC-USD": "Crypto",
    "ETH-USD": "Crypto",
    "SOL-USD": "Crypto",
    "BNB-USD": "Crypto",
}

# ── Load risk parameters from YAML ────────────────────────────────────────────
def _load_risk_params() -> dict:
    """Load risk_params.yaml. Returns defaults if file is missing or malformed."""
    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "risk_params.yaml"
    )
    yaml_path = os.path.normpath(yaml_path)
    try:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

# ── Risk thresholds ────────────────────────────────────────────────────────────
VAR_CONFIDENCE = 0.95          # 95% VaR
MAX_SECTOR_PCT = 0.60          # >60% of bullish signals in one sector → warning
MAX_PORTFOLIO_VAR = 0.04       # >4% daily VaR → warning
MAX_DRAWDOWN_LIMIT = 0.15      # >15% estimated drawdown → warning
MAX_SINGLE_TICKER_PCT = 0.25   # single ticker >25% sizing → warning

# ── Trade level parameters ─────────────────────────────────────────────────────
ATR_PERIOD  = 14    # ATR lookback (Wilder's smoothing)
ATR_SL_MULT = 1.0   # Stop Loss = entry ± 1×ATR
ATR_TP_MULT = 2.0   # Take Profit = entry ∓ 2×ATR  →  R/R = 1:2


# ══════════════════════════════════════════════════════════════════════════════
# Helper: extract OHLCV returns from prefetched data
# ══════════════════════════════════════════════════════════════════════════════

def _get_returns(state: AgentState, tickers: list[str]) -> pd.DataFrame:
    """
    Extract daily close prices from prefetched_data and compute log returns.
    Returns a DataFrame with tickers as columns, dates as index.
    Falls back to empty DataFrame if data is missing.
    """
    prefetched: dict[str, Any] = state.get("data", {}).get("prefetched_data", {})
    series: dict[str, pd.Series] = {}

    for ticker in tickers:
        payload = prefetched.get(ticker, {})
        hist = payload.get("ohlcv_daily")

        try:
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                close = hist["Close"].dropna()
            elif isinstance(hist, dict) and "Close" in hist:
                close = pd.Series(hist["Close"]).dropna()
            else:
                continue

            if len(close) < 20:
                continue

            log_ret = np.log(close / close.shift(1)).dropna()
            series[ticker] = log_ret

        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    df = pd.DataFrame(series)
    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# Risk computations
# ══════════════════════════════════════════════════════════════════════════════

def _correlation_matrix(returns: pd.DataFrame) -> dict:
    """Compute pairwise correlation. Returns dict-of-dicts (JSON-serialisable)."""
    if returns.empty or returns.shape[1] < 2:
        return {}
    corr = returns.corr().round(3)
    return corr.to_dict()


def _historical_var(returns: pd.DataFrame, weights: dict[str, float]) -> float:
    """
    Historical (empirical) portfolio VaR at VAR_CONFIDENCE level.
    weights: {ticker: weight_fraction}  (must sum to 1)
    Returns daily VaR as a positive fraction (e.g. 0.032 = 3.2%).
    """
    if returns.empty:
        return 0.0

    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return 0.0

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    port_returns = returns[tickers].values @ w

    percentile = (1 - VAR_CONFIDENCE) * 100
    var = -float(np.percentile(port_returns, percentile))
    return max(0.0, round(var, 4))


def _max_drawdown_estimate(returns: pd.DataFrame, weights: dict[str, float]) -> float:
    """
    Estimate max drawdown of equal-weight portfolio from historical returns.
    Returns positive fraction.
    """
    if returns.empty:
        return 0.0

    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return 0.0

    w = np.array([weights.get(t, 1.0 / len(tickers)) for t in tickers])
    w = w / w.sum()

    port_ret = returns[tickers].values @ w
    cum = np.exp(np.cumsum(port_ret))
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_dd = float(abs(drawdowns.min()))
    return round(max_dd, 4)


def _sector_concentration(bullish_tickers: list[str]) -> dict[str, float]:
    """
    Given a list of bullish tickers, compute fraction per sector.
    Returns {sector: fraction}.
    """
    if not bullish_tickers:
        return {}

    counts: dict[str, int] = {}
    for t in bullish_tickers:
        sector = SECTOR_MAP.get(t, "Unknown")
        counts[sector] = counts.get(sector, 0) + 1

    total = len(bullish_tickers)
    return {s: round(c / total, 3) for s, c in counts.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Fase 6 – Sector Cap Enforcement
# ══════════════════════════════════════════════════════════════════════════════

def _enforce_sector_cap(
    bullish_tickers: list[str],
    agg: dict[str, dict],
    sector_cap: float,
) -> tuple[list[str], list[str], list[str]]:
    """
    Enforce sector cap: no single sector can represent more than `sector_cap`
    fraction of the bullish tickers selected.

    Strategy: within each over-weight sector, drop the tickers with the
    lowest avg_confidence first, until the sector is within the cap.

    Returns:
        approved  – tickers that pass the sector cap
        excluded  – tickers dropped due to sector cap
        log_lines – human-readable log entries for each sector decision
    """
    if not bullish_tickers:
        return [], [], []

    # Group tickers by sector
    by_sector: dict[str, list[str]] = {}
    for t in bullish_tickers:
        sector = SECTOR_MAP.get(t, "Unknown")
        by_sector.setdefault(sector, []).append(t)

    total = len(bullish_tickers)
    max_allowed_per_sector = max(1, int(sector_cap * total))  # floor at 1

    approved: list[str] = []
    excluded: list[str] = []
    log_lines: list[str] = []

    for sector, tickers in by_sector.items():
        n_in_sector = len(tickers)
        sector_frac = round(n_in_sector / total, 3)

        if n_in_sector <= max_allowed_per_sector:
            # Sector is within cap – keep all
            approved.extend(tickers)
            log_lines.append(
                f"[SECTOR CAP] {sector}: {n_in_sector}/{total} tickers "
                f"({sector_frac*100:.0f}%) — OK (cap {int(sector_cap*100)}%)"
            )
        else:
            # Sector exceeds cap – sort by confidence descending, keep top N
            sorted_tickers = sorted(
                tickers,
                key=lambda t: agg.get(t, {}).get("avg_confidence", 0.0),
                reverse=True,
            )
            kept = sorted_tickers[:max_allowed_per_sector]
            dropped = sorted_tickers[max_allowed_per_sector:]

            approved.extend(kept)
            excluded.extend(dropped)

            log_lines.append(
                f"[SECTOR CAP] {sector}: {n_in_sector}/{total} tickers "
                f"({sector_frac*100:.0f}%) — CAPPED to {max_allowed_per_sector} "
                f"(cap {int(sector_cap*100)}%). "
                f"Kept: {kept}. Dropped: {dropped}."
            )

    return approved, excluded, log_lines


# ══════════════════════════════════════════════════════════════════════════════
# ATR-based trade levels
# ══════════════════════════════════════════════════════════════════════════════

def _compute_atr(ohlcv: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """
    Compute ATR using Wilder's EMA smoothing (alpha = 1/period).
    Returns the latest ATR value, or 0.0 if data is insufficient.
    """
    if ohlcv is None or len(ohlcv) < period + 1:
        return 0.0
    try:
        high  = ohlcv["High"]
        low   = ohlcv["Low"]
        close = ohlcv["Close"]
        tr = pd.concat(
            [high - low,
             (high - close.shift(1)).abs(),
             (low  - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return float(atr.iloc[-1])
    except Exception:
        return 0.0


def _compute_trade_levels(
    state: AgentState,
    tickers: list[str],
    agg: dict[str, dict],
) -> dict[str, dict]:
    """
    For each ticker with bullish or bearish net signal, compute operational
    trade levels from the ATR of the last ATR_PERIOD candles.

      BUY  → entry = last close, SL = entry − 1×ATR, TP = entry + 2×ATR
      SELL → entry = last close, SL = entry + 1×ATR, TP = entry − 2×ATR

    R/R ratio is always ATR_TP_MULT / ATR_SL_MULT = 2.0.
    Tickers with neutral signal or missing OHLCV are skipped.
    """
    prefetched: dict[str, Any] = state.get("data", {}).get("prefetched_data", {})
    result: dict[str, dict] = {}

    for ticker in tickers:
        net_signal = agg.get(ticker, {}).get("net_signal", "neutral")
        if net_signal == "neutral":
            continue

        payload = prefetched.get(ticker, {})
        ohlcv   = payload.get("ohlcv_daily")

        if isinstance(ohlcv, dict) and "Close" in ohlcv:
            try:
                ohlcv = pd.DataFrame(ohlcv)
            except Exception:
                continue

        if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
            continue

        try:
            entry = round(float(ohlcv["Close"].iloc[-1]), 2)
        except Exception:
            continue

        atr = _compute_atr(ohlcv)
        if atr <= 0.0:
            continue

        if net_signal == "bullish":
            sl        = round(entry - ATR_SL_MULT * atr, 2)
            tp        = round(entry + ATR_TP_MULT * atr, 2)
            direction = "long"
        else:  # bearish
            sl        = round(entry + ATR_SL_MULT * atr, 2)
            tp        = round(entry - ATR_TP_MULT * atr, 2)
            direction = "short"

        result[ticker] = {
            "direction":   direction,
            "entry_price": entry,
            "stop_loss":   sl,
            "take_profit": tp,
            "atr":         round(atr, 4),
            "rr_ratio":    round(ATR_TP_MULT / ATR_SL_MULT, 1),
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Signal aggregation helper
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_signals(state: AgentState, tickers: list[str]) -> dict[str, dict]:
    """
    For each ticker, collect all agent signals and compute:
      - bull_count, bear_count, neutral_count
      - average confidence
      - net_signal: "bullish" / "bearish" / "neutral"
    """
    analyst_signals: dict[str, Any] = state.get("data", {}).get("analyst_signals", {})

    result: dict[str, dict] = {}

    for ticker in tickers:
        bull, bear, neut, total_conf, n = 0, 0, 0, 0.0, 0

        for agent_id, agent_data in analyst_signals.items():
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

            signal = str(sig_data.get("signal", "neutral")).lower()
            confidence = float(sig_data.get("confidence", 0.5))

            if signal == "bullish":
                bull += 1
            elif signal == "bearish":
                bear += 1
            else:
                neut += 1

            total_conf += confidence
            n += 1

        if n == 0:
            result[ticker] = {
                "bull_count": 0, "bear_count": 0, "neutral_count": 0,
                "avg_confidence": 0.0, "net_signal": "neutral", "n_agents": 0,
            }
            continue

        avg_conf = round(total_conf / n, 3)

        if bull > bear and bull > neut:
            net = "bullish"
        elif bear > bull and bear > neut:
            net = "bearish"
        else:
            net = "neutral"

        result[ticker] = {
            "bull_count": bull,
            "bear_count": bear,
            "neutral_count": neut,
            "avg_confidence": avg_conf,
            "net_signal": net,
            "n_agents": n,
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main agent node
# ══════════════════════════════════════════════════════════════════════════════

def risk_manager_agent(state: AgentState) -> dict:
    """
    LangGraph node – Risk Manager.
    Reads analyst_signals and prefetched_data from state.
    Writes risk_report to state["data"]["risk_report"].
    """
    data: dict[str, Any] = state.get("data", {})
    tickers: list[str] = data.get("tickers", list(SECTOR_MAP.keys()))

    # ── Load configurable thresholds from risk_params.yaml ────────────────────
    risk_params = _load_risk_params()
    sector_cap: float = float(risk_params.get("sector_cap", 0.30))

    progress.update_status(AGENT_ID, None, "Aggregating analyst signals")

    # ── 1. Aggregate signals across all agents ────────────────────────────────
    agg = _aggregate_signals(state, tickers)

    bullish_tickers = [t for t, v in agg.items() if v["net_signal"] == "bullish"]
    bearish_tickers = [t for t, v in agg.items() if v["net_signal"] == "bearish"]

    # ── 2. Fase 6 – Enforce sector cap on bullish tickers ─────────────────────
    progress.update_status(AGENT_ID, None, f"Enforcing sector cap ({int(sector_cap*100)}%)")
    bullish_approved, bullish_excluded, sector_cap_log = _enforce_sector_cap(
        bullish_tickers, agg, sector_cap
    )

    # Log sector cap decisions
    for line in sector_cap_log:
        print(line)

    # ── 3. Equal-weight sizing for risk computation ────────────────────────────
    # Use approved bullish tickers only
    if bullish_approved:
        eq_weights = {t: 1.0 / len(bullish_approved) for t in bullish_approved}
    else:
        eq_weights = {t: 1.0 / len(tickers) for t in tickers}

    # ── 4. Historical returns ──────────────────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Computing correlation matrix")
    returns = _get_returns(state, tickers)

    corr_matrix = _correlation_matrix(returns)

    # ── 5. Sector concentration (on approved tickers) ─────────────────────────
    sector_fractions = _sector_concentration(bullish_approved)
    concentrated_sectors = {s: f for s, f in sector_fractions.items() if f > MAX_SECTOR_PCT}

    # ── 6. VaR (Historical / empirical) ───────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Computing historical VaR")
    try:
        daily_var = _historical_var(returns, eq_weights)
    except Exception:
        daily_var = 0.0

    # ── 7. Max drawdown estimate ───────────────────────────────────────────────
    try:
        max_dd = _max_drawdown_estimate(returns, eq_weights)
    except Exception:
        max_dd = 0.0

    # ── 8. Warnings ────────────────────────────────────────────────────────────
    warnings: list[str] = []

    # Sector cap exclusions → warning
    if bullish_excluded:
        warnings.append(
            f"Sector cap ({int(sector_cap*100)}%): {len(bullish_excluded)} ticker(s) "
            f"excluded — {bullish_excluded}"
        )

    if concentrated_sectors:
        for sector, frac in concentrated_sectors.items():
            warnings.append(
                f"Sector concentration: {int(frac*100)}% of bullish signals are {sector}"
            )

    if daily_var > MAX_PORTFOLIO_VAR:
        warnings.append(
            f"High portfolio VaR: {daily_var*100:.1f}% (threshold {MAX_PORTFOLIO_VAR*100:.0f}%)"
        )

    if max_dd > MAX_DRAWDOWN_LIMIT:
        warnings.append(
            f"Historical max drawdown: {max_dd*100:.1f}% (threshold {MAX_DRAWDOWN_LIMIT*100:.0f}%)"
        )

    # ── 9. Per-ticker risk flags ───────────────────────────────────────────────
    ticker_flags: dict[str, list[str]] = {t: [] for t in tickers}

    # Mark excluded tickers with sector cap flag
    for t in bullish_excluded:
        ticker_flags.setdefault(t, []).append(
            f"Excluded by sector cap ({int(sector_cap*100)}%): "
            f"sector {SECTOR_MAP.get(t, 'Unknown')} over limit"
        )

    # High pairwise correlation warning
    if corr_matrix:
        for t1 in tickers:
            for t2 in tickers:
                if t1 >= t2:
                    continue
                c = corr_matrix.get(t1, {}).get(t2, 0.0)
                if abs(c) > 0.85:
                    ticker_flags[t1].append(f"High correlation with {t2} ({c:.2f})")
                    ticker_flags[t2].append(f"High correlation with {t1} ({c:.2f})")

    # ── 10. ATR-based trade levels (entry / SL / TP) ──────────────────────────
    # Compute only on approved bullish + all bearish tickers
    progress.update_status(AGENT_ID, None, "Computing ATR trade levels")
    active_tickers = bullish_approved + bearish_tickers
    trade_levels = _compute_trade_levels(state, active_tickers, agg)

    # ── 11. Macro regime (VIX) ────────────────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Fetching VIX macro regime")
    try:
        macro_regime = detect_macro_regime()
    except Exception:
        macro_regime = {
            "macro_regime": "CAUTION",
            "vix": None,
            "sizing_multiplier": 0.7,
            "description": "VIX unavailable",
        }

    if macro_regime["macro_regime"] == "RISK_OFF":
        warnings.append(f"MACRO RISK_OFF: {macro_regime['description']} – evitare nuovi long.")
    elif macro_regime["macro_regime"] == "CAUTION":
        warnings.append(f"MACRO CAUTION: {macro_regime['description']}")

    # ── 12. Assemble risk report ───────────────────────────────────────────────
    risk_report = {
        "signal_summary": agg,
        "bullish_tickers": bullish_approved,          # only approved tickers
        "bullish_tickers_excluded": bullish_excluded, # Fase 6: excluded by sector cap
        "bearish_tickers": bearish_tickers,
        "sector_cap": sector_cap,                     # Fase 6: cap used
        "sector_cap_log": sector_cap_log,             # Fase 6: per-sector decision log
        "sector_concentration": sector_fractions,
        "concentrated_sectors": concentrated_sectors,
        "daily_var_95": daily_var,
        "max_drawdown_estimate": max_dd,
        "correlation_matrix": corr_matrix,
        "ticker_flags": ticker_flags,
        "trade_levels": trade_levels,
        "macro_regime": macro_regime,
        "warnings": warnings,
        "risk_ok": len(warnings) == 0,
    }

    # ── 13. Write to state ─────────────────────────────────────────────────────
    data["risk_report"] = risk_report
    data["analyst_signals"][AGENT_ID] = {
        "signal": "neutral",
        "confidence": 1.0,
        "reasoning": (
            f"Risk check complete. Sector cap {int(sector_cap*100)}%: "
            f"{len(bullish_approved)} approved, {len(bullish_excluded)} excluded. "
            f"Warnings: {len(warnings)}. "
            f"Daily VaR 95%: {daily_var*100:.1f}%. "
            f"Max DD estimate: {max_dd*100:.1f}%. "
            f"Trade levels computed for {len(trade_levels)} tickers."
        ),
    }

    # ── 14. Show reasoning ────────────────────────────────────────────────────
    show_agent_reasoning(risk_report, AGENT_ID)

    summary = (
        f"Risk Manager | {len(bullish_approved)} bullish approved "
        f"({len(bullish_excluded)} excluded by sector cap) "
        f"/ {len(bearish_tickers)} bearish "
        f"| VaR {daily_var*100:.1f}% | MaxDD {max_dd*100:.1f}% "
        f"| Trade levels: {len(trade_levels)} | Warnings: {len(warnings)}"
    )
    progress.update_status(
        AGENT_ID, None,
        "Done" if not warnings else f"⚠ {len(warnings)} warnings"
    )

    message = HumanMessage(content=summary, name=AGENT_ID)

    return {
        "messages": [message],
        "data": data,
    }
