"""
risk_manager.py — Athanor Alpha Phase 3.4
Reads all analyst signals from state, computes:
  - Correlation matrix (10 tickers)
  - Sector concentration
  - Parametric VaR
  - Max drawdown limit check
Writes risk_report to state. Does NOT block signals — only annotates.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import Any

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress

AGENT_ID = "risk_manager"

# ── Sector map (hardcoded for the 10-ticker universe) ─────────────────────────
SECTOR_MAP: dict[str, str] = {
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
}

# ── Risk thresholds ────────────────────────────────────────────────────────────
VAR_CONFIDENCE = 0.95          # 95% VaR
MAX_SECTOR_PCT = 0.60          # >60% of bullish signals in one sector → warning
MAX_PORTFOLIO_VAR = 0.04       # >4% daily VaR → warning
MAX_DRAWDOWN_LIMIT = 0.15      # >15% estimated drawdown → warning
MAX_SINGLE_TICKER_PCT = 0.25   # single ticker >25% sizing → warning


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
        hist = payload.get("ohlcv_daily")  # expected: pd.DataFrame or dict

        # Handle both DataFrame and dict-of-lists from prefetch cache
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


def _parametric_var(returns: pd.DataFrame, weights: dict[str, float]) -> float:
    """
    Parametric (Gaussian) portfolio VaR at VAR_CONFIDENCE level.
    weights: {ticker: weight_fraction}  (must sum to 1)
    Returns daily VaR as a positive fraction (e.g. 0.032 = 3.2%).
    """
    if returns.empty:
        return 0.0

    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return 0.0

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()  # re-normalise

    mu = returns[tickers].mean().values
    cov = returns[tickers].cov().values

    port_mean = float(np.dot(w, mu))
    port_std = float(np.sqrt(w @ cov @ w))

    # z-score for one-tailed confidence level
    from scipy.stats import norm
    z = norm.ppf(VAR_CONFIDENCE)
    var = -(port_mean - z * port_std)
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

            # Agent data may be keyed by ticker or be a flat dict for one ticker
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
    LangGraph node — Risk Manager.
    Reads analyst_signals and prefetched_data from state.
    Writes risk_report to state["data"]["risk_report"].
    """
    data: dict[str, Any] = state.get("data", {})
    tickers: list[str] = data.get("tickers", list(SECTOR_MAP.keys()))

    progress.update_status(AGENT_ID, None, "Aggregating analyst signals")

    # ── 1. Aggregate signals across all agents ─────────────────────────────
    agg = _aggregate_signals(state, tickers)

    bullish_tickers = [t for t, v in agg.items() if v["net_signal"] == "bullish"]
    bearish_tickers = [t for t, v in agg.items() if v["net_signal"] == "bearish"]

    # ── 2. Equal-weight sizing for risk computation ────────────────────────
    # Use bullish tickers only; uniform weights for now (portfolio manager
    # will refine based on conviction later)
    if bullish_tickers:
        eq_weights = {t: 1.0 / len(bullish_tickers) for t in bullish_tickers}
    else:
        eq_weights = {t: 1.0 / len(tickers) for t in tickers}

    # ── 3. Historical returns ──────────────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Computing correlation matrix")
    returns = _get_returns(state, tickers)

    corr_matrix = _correlation_matrix(returns)

    # ── 4. Sector concentration ────────────────────────────────────────────
    sector_fractions = _sector_concentration(bullish_tickers)
    concentrated_sectors = {s: f for s, f in sector_fractions.items() if f > MAX_SECTOR_PCT}

    # ── 5. VaR ────────────────────────────────────────────────────────────
    progress.update_status(AGENT_ID, None, "Computing parametric VaR")
    try:
        daily_var = _parametric_var(returns, eq_weights)
    except Exception:
        daily_var = 0.0

    # ── 6. Max drawdown estimate ───────────────────────────────────────────
    try:
        max_dd = _max_drawdown_estimate(returns, eq_weights)
    except Exception:
        max_dd = 0.0

    # ── 7. Warnings ───────────────────────────────────────────────────────
    warnings: list[str] = []

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

    # ── 8. Per-ticker risk flags ───────────────────────────────────────────
    ticker_flags: dict[str, list[str]] = {t: [] for t in tickers}

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

    # ── 9. Assemble risk report ────────────────────────────────────────────
    risk_report = {
        "signal_summary": agg,
        "bullish_tickers": bullish_tickers,
        "bearish_tickers": bearish_tickers,
        "sector_concentration": sector_fractions,
        "concentrated_sectors": concentrated_sectors,
        "daily_var_95": daily_var,
        "max_drawdown_estimate": max_dd,
        "correlation_matrix": corr_matrix,
        "ticker_flags": ticker_flags,
        "warnings": warnings,
        "risk_ok": len(warnings) == 0,
    }

    # ── 10. Write to state ────────────────────────────────────────────────
    data["risk_report"] = risk_report
    data["analyst_signals"][AGENT_ID] = {
        "signal": "neutral",  # risk manager does not emit a trading signal
        "confidence": 1.0,
        "reasoning": f"Risk check complete. Warnings: {len(warnings)}. "
                     f"Daily VaR 95%: {daily_var*100:.1f}%. "
                     f"Max DD estimate: {max_dd*100:.1f}%.",
    }

    # ── 11. Show reasoning ────────────────────────────────────────────────
    show_agent_reasoning(risk_report, AGENT_ID)

    summary = (
        f"Risk Manager | {len(bullish_tickers)} bullish / {len(bearish_tickers)} bearish "
        f"| VaR {daily_var*100:.1f}% | MaxDD {max_dd*100:.1f}% "
        f"| Warnings: {len(warnings)}"
    )
    progress.update_status(AGENT_ID, None, "Done" if not warnings else f"⚠ {len(warnings)} warnings")

    message = HumanMessage(content=summary, name=AGENT_ID)

    return {
        "messages": [message],
        "data": data,
    }
