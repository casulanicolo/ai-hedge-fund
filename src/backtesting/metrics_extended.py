"""
src/backtesting/metrics_extended.py
───────────────────────────────────
Extended performance metrics for forward backtest reporting.

Extends src/backtesting/metrics.py (Sharpe/Sortino/MaxDD) with:

  Risk-adjusted
    - calmar_ratio        : ann_return / |max_drawdown|
    - mar_ratio           : same as calmar (alias commonly used)
    - omega_ratio         : Pr[gain] / Pr[loss], threshold = 0
    - tail_ratio          : 95th-percentile gain / |5th-percentile loss|

  Path-shape
    - ulcer_index         : RMS drawdown
    - time_under_water    : longest consecutive days below previous peak
    - sequential_win_rate : longest streak of positive days

  Trade-level
    - profit_factor       : sum(wins) / sum(|losses|)
    - r_multiple_avg      : avg trade P/L / avg risked capital per trade
    - r_multiple_dist     : list of R per closed trade

All inputs accept the standard portfolio_values list-of-dicts shape used by
the legacy engine: [{"Date": ..., "Portfolio Value": float}, ...].
Closed-trade metrics also accept a trade list:
  [{"entry": float, "exit": float, "qty": float, "side": "LONG"|"SHORT", "stop_loss": float}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


ANNUAL_TRADING_DAYS = 252


# ── helpers ───────────────────────────────────────────────────────────────
def _returns(values: Sequence[dict]) -> pd.Series:
    if not values:
        return pd.Series(dtype=float)
    df = pd.DataFrame(values).set_index("Date")
    if "Portfolio Value" not in df:
        return pd.Series(dtype=float)
    return df["Portfolio Value"].pct_change().dropna()


def _equity(values: Sequence[dict]) -> pd.Series:
    if not values:
        return pd.Series(dtype=float)
    return pd.DataFrame(values).set_index("Date")["Portfolio Value"]


def _max_drawdown_pct(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max
    return float(dd.min() * 100.0) if len(dd) else 0.0


def _ann_return_pct(values: Sequence[dict]) -> float:
    eq = _equity(values)
    if eq.empty or len(eq) < 2:
        return 0.0
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
    n = len(eq) - 1
    if n <= 0:
        return 0.0
    return ((1 + total / 100.0) ** (ANNUAL_TRADING_DAYS / n) - 1.0) * 100.0


# ── extended ratios ───────────────────────────────────────────────────────
def calmar_ratio(values: Sequence[dict]) -> Optional[float]:
    eq = _equity(values)
    if eq.empty:
        return None
    mdd = _max_drawdown_pct(eq)
    if mdd >= 0:  # no drawdown observed
        return None
    return _ann_return_pct(values) / abs(mdd)


def mar_ratio(values: Sequence[dict]) -> Optional[float]:
    return calmar_ratio(values)  # alias


def omega_ratio(values: Sequence[dict], threshold: float = 0.0) -> Optional[float]:
    r = _returns(values)
    if r.empty:
        return None
    gains  = r[r > threshold].sum()
    losses = -r[r < threshold].sum()
    if losses < 1e-12:
        return None
    return float(gains / losses)


def tail_ratio(values: Sequence[dict]) -> Optional[float]:
    r = _returns(values)
    if r.empty or len(r) < 20:
        return None
    p95 = float(np.percentile(r,  95))
    p05 = float(np.percentile(r,   5))
    if p05 == 0:
        return None
    return p95 / abs(p05)


# ── path-shape metrics ────────────────────────────────────────────────────
def ulcer_index(values: Sequence[dict]) -> Optional[float]:
    eq = _equity(values)
    if eq.empty:
        return None
    rolling_max = eq.cummax()
    dd_pct = ((eq - rolling_max) / rolling_max) * 100.0
    return float(np.sqrt((dd_pct ** 2).mean()))


def time_under_water_days(values: Sequence[dict]) -> int:
    """Longest run of consecutive days where equity < previous peak."""
    eq = _equity(values)
    if eq.empty:
        return 0
    rolling_max = eq.cummax()
    underwater = (eq < rolling_max).astype(int)
    longest = run = 0
    for v in underwater:
        if v:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return int(longest)


def sequential_win_rate(values: Sequence[dict]) -> dict[str, Any]:
    r = _returns(values)
    if r.empty:
        return {"longest_win_streak": 0, "longest_loss_streak": 0, "pct_positive_days": 0.0}
    longest_win = longest_loss = win = loss = 0
    for x in r:
        if x > 0:
            win += 1; loss = 0; longest_win = max(longest_win, win)
        elif x < 0:
            loss += 1; win = 0; longest_loss = max(longest_loss, loss)
        else:
            win = loss = 0
    return {
        "longest_win_streak":   int(longest_win),
        "longest_loss_streak":  int(longest_loss),
        "pct_positive_days":    float((r > 0).mean() * 100.0),
    }


# ── trade-level metrics ───────────────────────────────────────────────────
def profit_factor(trades: Sequence[dict]) -> Optional[float]:
    if not trades:
        return None
    pls = [_trade_pl(t) for t in trades]
    wins   = sum(p for p in pls if p > 0)
    losses = -sum(p for p in pls if p < 0)
    if losses < 1e-12:
        return None
    return float(wins / losses)


def r_multiples(trades: Sequence[dict]) -> dict[str, Any]:
    """Compute per-trade R = P/L ÷ initial-risk where risk = |entry - stop_loss| × qty."""
    rs: list[float] = []
    for t in trades:
        risk = _trade_risk(t)
        if risk <= 0:
            continue
        rs.append(_trade_pl(t) / risk)
    if not rs:
        return {"r_multiple_avg": None, "r_multiple_distribution": []}
    return {
        "r_multiple_avg":          float(np.mean(rs)),
        "r_multiple_distribution": [round(x, 3) for x in rs],
    }


def _trade_pl(t: dict) -> float:
    side = (t.get("side") or "LONG").upper()
    entry = float(t.get("entry", 0.0))
    exit_ = float(t.get("exit",  0.0))
    qty   = float(t.get("qty",   0.0))
    sign  = 1.0 if side == "LONG" else -1.0
    return sign * (exit_ - entry) * qty


def _trade_risk(t: dict) -> float:
    entry = float(t.get("entry", 0.0))
    sl    = t.get("stop_loss")
    qty   = float(t.get("qty", 0.0))
    if sl is None or entry <= 0 or qty <= 0:
        return 0.0
    return abs(entry - float(sl)) * qty


# ── aggregation ───────────────────────────────────────────────────────────
@dataclass
class ExtendedMetrics:
    calmar_ratio:            Optional[float]
    mar_ratio:               Optional[float]
    omega_ratio:             Optional[float]
    tail_ratio:              Optional[float]
    ulcer_index:             Optional[float]
    time_under_water_days:   int
    longest_win_streak:      int
    longest_loss_streak:     int
    pct_positive_days:       float
    profit_factor:           Optional[float]
    r_multiple_avg:          Optional[float]


def compute_extended(
    values: Sequence[dict],
    trades: Optional[Sequence[dict]] = None,
) -> dict[str, Any]:
    seq = sequential_win_rate(values)
    rmu = r_multiples(trades or [])
    out = ExtendedMetrics(
        calmar_ratio          = calmar_ratio(values),
        mar_ratio             = mar_ratio(values),
        omega_ratio           = omega_ratio(values),
        tail_ratio            = tail_ratio(values),
        ulcer_index           = ulcer_index(values),
        time_under_water_days = time_under_water_days(values),
        longest_win_streak    = seq["longest_win_streak"],
        longest_loss_streak   = seq["longest_loss_streak"],
        pct_positive_days     = seq["pct_positive_days"],
        profit_factor         = profit_factor(trades or []),
        r_multiple_avg        = rmu["r_multiple_avg"],
    )
    return {**asdict(out), "r_multiple_distribution": rmu["r_multiple_distribution"]}
