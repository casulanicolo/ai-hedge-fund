"""
05_backtest_operative_agents.py  -  Athanor Alpha | Backtest F4/F5 + Fix B1/B2/B5
===================================================================================
Backtest ISOLATO degli Agenti Operativi (Categoria B).

Fix B2 (transaction costs e slippage):
  - apply_costs(pnl, ticker, direction): applica costi round-trip a ogni trade
    * Equity standard:          pnl -= 2 × 0.05% = 0.10% round trip
    * Crypto/high-vol tickers:  pnl -= 2 × 0.15% = 0.30% round trip
      (HIGH_VOL_TICKERS = BTC-USD, ETH-USD, SOL-USD, MSTR, COIN, SMCI)
  - Colonne aggiuntive nei CSV: fwd_5d_net, fwd_20d_net (netti di costi)
  - Metriche riportate sia lordi (gross) che netti (net) per confronto esplicito
  - Colonne CSV: win_rate_20d_gross_pct, win_rate_20d_net_pct,
                 pnl_gross_pct, pnl_net_pct, avg_ret_gross_pct, avg_ret_net_pct

Fix B1 (finestra temporale estesa):
  - Loop su signal_dates: ogni 20 giorni trading dal 2020-01-01 a oggi
  - Colonna sub_period: 2020-2021 / 2022 / 2023-2024 / 2025-2026
  - CSV aggregato per sotto-periodo: 05_operative_by_subperiod.csv

Fix B5 (look-ahead bias):
  - filter_fundamentals_by_date(df, backtest_date)
  # LAG ASSUMPTION: fundamentals assumed available 45 days after quarter end to avoid look-ahead bias

Fix 5 (EMA filter):
  - EMA8 vs EMA21 calcolato alla data del segnale

Fix 7 (peter_lynch PEG fallback):
  - Tier 1: peg_ratio yfinance, Tier 2: pe/earnings_growth (cap PEG_CAP=10)

Output:
  backtest/results/05_operative_scores.csv         — tutti i segnali con colonne gross e net
  backtest/results/05_operative_by_subperiod.csv   — aggregato per sotto-periodo (gross + net)
  backtest/results/05_operative_comparison.csv     — ranking agenti (gross + net)
  backtest/results/05_operative_summary.csv        — riepilogo metriche globali
  backtest/results/05_lynch_peg_coverage.csv       — copertura PEG Fix 7

Esegui: python backtest/05_backtest_operative_agents.py
"""

import os
import sys
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "backtest", "data")
RESULT_DIR = os.path.join(BASE_DIR, "backtest", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

OHLCV_PATH = os.path.join(DATA_DIR, "ohlcv_5y.pkl")
FUND_PATH  = os.path.join(DATA_DIR, "fundamentals_12m.pkl")

LONG_THRESH  = 62.0
SHORT_THRESH = 38.0

# LAG ASSUMPTION: fundamentals assumed available 45 days after quarter end to avoid look-ahead bias
REPORTING_LAG_DAYS = 45

SIGNAL_FREQ_DAYS = 20

SUB_PERIODS = {
    "2020-2021": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
    "2022":      (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    "2023-2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    "2025-2026": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31")),
}

# ── Fix B2: costi di transazione ──────────────────────────────────────────────
# Equity standard:   commissioni entrata + uscita = 2 × 0.05% = 0.10% round trip
# Crypto/high-vol:   spread più ampio + slippage  = 2 × 0.15% = 0.30% round trip
COST_EQUITY_ONE_WAY  = 0.0005   # 0.05%
COST_HIGHVOL_ONE_WAY = 0.0015   # 0.15%
HIGH_VOL_TICKERS = frozenset({"BTC-USD", "ETH-USD", "SOL-USD", "MSTR", "COIN", "SMCI"})

SECTOR_MAP = {
    "AAPL": "Technology",    "MSFT": "Technology",
    "GOOGL": "Technology",   "NVDA": "Technology",
    "META": "Technology",    "AMZN": "Consumer Disc.",
    "TSLA": "Consumer Disc.","HD": "Consumer Disc.",
    "NKE": "Consumer Disc.", "WMT": "Consumer Disc.",
    "JPM": "Financials",     "V": "Financials",
    "MA": "Financials",      "BRK-B": "Financials",
    "GS": "Financials",      "UNH": "Healthcare",
    "JNJ": "Healthcare",     "PFE": "Healthcare",
    "ABBV": "Healthcare",    "MRK": "Healthcare",
    "XOM": "Energy",         "CVX": "Energy",
    "LIN": "Materials",      "APD": "Materials",
    "FCX": "Materials",      "SMCI": "Technology",
    "MELI": "Consumer Disc.","COIN": "Crypto-Proxy",
    "MSTR": "Crypto-Proxy",  "PLTR": "Technology",
}


# ── Fix B2: funzione costi ────────────────────────────────────────────────────

def apply_costs(pnl: float, ticker: str, direction: str) -> float:
    """
    Fix B2: applica costi di transazione round-trip al P&L lordo.

    Equity standard (tutti i ticker non in HIGH_VOL_TICKERS):
        pnl_net = pnl - 2 × 0.05%  (entrata + uscita)

    Crypto e ticker ad alta volatilità (HIGH_VOL_TICKERS):
        pnl_net = pnl - 2 × 0.15%  (spread più ampio + slippage stimato)

    Args:
        pnl:       return lordo del trade (es. 0.032 = +3.2%)
        ticker:    simbolo del ticker
        direction: "LONG" o "SHORT"

    Returns:
        return netto di costi
    """
    if pnl is None or np.isnan(pnl):
        return pnl
    cost_one_way = COST_HIGHVOL_ONE_WAY if ticker in HIGH_VOL_TICKERS else COST_EQUITY_ONE_WAY
    round_trip   = 2 * cost_one_way
    return pnl - round_trip


def compute_net_returns(row: pd.Series) -> tuple:
    """
    Calcola fwd_5d_net e fwd_20d_net applicando apply_costs.
    Usata su ogni riga del DataFrame segnali.
    """
    ticker    = row["ticker"]
    direction = row["direction_filtered"] if row["direction_filtered"] != "NEUTRAL" else row["direction"]
    fwd_5d_net  = apply_costs(row["fwd_5d"],  ticker, direction) if row.get("fwd_5d")  is not None else None
    fwd_20d_net = apply_costs(row["fwd_20d"], ticker, direction) if row.get("fwd_20d") is not None else None
    return fwd_5d_net, fwd_20d_net


# ── Fix B1: utilità sotto-periodi ─────────────────────────────────────────────

def get_sub_period(date: pd.Timestamp) -> str:
    for name, (start, end) in SUB_PERIODS.items():
        if start <= date <= end:
            return name
    return "other"


def build_signal_dates(ohlcv_data: dict, tickers: list) -> list:
    """
    Fix B1: costruisce la lista di date di segnale ogni SIGNAL_FREQ_DAYS giorni
    trading, dalla prima data disponibile a oggi - 21 barre.
    """
    all_dates = pd.DatetimeIndex([])
    for ticker in tickers:
        df = ohlcv_data.get(ticker)
        if df is not None and not df.empty:
            all_dates = all_dates.union(df.index)
    if all_dates.empty:
        return []
    all_dates = all_dates.sort_values()
    all_dates = all_dates[:-21] if len(all_dates) > 21 else all_dates[:0]
    signal_dates = all_dates[::SIGNAL_FREQ_DAYS].tolist()
    return signal_dates


# ── Fix B5: look-ahead bias filter ────────────────────────────────────────────

def filter_fundamentals_by_date(df: pd.DataFrame, backtest_date: datetime) -> pd.DataFrame:
    """
    Filtra le colonne di un DataFrame trimestrale yfinance tenendo solo i periodi
    con period_end_date + REPORTING_LAG_DAYS <= backtest_date.

    # LAG ASSUMPTION: fundamentals assumed available 45 days after quarter end to avoid look-ahead bias
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    cutoff = pd.Timestamp(backtest_date)
    valid_cols = [
        col for col in df.columns
        if isinstance(col, pd.Timestamp) and (col + timedelta(days=REPORTING_LAG_DAYS)) <= cutoff
    ]
    if not valid_cols:
        return pd.DataFrame()
    return df[valid_cols]


# ── Helper: estrai metriche ───────────────────────────────────────────────────

def extract_metrics(payload: dict, backtest_date: datetime) -> dict:
    """
    Estrae metriche fondamentali con filtro look-ahead (Fix B5).
    # LAG ASSUMPTION: fundamentals assumed available 45 days after quarter end to avoid look-ahead bias
    """
    info = payload.get("info", {})
    incq = filter_fundamentals_by_date(payload.get("income_stmt_q"),   backtest_date)
    bsq  = filter_fundamentals_by_date(payload.get("balance_sheet_q"), backtest_date)
    cfq  = filter_fundamentals_by_date(payload.get("cash_flow_q"),     backtest_date)

    m = {
        "pe_ratio":        info.get("trailingPE"),
        "pb_ratio":        info.get("priceToBook"),
        "ps_ratio":        info.get("priceToSalesTrailing12Months"),
        "peg_ratio":       info.get("pegRatio"),
        "ev_ebitda":       info.get("enterpriseToEbitda"),
        "market_cap":      info.get("marketCap"),
        "roe":             info.get("returnOnEquity"),
        "roa":             info.get("returnOnAssets"),
        "op_margin":       info.get("operatingMargins"),
        "net_margin":      info.get("profitMargins"),
        "gross_margin":    info.get("grossMargins"),
        "revenue_growth":  info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "debt_equity":     info.get("debtToEquity"),
        "current_ratio":   info.get("currentRatio"),
        "quick_ratio":     info.get("quickRatio"),
        "beta":            info.get("beta"),
        "dividend_yield":  info.get("dividendYield"),
        "insider_pct":     info.get("heldPercentInsiders"),
        "short_ratio":     info.get("shortRatio"),
        "52w_high":        info.get("fiftyTwoWeekHigh"),
        "52w_low":         info.get("fiftyTwoWeekLow"),
        "current_price":   info.get("currentPrice") or info.get("regularMarketPrice"),
        "total_cash":      info.get("totalCash"),
        "total_debt_abs":  info.get("totalDebt"),
    }

    if m["revenue_growth"] is None and isinstance(incq, pd.DataFrame) and not incq.empty:
        try:
            rev = incq.loc["Total Revenue"]
            q0  = float(rev.iloc[0])
            q4  = float(rev.iloc[4]) if len(rev) > 4 else None
            if q4 and q4 != 0:
                m["revenue_growth"] = (q0 - q4) / abs(q4)
        except Exception:
            pass

    m["fcf_yield"] = None
    m["fcf_abs"]   = None
    if isinstance(cfq, pd.DataFrame) and not cfq.empty and m["market_cap"]:
        try:
            fcf_vals = cfq.loc["Free Cash Flow"]
            ttm_fcf  = sum(float(fcf_vals.iloc[i]) for i in range(min(4, len(fcf_vals))))
            m["fcf_abs"]   = ttm_fcf
            m["fcf_yield"] = ttm_fcf / m["market_cap"]
        except Exception:
            pass

    m["net_cash_positive"] = None
    if m["total_cash"] is not None and m["total_debt_abs"] is not None:
        m["net_cash_positive"] = m["total_cash"] > m["total_debt_abs"]

    m["debt_equity_frac"] = m["debt_equity"] / 100.0 if m["debt_equity"] is not None else None
    return m


def safe(val, default=None):
    if val is None:
        return default
    try:
        if np.isnan(float(val)):
            return default
    except (TypeError, ValueError):
        return default
    return val


def direction(score: float) -> str:
    if score >= LONG_THRESH:  return "LONG"
    if score <= SHORT_THRESH: return "SHORT"
    return "NEUTRAL"


def get_ema_trend_at_date(df: pd.DataFrame, signal_date: pd.Timestamp) -> str:
    if df is None or df.empty:
        return "FLAT"
    try:
        past = df[df.index <= signal_date]["Close"].astype(float).dropna()
        if len(past) < 22:
            return "FLAT"
        ema8  = past.ewm(span=8,  adjust=False).mean()
        ema21 = past.ewm(span=21, adjust=False).mean()
        if float(ema8.iloc[-1]) > float(ema21.iloc[-1]):   return "UP"
        elif float(ema8.iloc[-1]) < float(ema21.iloc[-1]): return "DOWN"
        return "FLAT"
    except Exception:
        return "FLAT"


def apply_ema_filter_backtest(direction_raw: str, ema_trend: str) -> str:
    if direction_raw == "NEUTRAL": return "NEUTRAL"
    if ema_trend == "FLAT":        return direction_raw
    if direction_raw == "LONG":    return "LONG"  if ema_trend == "UP"   else "NEUTRAL"
    if direction_raw == "SHORT":   return "SHORT" if ema_trend == "DOWN" else "NEUTRAL"
    return direction_raw


def get_forward_returns_at_date(df: pd.DataFrame, signal_date: pd.Timestamp) -> dict:
    if df is None or df.empty:
        return {"fwd_5d": None, "fwd_20d": None}
    try:
        future = df[df.index >= signal_date]["Close"].astype(float).dropna()
        if len(future) < 2:
            return {"fwd_5d": None, "fwd_20d": None}
        p0      = float(future.iloc[0])
        fwd_5d  = round(float(future.iloc[min(5,  len(future)-1)]) / p0 - 1, 4) if len(future) > 5  else None
        fwd_20d = round(float(future.iloc[min(20, len(future)-1)]) / p0 - 1, 4) if len(future) > 20 else None
        return {"fwd_5d": fwd_5d, "fwd_20d": fwd_20d}
    except Exception:
        return {"fwd_5d": None, "fwd_20d": None}


# ── Scoring per filosofia ─────────────────────────────────────────────────────

def score_warren_buffett(m: dict) -> float:
    score = 50.0
    roe = safe(m["roe"])
    if roe is not None:
        if roe > 0.20:   score += 15
        elif roe > 0.15: score += 8
        elif roe < 0.05: score -= 10
    op_m = safe(m["op_margin"])
    if op_m is not None:
        if op_m > 0.20:   score += 10
        elif op_m > 0.12: score += 5
        elif op_m < 0.05: score -= 8
    pe = safe(m["pe_ratio"])
    if pe is not None:
        if 10 < pe < 20:    score += 10
        elif 20 <= pe < 30: score += 5
        elif pe > 50:       score -= 10
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf > 0.05:   score += 10
        elif fcf > 0.02: score += 5
        elif fcf < 0:    score -= 10
    de = safe(m["debt_equity_frac"])
    if de is not None:
        if de < 0.5:   score += 5
        elif de > 2.0: score -= 8
    return round(max(0, min(100, score)), 1)


def score_ben_graham(m: dict) -> float:
    score = 50.0
    pb = safe(m["pb_ratio"])
    if pb is not None:
        if pb < 1.0:   score += 20
        elif pb < 1.5: score += 12
        elif pb < 2.5: score += 5
        elif pb > 5.0: score -= 15
    pe = safe(m["pe_ratio"])
    if pe is not None:
        if pe < 10:   score += 15
        elif pe < 15: score += 8
        elif pe > 25: score -= 10
    cr = safe(m["current_ratio"])
    if cr is not None:
        if cr >= 2.0:   score += 10
        elif cr >= 1.5: score += 5
        elif cr < 1.0:  score -= 10
    div = safe(m["dividend_yield"])
    if div is not None and div > 0.02:
        score += 8
    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac < 0.5:   score += 7
        elif de_frac > 1.0: score -= 10
    return round(max(0, min(100, score)), 1)


def score_charlie_munger(m: dict) -> float:
    score = 50.0
    gm = safe(m["gross_margin"])
    if gm is not None:
        if gm > 0.50:   score += 15
        elif gm > 0.35: score += 8
        elif gm < 0.15: score -= 10
    roe = safe(m["roe"])
    if roe is not None:
        if roe > 0.25:   score += 15
        elif roe > 0.15: score += 7
        elif roe < 0:    score -= 15
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf > 0.05:   score += 10
        elif fcf > 0.02: score += 5
        elif fcf < 0:    score -= 10
    nm = safe(m["net_margin"])
    if nm is not None:
        if nm > 0.20: score += 8
        elif nm < 0:  score -= 10
    pe = safe(m["pe_ratio"])
    if pe is not None:
        if pe > 60: score -= 12
    return round(max(0, min(100, score)), 1)


def score_bill_ackman(m: dict) -> float:
    score = 50.0
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf > 0.07:   score += 20
        elif fcf > 0.04: score += 10
        elif fcf < 0:    score -= 15
    op_m = safe(m["op_margin"])
    if op_m is not None:
        if op_m > 0.25:   score += 15
        elif op_m > 0.15: score += 7
        elif op_m < 0.05: score -= 10
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.20:   score += 8
        elif rg > 0.05: score += 4
        elif rg < 0:    score -= 8
    roe = safe(m["roe"])
    if roe is not None:
        if roe > 0.15: score += 7
    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac < 1.0:   score += 5
        elif de_frac > 3.0: score -= 10
    return round(max(0, min(100, score)), 1)


def score_cathie_wood(m: dict) -> float:
    score = 50.0
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.50:   score += 25
        elif rg > 0.30: score += 15
        elif rg > 0.10: score += 7
        elif rg < 0:    score -= 15
    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.50: score += 10
        elif eg > 0.20: score += 5
    gm = safe(m["gross_margin"])
    if gm is not None:
        if gm > 0.60:   score += 10
        elif gm > 0.40: score += 5
    beta = safe(m["beta"])
    if beta is not None:
        if beta > 1.8:   score += 8
        elif beta > 1.2: score += 4
        elif beta < 0.8: score -= 5
    div = safe(m["dividend_yield"])
    if div is not None:
        if div < 0.005:  score += 5
        elif div > 0.03: score -= 5
    return round(max(0, min(100, score)), 1)


def score_michael_burry(m: dict) -> float:
    score = 50.0
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf >= 0.15:   score += 25
        elif fcf >= 0.12: score += 18
        elif fcf >= 0.08: score += 12
        elif fcf >= 0.04: score += 5
        elif fcf < 0:     score -= 15
    pb = safe(m["pb_ratio"])
    if pb is not None:
        if pb < 0.8:   score += 15
        elif pb < 1.5: score += 7
        elif pb > 4.0: score -= 15
    net_cash = safe(m["net_cash_positive"])
    if net_cash is True:   score += 10
    elif net_cash is False: score -= 5
    sr = safe(m["short_ratio"])
    if sr is not None:
        if sr > 5:   score += 8
        elif sr > 2: score += 4
    pe = safe(m["pe_ratio"])
    if pe is not None:
        if 0 < pe < 12: score += 10
        elif pe > 40:   score -= 10
    return round(max(0, min(100, score)), 1)


def score_mohnish_pabrai(m: dict) -> float:
    score = 50.0
    if safe(m["net_cash_positive"]) is True:   score += 15
    elif safe(m["net_cash_positive"]) is False: score -= 5
    cr = safe(m["current_ratio"])
    if cr is not None:
        if cr >= 2.0:   score += 10
        elif cr >= 1.2: score += 5
        elif cr < 1.0:  score -= 10
    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac < 0.3:   score += 8
        elif de_frac < 0.7: score += 4
        elif de_frac > 1.5: score -= 8
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf > 0.06:    score += 12
        elif fcf > 0.045: score += 8
        elif fcf > 0.03:  score += 5
        elif fcf > 0.02:  score += 2
        elif fcf < 0:     score -= 10
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.15:   score += 5
        elif rg > 0.05: score += 2
    return round(max(0, min(100, score)), 1)


PEG_CAP = 10.0


def _compute_peg_internal(m: dict) -> tuple:
    peg_ext = safe(m["peg_ratio"])
    if peg_ext is not None and peg_ext > 0:
        return peg_ext, "external_yfinance"
    pe = safe(m["pe_ratio"])
    if pe is None or pe <= 0:
        return None, "no_pe"
    eg = safe(m["earnings_growth"])
    if eg is not None and eg > 0:
        peg = pe / (eg * 100)
        if peg < PEG_CAP:
            return peg, "internal_earnings_growth"
        return None, f"internal_earnings_growth_capped ({peg:.1f} >= {PEG_CAP})"
    return None, "no_growth_data"


def score_peter_lynch(m: dict) -> float:
    score = 50.0
    peg = safe(m["peg_ratio"])
    if peg is not None:
        if peg < 0.5:   score += 25
        elif peg < 1.0: score += 15
        elif peg < 1.5: score += 5
        elif peg > 2.5: score -= 15
        elif peg > 3.0: score -= 25
    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.25:   score += 10
        elif eg > 0.10: score += 5
        elif eg < 0:    score -= 10
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.15:    score += 7
        elif rg > 0.05:  score += 3
        elif rg < -0.05: score -= 7
    cr = safe(m["current_ratio"])
    if cr is not None:
        if cr > 1.5:   score += 5
        elif cr < 1.0: score -= 5
    div = safe(m["dividend_yield"])
    if div is not None and div > 0.01:
        score += 3
    return round(max(0, min(100, score)), 1)


def score_peter_lynch_fix7(m: dict) -> float:
    score = 50.0
    peg, _ = _compute_peg_internal(m)
    if peg is not None:
        if peg < 0.5:   score += 25
        elif peg < 1.0: score += 15
        elif peg < 1.5: score += 5
        elif peg > 2.5: score -= 15
        elif peg > 3.0: score -= 25
    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.25:   score += 10
        elif eg > 0.10: score += 5
        elif eg < 0:    score -= 10
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.15:    score += 7
        elif rg > 0.05:  score += 3
        elif rg < -0.05: score -= 7
    cr = safe(m["current_ratio"])
    if cr is not None:
        if cr > 1.5:   score += 5
        elif cr < 1.0: score -= 5
    div = safe(m["dividend_yield"])
    if div is not None and div > 0.01:
        score += 3
    return round(max(0, min(100, score)), 1)


def score_phil_fisher(m: dict) -> float:
    score = 50.0
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.20:   score += 18
        elif rg > 0.10: score += 10
        elif rg > 0.05: score += 5
        elif rg < 0:    score -= 12
    gm = safe(m["gross_margin"])
    if gm is not None:
        if gm > 0.55:   score += 12
        elif gm > 0.35: score += 6
        elif gm < 0.15: score -= 8
    op_m = safe(m["op_margin"])
    if op_m is not None:
        if op_m > 0.20:   score += 10
        elif op_m > 0.10: score += 5
        elif op_m < 0:    score -= 10
    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac < 0.5:   score += 8
        elif de_frac > 1.5: score -= 8
    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.15: score += 7
        elif eg < 0:  score -= 7
    return round(max(0, min(100, score)), 1)


def score_rakesh_jhunjhunwala(m: dict) -> float:
    score = 50.0
    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.25:   score += 18
        elif eg > 0.15: score += 10
        elif eg > 0.05: score += 5
        elif eg < 0:    score -= 12
    op_m = safe(m["op_margin"])
    if op_m is not None:
        if op_m > 0.20:   score += 10
        elif op_m > 0.12: score += 5
        elif op_m < 0.05: score -= 8
    roe = safe(m["roe"])
    if roe is not None:
        if roe > 0.20:   score += 8
        elif roe > 0.12: score += 4
        elif roe < 0:    score -= 8
    cr = safe(m["current_ratio"])
    if cr is not None:
        if cr >= 1.5:  score += 5
        elif cr < 1.0: score -= 7
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf > 0.05:   score += 8
        elif fcf > 0.02: score += 4
        elif fcf < 0:    score -= 8
    return round(max(0, min(100, score)), 1)


def score_stanley_druckenmiller(m: dict) -> float:
    score = 50.0
    cur = safe(m["current_price"])
    h52 = safe(m["52w_high"])
    if cur and h52 and h52 > 0:
        prox = cur / h52
        if prox > 0.95:   score += 20
        elif prox > 0.85: score += 12
        elif prox > 0.75: score += 5
        elif prox < 0.60: score -= 12
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.30:   score += 15
        elif rg > 0.10: score += 8
        elif rg < 0:    score -= 12
    beta = safe(m["beta"])
    if beta is not None:
        if 1.2 < beta < 2.0:  score += 8
        elif beta > 2.5:      score -= 3
        elif beta < 0.7:      score -= 5
    op_m = safe(m["op_margin"])
    if op_m is not None:
        if op_m > 0.20:   score += 7
        elif op_m < 0.05: score -= 5
    return round(max(0, min(100, score)), 1)


def score_aswath_damodaran(m: dict) -> float:
    score = 50.0
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        beta    = safe(m["beta"], 1.0)
        cost_eq = 0.045 + beta * 0.05
        if fcf > cost_eq + 0.03:  score += 20
        elif fcf > cost_eq:       score += 10
        elif fcf > 0:             score += 3
        elif fcf < 0:             score -= 15
    peg = safe(m["peg_ratio"])
    if peg is not None:
        if peg < 1.0:   score += 12
        elif peg < 2.0: score += 5
        elif peg > 3.0: score -= 10
    roe = safe(m["roe"])
    if roe is not None:
        if roe > 0.18:   score += 10
        elif roe > 0.10: score += 5
        elif roe < 0:    score -= 10
    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac < 0.6:   score += 5
        elif de_frac > 2.0: score -= 8
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.15:    score += 5
        elif rg < -0.05: score -= 5
    return round(max(0, min(100, score)), 1)


def score_growth_agent(m: dict) -> float:
    score = 50.0
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.40:    score += 25
        elif rg > 0.20:  score += 15
        elif rg > 0.10:  score += 7
        elif rg < -0.05: score -= 15
    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.40:    score += 12
        elif eg > 0.20:  score += 6
        elif eg < -0.10: score -= 8
    gm = safe(m["gross_margin"])
    if gm is not None:
        if gm > 0.50:   score += 8
        elif gm > 0.35: score += 4
    op_m = safe(m["op_margin"])
    if op_m is not None:
        if op_m > 0.15: score += 7
        elif op_m < 0:  score -= 8
    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac > 1.5:   score -= 5
        elif de_frac < 0.8: score += 3
    cr = safe(m["current_ratio"])
    if cr is not None:
        if cr < 1.0:    score -= 5
        elif cr >= 1.5: score += 3
    peg = safe(m["peg_ratio"])
    if peg is not None:
        if peg < 1.0:   score += 8
        elif peg < 2.0: score += 4
        elif peg > 4.0: score -= 5
    return round(max(0, min(100, score)), 1)


# ── Registry ──────────────────────────────────────────────────────────────────

AGENTS = {
    "warren_buffett":        score_warren_buffett,
    "ben_graham":            score_ben_graham,
    "charlie_munger":        score_charlie_munger,
    "bill_ackman":           score_bill_ackman,
    "cathie_wood":           score_cathie_wood,
    "michael_burry":         score_michael_burry,
    "mohnish_pabrai":        score_mohnish_pabrai,
    "peter_lynch":           score_peter_lynch,
    "peter_lynch_fix7":      score_peter_lynch_fix7,
    "phil_fisher":           score_phil_fisher,
    "rakesh_jhunjhunwala":   score_rakesh_jhunjhunwala,
    "stanley_druckenmiller": score_stanley_druckenmiller,
    "aswath_damodaran":      score_aswath_damodaran,
    "growth_agent":          score_growth_agent,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 70)
    print("  ATHANOR ALPHA — Fix B1/B2/B5/Fix7  |  Backtest: Agenti Operativi")
    print(f"  Eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Fix B2: riepilogo costi
    print(f"\n  [Fix B2] Costi transazione:")
    print(f"    Equity standard : {COST_EQUITY_ONE_WAY*2*100:.2f}% round trip  (2×{COST_EQUITY_ONE_WAY*100:.2f}%)")
    print(f"    High-vol/Crypto : {COST_HIGHVOL_ONE_WAY*2*100:.2f}% round trip  (2×{COST_HIGHVOL_ONE_WAY*100:.2f}%)")
    print(f"    Ticker high-vol : {sorted(HIGH_VOL_TICKERS)}")

    if not os.path.exists(OHLCV_PATH):
        old_path = os.path.join(DATA_DIR, "ohlcv_12m.pkl")
        if os.path.exists(old_path):
            print(f"\n  ERRORE: ohlcv_5y.pkl non trovato. Fix B1 richiede storia dal 2020.")
        else:
            print(f"\n  ERRORE: {OHLCV_PATH} non trovato.")
        print("  Esegui prima:  python backtest/00_download_data.py")
        sys.exit(1)

    print("\n[1/6] Carico dati ...")
    with open(FUND_PATH,  "rb") as f: fund_data  = pickle.load(f)
    with open(OHLCV_PATH, "rb") as f: ohlcv_data = pickle.load(f)
    tickers = list(fund_data.keys())
    print(f"  Ticker: {len(tickers)}  |  Agenti: {len(AGENTS)}")

    print("\n[2/6] Fix B1 — costruzione date di segnale ...")
    signal_dates = build_signal_dates(ohlcv_data, tickers)
    print(f"  Date di segnale: {len(signal_dates)}")
    if signal_dates:
        print(f"  Prima: {signal_dates[0].strftime('%Y-%m-%d')}  |  Ultima: {signal_dates[-1].strftime('%Y-%m-%d')}")
        for pname, (ps, pe) in SUB_PERIODS.items():
            n = sum(1 for d in signal_dates if ps <= d <= pe)
            print(f"    {pname:12s}: {n:3d} date")

    print("\n[3/6] Fix 7 — analisi copertura PEG ...")
    coverage_rows = []
    latest_date = signal_dates[-1] if signal_dates else datetime.today()
    for ticker in tickers:
        m = extract_metrics(fund_data[ticker], latest_date)
        peg_ext = safe(m["peg_ratio"])
        peg_val, peg_src = _compute_peg_internal(m)
        coverage_rows.append({
            "ticker":        ticker,
            "peg_external":  peg_ext,
            "peg_used":      round(peg_val, 3) if peg_val is not None else None,
            "peg_source":    peg_src,
            "peg_was_none":  peg_ext is None,
            "peg_recovered": peg_ext is None and peg_val is not None,
        })
    cov_df = pd.DataFrame(coverage_rows)
    n_none_ext  = cov_df["peg_was_none"].sum()
    n_recovered = cov_df["peg_recovered"].sum()
    print(f"  PEG esterno None: {n_none_ext}  |  Recuperati Fix 7: {n_recovered}")

    print(f"\n[4/6] Calcolo score — {len(signal_dates)} date × {len(tickers)} ticker × {len(AGENTS)} agenti ...")
    rows = []
    for i, sig_date in enumerate(signal_dates):
        if i % 10 == 0:
            pct = i / len(signal_dates) * 100
            print(f"  {sig_date.strftime('%Y-%m-%d')}  [{pct:4.0f}%]  righe={len(rows):,}", end="\r")

        sub_period = get_sub_period(sig_date)

        for ticker in tickers:
            ohlcv_df = ohlcv_data.get(ticker)
            if ohlcv_df is None or ohlcv_df.empty:
                continue
            if ohlcv_df.index[0] > sig_date:
                continue

            # Fix B5: fundamentals filtrati alla data del segnale
            # LAG ASSUMPTION: fundamentals assumed available 45 days after quarter end to avoid look-ahead bias
            m         = extract_metrics(fund_data[ticker], sig_date)
            fwd       = get_forward_returns_at_date(ohlcv_df, sig_date)
            ema_trend = get_ema_trend_at_date(ohlcv_df, sig_date)

            for agent_name, score_fn in AGENTS.items():
                s = score_fn(m)
                if agent_name == "mohnish_pabrai":
                    dir_raw = "LONG" if s >= 58.0 else ("SHORT" if s <= SHORT_THRESH else "NEUTRAL")
                else:
                    dir_raw = direction(s)
                dir_filt = apply_ema_filter_backtest(dir_raw, ema_trend)

                # Fix B2: calcola ritorni netti di costi
                # Applica costi solo se c'è un segnale attivo (non NEUTRAL)
                active_dir = dir_filt if dir_filt != "NEUTRAL" else dir_raw
                fwd_5d_net  = apply_costs(fwd["fwd_5d"],  ticker, active_dir) if fwd["fwd_5d"]  is not None else None
                fwd_20d_net = apply_costs(fwd["fwd_20d"], ticker, active_dir) if fwd["fwd_20d"] is not None else None

                rows.append({
                    "signal_date":        sig_date.strftime("%Y-%m-%d"),
                    "sub_period":         sub_period,
                    "agent":              agent_name,
                    "ticker":             ticker,
                    "sector":             SECTOR_MAP.get(ticker, "Unknown"),
                    "score":              s,
                    "ema_trend":          ema_trend,
                    "direction":          dir_raw,
                    "direction_filtered": dir_filt,
                    "fwd_5d":             fwd["fwd_5d"],       # lordo
                    "fwd_20d":            fwd["fwd_20d"],      # lordo
                    "fwd_5d_net":         fwd_5d_net,          # netto Fix B2
                    "fwd_20d_net":        fwd_20d_net,         # netto Fix B2
                })

    print(f"\n  Completato. Righe totali: {len(rows):,}")
    df = pd.DataFrame(rows)

    print("\n[5/6] Calcolo metriche per agente (gross + net) ...")
    agent_stats = []
    for agent in AGENTS.keys():
        ag = df[df["agent"] == agent].copy()
        ag_f = ag[ag["direction_filtered"].isin(["LONG", "SHORT"])].copy().dropna(subset=["fwd_20d", "fwd_20d_net"])

        if len(ag_f) == 0:
            continue

        # Gross metrics
        ag_f["win_gross"] = np.where(ag_f["direction_filtered"] == "LONG",
                                     ag_f["fwd_20d"] > 0, ag_f["fwd_20d"] < 0)
        wr_gross = ag_f["win_gross"].mean() * 100

        long_g  = ag_f[ag_f["direction_filtered"] == "LONG"]["fwd_20d"]
        short_g = ag_f[ag_f["direction_filtered"] == "SHORT"]["fwd_20d"]
        pnl_gross     = long_g.sum() * 100 + (-short_g).sum() * 100
        avg_ret_gross = ag_f["fwd_20d"].mean() * 100

        # Net metrics (Fix B2)
        ag_f["win_net"] = np.where(ag_f["direction_filtered"] == "LONG",
                                   ag_f["fwd_20d_net"] > 0, ag_f["fwd_20d_net"] < 0)
        wr_net = ag_f["win_net"].mean() * 100

        long_n  = ag_f[ag_f["direction_filtered"] == "LONG"]["fwd_20d_net"]
        short_n = ag_f[ag_f["direction_filtered"] == "SHORT"]["fwd_20d_net"]
        pnl_net     = long_n.sum() * 100 + (-short_n).sum() * 100
        avg_ret_net = ag_f["fwd_20d_net"].mean() * 100

        n_long_f  = (ag_f["direction_filtered"] == "LONG").sum()
        n_short_f = (ag_f["direction_filtered"] == "SHORT").sum()
        n_long    = (ag["direction"] == "LONG").sum()
        n_short   = (ag["direction"] == "SHORT").sum()

        # Fix B2: cost drag = differenza pnl lordo - netto
        cost_drag = pnl_gross - pnl_net

        agent_stats.append({
            "agent":                    agent,
            "n_signals_ema":            len(ag_f),
            "n_long":                   n_long,
            "n_short":                  n_short,
            "n_long_ema":               n_long_f,
            "n_short_ema":              n_short_f,
            "avg_score":                round(ag["score"].mean(), 1),
            # Gross
            "win_rate_20d_gross_pct":   round(wr_gross, 1),
            "pnl_gross_pct":            round(pnl_gross, 2),
            "avg_ret_gross_pct":        round(avg_ret_gross, 3),
            # Net (Fix B2)
            "win_rate_20d_net_pct":     round(wr_net, 1),
            "pnl_net_pct":              round(pnl_net, 2),
            "avg_ret_net_pct":          round(avg_ret_net, 3),
            # Delta gross-net
            "cost_drag_pct":            round(cost_drag, 2),
            "delta_wr_gross_net_pp":    round(wr_gross - wr_net, 1),
        })

        print(f"  {agent:25s}  "
              f"WR_gross={wr_gross:.0f}%  WR_net={wr_net:.0f}%  "
              f"PnL_gross={pnl_gross:+.1f}%  PnL_net={pnl_net:+.1f}%  "
              f"cost_drag={cost_drag:.1f}%")

    agent_df = pd.DataFrame(agent_stats).sort_values("win_rate_20d_net_pct", ascending=False)

    # ── Aggregato per sotto-periodo (gross + net) ─────────────────────────────
    print("\n  Aggregato per sotto-periodo (gross vs net):")
    subperiod_rows = []
    prod_list = [a for a in AGENTS.keys() if a != "peter_lynch_fix7"]
    for sp in list(SUB_PERIODS.keys()) + ["other"]:
        sp_df = df[(df["sub_period"] == sp) & (df["agent"].isin(prod_list))].copy()
        sp_f  = sp_df[sp_df["direction_filtered"].isin(["LONG", "SHORT"])].dropna(subset=["fwd_20d", "fwd_20d_net"])
        if sp_f.empty:
            continue

        sp_f["win_g"] = np.where(sp_f["direction_filtered"] == "LONG", sp_f["fwd_20d"] > 0,     sp_f["fwd_20d"] < 0)
        sp_f["win_n"] = np.where(sp_f["direction_filtered"] == "LONG", sp_f["fwd_20d_net"] > 0, sp_f["fwd_20d_net"] < 0)
        wr_g = sp_f["win_g"].mean() * 100
        wr_n = sp_f["win_n"].mean() * 100

        long_g  = sp_f[sp_f["direction_filtered"] == "LONG"]["fwd_20d"]
        short_g = sp_f[sp_f["direction_filtered"] == "SHORT"]["fwd_20d"]
        long_n  = sp_f[sp_f["direction_filtered"] == "LONG"]["fwd_20d_net"]
        short_n = sp_f[sp_f["direction_filtered"] == "SHORT"]["fwd_20d_net"]
        avg_g = sp_f["fwd_20d"].mean() * 100
        avg_n = sp_f["fwd_20d_net"].mean() * 100

        n_dates = sp_df["signal_date"].nunique()
        print(f"    {sp:12s}  dates={n_dates:3d}  WR_gross={wr_g:.1f}%  WR_net={wr_n:.1f}%  avg_gross={avg_g:.3f}%  avg_net={avg_n:.3f}%")

        subperiod_rows.append({
            "sub_period":           sp,
            "n_signal_dates":       n_dates,
            "n_signals_ema":        len(sp_f),
            "win_rate_20d_gross_pct": round(wr_g, 1),
            "win_rate_20d_net_pct":   round(wr_n, 1),
            "avg_ret_gross_pct":      round(avg_g, 3),
            "avg_ret_net_pct":        round(avg_n, 3),
            "delta_wr_gross_net_pp":  round(wr_g - wr_n, 1),
        })
    subperiod_df = pd.DataFrame(subperiod_rows)

    # ── Salvataggio CSV ───────────────────────────────────────────────────────
    print("\n[6/6] Salvataggio CSV ...")
    scores_path    = os.path.join(RESULT_DIR, "05_operative_scores.csv")
    comp_path      = os.path.join(RESULT_DIR, "05_operative_comparison.csv")
    summ_path      = os.path.join(RESULT_DIR, "05_operative_summary.csv")
    coverage_path  = os.path.join(RESULT_DIR, "05_lynch_peg_coverage.csv")
    subperiod_path = os.path.join(RESULT_DIR, "05_operative_by_subperiod.csv")

    df.to_csv(scores_path, index=False, float_format="%.4f")
    agent_df.to_csv(comp_path, index=False, float_format="%.3f")
    cov_df.to_csv(coverage_path, index=False, float_format="%.4f")
    subperiod_df.to_csv(subperiod_path, index=False, float_format="%.3f")

    prod_df = agent_df[agent_df["agent"] != "peter_lynch_fix7"]
    avg_wr_gross = prod_df["win_rate_20d_gross_pct"].mean()
    avg_wr_net   = prod_df["win_rate_20d_net_pct"].mean()
    best_gross   = agent_df.sort_values("win_rate_20d_gross_pct", ascending=False).iloc[0]["agent"]
    best_net     = agent_df.sort_values("win_rate_20d_net_pct",   ascending=False).iloc[0]["agent"]

    pd.DataFrame([
        {"metrica": "Ticker",                      "valore": len(tickers)},
        {"metrica": "Agenti (prod)",               "valore": len(AGENTS) - 1},
        {"metrica": "Signal dates (Fix B1)",       "valore": len(signal_dates)},
        {"metrica": "Signal freq giorni",          "valore": SIGNAL_FREQ_DAYS},
        {"metrica": "Prima signal date",           "valore": signal_dates[0].strftime("%Y-%m-%d") if signal_dates else "N/A"},
        {"metrica": "Ultima signal date",          "valore": signal_dates[-1].strftime("%Y-%m-%d") if signal_dates else "N/A"},
        {"metrica": "Righe totali",                "valore": len(df)},
        {"metrica": "reporting_lag_days (Fix B5)", "valore": REPORTING_LAG_DAYS},
        {"metrica": "cost_equity_roundtrip (B2)",  "valore": f"{COST_EQUITY_ONE_WAY*2*100:.2f}%"},
        {"metrica": "cost_highvol_roundtrip (B2)", "valore": f"{COST_HIGHVOL_ONE_WAY*2*100:.2f}%"},
        {"metrica": "high_vol_tickers (B2)",       "valore": str(sorted(HIGH_VOL_TICKERS))},
        {"metrica": "WR 20d gross medio (prod)",   "valore": f"{avg_wr_gross:.1f}%"},
        {"metrica": "WR 20d net medio (prod)",     "valore": f"{avg_wr_net:.1f}%"},
        {"metrica": "Miglior agente gross WR20d",  "valore": best_gross},
        {"metrica": "Miglior agente net WR20d",    "valore": best_net},
        {"metrica": "[Fix7] PEG None",             "valore": int(n_none_ext)},
        {"metrica": "[Fix7] PEG recuperati",       "valore": int(n_recovered)},
        {"metrica": "LONG_THRESH",                 "valore": LONG_THRESH},
        {"metrica": "SHORT_THRESH",                "valore": SHORT_THRESH},
    ]).to_csv(summ_path, index=False)

    print(f"\n  {scores_path}")
    print(f"  {comp_path}")
    print(f"  {summ_path}")
    print(f"  {coverage_path}")
    print(f"  {subperiod_path}")

    # ── Report finale ─────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  CLASSIFICA AGENTI — Gross vs Net (Fix B2)  |  agenti di produzione")
    print("=" * 85)
    print(f"  {'Agente':25s}  {'WR_gross':>8}{'WR_net':>8}  {'PnL_gross':>10}{'PnL_net':>10}  {'CostDrag':>9}  {'N_L':>6}{'N_S':>6}")
    print("  " + "-" * 85)
    for _, r in prod_df.iterrows():
        print(f"  {r['agent']:25s}  "
              f"{r['win_rate_20d_gross_pct']:7.1f}% {r['win_rate_20d_net_pct']:7.1f}%  "
              f"{r['pnl_gross_pct']:+9.1f}% {r['pnl_net_pct']:+9.1f}%  "
              f"{r['cost_drag_pct']:+8.1f}%  "
              f"{r['n_long_ema']:5.0f} {r['n_short_ema']:5.0f}")

    print(f"\n  Media WR 20d  gross: {avg_wr_gross:.1f}%   net: {avg_wr_net:.1f}%   delta: {avg_wr_gross-avg_wr_net:+.1f}pp")
    print(f"  Miglior agente gross WR20d : {best_gross}")
    print(f"  Miglior agente net WR20d   : {best_net}")

    print(f"\n  Sotto-periodi (net WR 20d):")
    for _, r in subperiod_df.iterrows():
        print(f"    {r['sub_period']:12s}  WR_gross={r['win_rate_20d_gross_pct']:.1f}%  WR_net={r['win_rate_20d_net_pct']:.1f}%  "
              f"avg_net={r['avg_ret_net_pct']:.3f}%  segnali={r['n_signals_ema']}")

    print(f"\n  Prossimo step: python backtest/06_backtest_report.py")
    print("=" * 85)


if __name__ == "__main__":
    run_backtest()
