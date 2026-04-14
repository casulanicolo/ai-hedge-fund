"""
05_backtest_operative_agents.py  -  Athanor Alpha | Backtest F4/F5
===================================================================
Backtest ISOLATO degli Agenti Operativi (Categoria B).

Scoring proxy costruito leggendo la logica reale di ogni file agente.
NON usa LLM. Usa solo metriche yfinance gia scaricate.

Fix 5 (2026-04-14):
  - Aggiunto filtro EMA per simulare apply_ema_filter() in produzione
  - get_ema_trend(): calcola EMA8 vs EMA21 sull'ultimo giorno disponibile in OHLCV
  - apply_ema_filter_backtest(): LONG passa solo se EMA8>EMA21, SHORT solo se EMA8<EMA21
  - Metriche riportate sia SENZA filtro (baseline) che CON filtro (Fix 5)
  - Colonna "direction_filtered" nei CSV di output

Agenti testati:
  1. warren_buffett        – ROE, op margin, P/E ragionevole, FCF, D/E
  2. ben_graham            – P/B basso, current ratio, debt ratio, dividendi, EPS stabile
  3. charlie_munger        – ROIC, gross margin, predictability, FCF vs capex
  4. bill_ackman           – FCF yield, op margin, revenue growth, D/E
  5. cathie_wood           – revenue growth (accelerating), gross margin, R&D proxy (beta alto)
  6. michael_burry         – FCF yield altissimo, P/B basso, short ratio, net cash
  7. mohnish_pabrai        – net cash, current ratio, D/E basso, FCF yield normalizzato
  8. peter_lynch           – PEG ratio, earnings growth, current ratio, dividendi
  9. phil_fisher           – crescita revenue multi-anno, op margin, R&D proxy, D/E basso
  10. rakesh_jhunjhunwala  – crescita EPS + op margin + FCF + bilancio sano + buyback
  11. stanley_druckenmiller – momentum (52w high proximity), revenue growth, beta
  12. aswath_damodaran     – FCF yield disciplinato, PEG, D/E, ROE vs cost of equity
  13. growth_agent         – revenue growth + EPS growth + gross margin + op margin

Metriche misurate per agente:
  - n_long, n_short, n_neutral
  - bias direzionale
  - win rate a 5gg e 20gg (forward return reali da OHLCV)
  - P&L simulato equal-weight
  - settori preferiti (top 2)

Output:
  backtest/results/05_operative_scores.csv
  backtest/results/05_operative_comparison.csv
  backtest/results/05_operative_summary.csv

Esegui: python backtest/05_backtest_operative_agents.py
"""

import os
import sys
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "backtest", "data")
RESULT_DIR = os.path.join(BASE_DIR, "backtest", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

OHLCV_PATH = os.path.join(DATA_DIR, "ohlcv_12m.pkl")
FUND_PATH  = os.path.join(DATA_DIR, "fundamentals_12m.pkl")

LONG_THRESH  = 62.0
SHORT_THRESH = 38.0

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


# ── Helper: estrai metriche da payload yfinance ───────────────────────────────

def extract_metrics(payload: dict) -> dict:
    info = payload.get("info", {})
    incq = payload.get("income_stmt_q")
    bsq  = payload.get("balance_sheet_q")
    cfq  = payload.get("cash_flow_q")

    m = {
        "pe_ratio":       info.get("trailingPE"),
        "pb_ratio":       info.get("priceToBook"),
        "ps_ratio":       info.get("priceToSalesTrailing12Months"),
        "peg_ratio":      info.get("pegRatio"),
        "ev_ebitda":      info.get("enterpriseToEbitda"),
        "market_cap":     info.get("marketCap"),
        "roe":            info.get("returnOnEquity"),
        "roa":            info.get("returnOnAssets"),
        "op_margin":      info.get("operatingMargins"),
        "net_margin":     info.get("profitMargins"),
        "gross_margin":   info.get("grossMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth":info.get("earningsGrowth"),
        "debt_equity":    info.get("debtToEquity"),       # yfinance: % (es. 150 = 1.5x)
        "current_ratio":  info.get("currentRatio"),
        "quick_ratio":    info.get("quickRatio"),
        "beta":           info.get("beta"),
        "dividend_yield": info.get("dividendYield"),
        "insider_pct":    info.get("heldPercentInsiders"),
        "short_ratio":    info.get("shortRatio"),
        "52w_high":       info.get("fiftyTwoWeekHigh"),
        "52w_low":        info.get("fiftyTwoWeekLow"),
        "current_price":  info.get("currentPrice") or info.get("regularMarketPrice"),
        "total_cash":     info.get("totalCash"),
        "total_debt_abs": info.get("totalDebt"),
    }

    # Revenue growth da income statement se non in info
    if m["revenue_growth"] is None and isinstance(incq, pd.DataFrame) and not incq.empty:
        try:
            rev = incq.loc["Total Revenue"]
            q0 = float(rev.iloc[0])
            q4 = float(rev.iloc[4]) if len(rev) > 4 else None
            if q4 and q4 != 0:
                m["revenue_growth"] = (q0 - q4) / abs(q4)
        except Exception:
            pass

    # FCF yield da cash flow statement
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

    # Net cash = cash - total_debt
    m["net_cash_positive"] = None
    if m["total_cash"] is not None and m["total_debt_abs"] is not None:
        m["net_cash_positive"] = m["total_cash"] > m["total_debt_abs"]

    # debt_equity in yfinance è espresso in % (es. 150 = D/E 1.5)
    # normalizziamo a frazione
    if m["debt_equity"] is not None:
        m["debt_equity_frac"] = m["debt_equity"] / 100.0
    else:
        m["debt_equity_frac"] = None

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


def get_ema_trend(ohlcv_data: dict, ticker: str) -> str:
    """
    Calcola trend EMA8 vs EMA21 sull'ultimo giorno disponibile.
    Simula _get_ema_trend() di src/utils/ema_filter.py.
    Returns: "UP" | "DOWN" | "FLAT"
    """
    df = ohlcv_data.get(ticker)
    if df is None or len(df) < 22:
        return "FLAT"
    try:
        close = df["Close"].astype(float).dropna()
        if len(close) < 22:
            return "FLAT"
        ema8  = close.ewm(span=8,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        if float(ema8.iloc[-1]) > float(ema21.iloc[-1]):
            return "UP"
        elif float(ema8.iloc[-1]) < float(ema21.iloc[-1]):
            return "DOWN"
        return "FLAT"
    except Exception:
        return "FLAT"


def apply_ema_filter_backtest(direction_raw: str, ema_trend: str) -> str:
    """
    Applica il filtro EMA alla direzione grezza.
    Logica identica a apply_ema_filter() di src/utils/ema_filter.py.
    """
    if direction_raw == "NEUTRAL":
        return "NEUTRAL"
    if ema_trend == "FLAT":
        return direction_raw   # fail-safe: dati insufficienti
    if direction_raw == "LONG":
        return "LONG" if ema_trend == "UP" else "NEUTRAL"
    if direction_raw == "SHORT":
        return "SHORT" if ema_trend == "DOWN" else "NEUTRAL"
    return direction_raw


# ── Scoring per filosofia (calibrate su logica reale dei file agente) ─────────

def score_warren_buffett(m: dict) -> float:
    """
    Logica reale: analyze_fundamentals (ROE, D/E, op_margin, current_ratio)
    + analyze_moat (ROE consistency) + calculate_intrinsic_value
    Peso: qualità fondamentale + FCF yield + margine di sicurezza
    """
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
        if 10 < pe < 20:   score += 10
        elif 20 <= pe < 30: score += 5
        elif pe > 50:       score -= 10

    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf > 0.05:   score += 10
        elif fcf > 0.02: score += 5
        elif fcf < 0:    score -= 10

    de = safe(m["debt_equity_frac"])
    if de is not None:
        if de < 0.5:  score += 5
        elif de > 2.0: score -= 8

    return round(max(0, min(100, score)), 1)


def score_ben_graham(m: dict) -> float:
    """
    Logica reale: analyze_earnings_stability (EPS positivo, crescita)
    + analyze_financial_strength (current ratio >=2, debt_ratio <0.5, dividendi)
    + analyze_valuation_graham (P/B, Graham Number, NCAV)
    """
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
        if cr >= 2.0:  score += 10
        elif cr >= 1.5: score += 5
        elif cr < 1.0: score -= 10

    div = safe(m["dividend_yield"])
    if div is not None and div > 0.02:
        score += 8

    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac < 0.5:  score += 7
        elif de_frac > 1.0: score -= 10

    return round(max(0, min(100, score)), 1)


def score_charlie_munger(m: dict) -> float:
    """
    Logica reale: moat_strength (ROIC, gross margin trend), management quality,
    predictability (FCF consistency), valuation (FCF-based).
    Munger pesa ROIC e gross margin più di PE.
    """
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
        if nm > 0.20:   score += 8
        elif nm < 0:    score -= 10

    pe = safe(m["pe_ratio"])
    if pe is not None:
        if pe > 60: score -= 12

    return round(max(0, min(100, score)), 1)


def score_bill_ackman(m: dict) -> float:
    """
    Logica reale: business quality (ROE >15%, op margin >15%, FCF+),
    financial discipline (D/E <1, buybacks), activism potential,
    valuation (FCF-based DCF con MOS >30%)
    """
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
        if de_frac < 1.0:  score += 5
        elif de_frac > 3.0: score -= 10

    return round(max(0, min(100, score)), 1)


def score_cathie_wood(m: dict) -> float:
    """
    Logica reale: disruptive_potential (revenue growth accelerante, gross margin,
    R&D intensity, operating leverage) + innovation_growth (FCF growth, margin trend)
    + valuation (DCF con crescita 20%). Cathie tollera P/E alto ma vuole crescita FORTE.
    """
    score = 50.0
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.50:    score += 25
        elif rg > 0.30:  score += 15
        elif rg > 0.10:  score += 7
        elif rg < 0:     score -= 15

    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.50: score += 10
        elif eg > 0.20: score += 5

    gm = safe(m["gross_margin"])
    if gm is not None:
        if gm > 0.60:   score += 10
        elif gm > 0.40: score += 5

    # Beta alto = Cathie ama ticker ad alta volatilità (innovation proxy)
    beta = safe(m["beta"])
    if beta is not None:
        if beta > 1.8:   score += 8
        elif beta > 1.2: score += 4
        elif beta < 0.8: score -= 5

    # Dividendi bassi = reinvestimento (Cathie preferisce no-dividend)
    div = safe(m["dividend_yield"])
    if div is not None:
        if div < 0.005:  score += 5   # quasi zero dividendi = tutto reinvestito
        elif div > 0.03: score -= 5   # dividendi alti = matura, non growth

    return round(max(0, min(100, score)), 1)


def score_michael_burry(m: dict) -> float:
    """
    Logica reale: _analyze_value (FCF yield >8-15%, EV/EBIT basso)
    + _analyze_balance_sheet (D/E <0.5, net cash)
    + _analyze_insider_activity + contrarian_sentiment
    Burry cerca FCF yield altissimo + bilancio solido. Contrarian.
    """
    score = 50.0
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf >= 0.15:   score += 25   # "extraordinary"
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
    if net_cash is True:
        score += 10
    elif net_cash is False:
        score -= 5

    # Short ratio alto = potenziale contrarian
    sr = safe(m["short_ratio"])
    if sr is not None:
        if sr > 5:   score += 8
        elif sr > 2: score += 4

    pe = safe(m["pe_ratio"])
    if pe is not None:
        if 0 < pe < 12:   score += 10
        elif pe > 40:     score -= 10

    return round(max(0, min(100, score)), 1)


def score_mohnish_pabrai(m: dict) -> float:
    """
    Logica reale: downside_protection (net cash, current ratio >=2, D/E <0.3, FCF stabile)
    + pabrai_valuation (FCF yield >5-10%, asset-light basso capex)
    + double_potential (revenue growth + FCF growth)
    Pesi: downside 45%, valuation 35%, doubling 20%
    """
    score = 50.0
    # Downside protection (peso 45%)
    if safe(m["net_cash_positive"]) is True:
        score += 15
    elif safe(m["net_cash_positive"]) is False:
        score -= 5

    cr = safe(m["current_ratio"])
    if cr is not None:
        if cr >= 2.0:   score += 10
        elif cr >= 1.2: score += 5
        elif cr < 1.0:  score -= 10

    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac < 0.3:  score += 8
        elif de_frac < 0.7: score += 4
        elif de_frac > 1.5: score -= 8

    # Valuation (peso 35%)
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf > 0.10:   score += 12
        elif fcf > 0.07: score += 8
        elif fcf > 0.05: score += 5
        elif fcf > 0.03: score += 2
        elif fcf < 0:    score -= 10

    # Doubling potential (peso 20%)
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.15:  score += 5
        elif rg > 0.05: score += 2

    return round(max(0, min(100, score)), 1)


def score_peter_lynch(m: dict) -> float:
    """
    Logica reale: PEG ratio è la metrica principale (< 1 = ottimo).
    + analyze_lynch_growth (EPS + revenue multi-anno)
    + current_ratio, dividend check
    """
    score = 50.0
    peg = safe(m["peg_ratio"])
    if peg is not None:
        if peg < 0.5:    score += 25
        elif peg < 1.0:  score += 15
        elif peg < 1.5:  score += 5
        elif peg > 2.5:  score -= 15
        elif peg > 3.0:  score -= 25

    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.25:   score += 10
        elif eg > 0.10: score += 5
        elif eg < 0:    score -= 10

    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.15:   score += 7
        elif rg > 0.05: score += 3
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
    """
    Logica reale: analyze_fisher_growth_quality (revenue multi-anno, EPS),
    analyze_margins_stability (op margin, gross margin),
    management quality (D/E basso, R&D proxy: gross margin alto = pricing power)
    Fisher paga di più per qualità — P/E alto OK se crescita forte.
    """
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
        if de_frac < 0.5:  score += 8
        elif de_frac > 1.5: score -= 8

    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.15: score += 7
        elif eg < 0:  score -= 7

    return round(max(0, min(100, score)), 1)


def score_rakesh_jhunjhunwala(m: dict) -> float:
    """
    Logica reale: analyze_growth (EPS, revenue, op income crescita),
    analyze_profitability (op margin, ROE), analyze_balance_sheet (current ratio, D/E),
    analyze_cash_flow (FCF+), analyze_management_actions (buyback, dividendi),
    calculate_intrinsic_value.
    Jhunjhunwala: crescita EPS forte + bilancio solido + FCF positivo.
    """
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
        if cr >= 1.5:   score += 5
        elif cr < 1.0:  score -= 7

    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        if fcf > 0.05:   score += 8
        elif fcf > 0.02: score += 4
        elif fcf < 0:    score -= 8

    return round(max(0, min(100, score)), 1)


def score_stanley_druckenmiller(m: dict) -> float:
    """
    Logica reale: macro + momentum. 52w high proximity è il segnale principale.
    + revenue growth, beta, op margin.
    Druckenmiller entra quando il momentum è chiaro e il business è in crescita.
    """
    score = 50.0
    cur   = safe(m["current_price"])
    h52   = safe(m["52w_high"])
    if cur and h52 and h52 > 0:
        proximity = cur / h52
        if proximity > 0.95:    score += 20   # vicino al massimo = momentum forte
        elif proximity > 0.85:  score += 12
        elif proximity > 0.75:  score += 5
        elif proximity < 0.60:  score -= 12   # lontano dal max = momentum assente

    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.30:    score += 15
        elif rg > 0.10:  score += 8
        elif rg < 0:     score -= 12

    beta = safe(m["beta"])
    if beta is not None:
        if 1.2 < beta < 2.0:   score += 8
        elif beta > 2.5:       score -= 3
        elif beta < 0.7:       score -= 5

    op_m = safe(m["op_margin"])
    if op_m is not None:
        if op_m > 0.20:   score += 7
        elif op_m < 0.05: score -= 5

    return round(max(0, min(100, score)), 1)


def score_aswath_damodaran(m: dict) -> float:
    """
    Logica reale: analyze_growth_and_reinvestment (FCF growth),
    analyze_risk_profile (beta → cost of equity),
    calculate_intrinsic_value_dcf (FCFF DCF),
    analyze_relative_valuation (PE vs settore).
    Damodaran: DCF disciplinato. FCF yield vs cost of equity.
    """
    score = 50.0
    fcf = safe(m["fcf_yield"])
    if fcf is not None:
        # Cost of equity proxy: risk-free 4.5% + beta * 5% ERP
        beta = safe(m["beta"], 1.0)
        cost_eq = 0.045 + beta * 0.05
        if fcf > cost_eq + 0.03:   score += 20
        elif fcf > cost_eq:        score += 10
        elif fcf > 0:              score += 3
        elif fcf < 0:              score -= 15

    peg = safe(m["peg_ratio"])
    if peg is not None:
        if peg < 1.0:    score += 12
        elif peg < 2.0:  score += 5
        elif peg > 3.0:  score -= 10

    roe = safe(m["roe"])
    if roe is not None:
        if roe > 0.18:   score += 10
        elif roe > 0.10: score += 5
        elif roe < 0:    score -= 10

    de_frac = safe(m["debt_equity_frac"])
    if de_frac is not None:
        if de_frac < 0.6:  score += 5
        elif de_frac > 2.0: score -= 8

    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.15:   score += 5
        elif rg < -0.05: score -= 5

    return round(max(0, min(100, score)), 1)


def score_growth_agent(m: dict) -> float:
    """
    Logica reale: analyze_growth_trends (revenue_growth >20%, EPS growth),
    analyze_valuation (PEG <1, P/S <2),
    analyze_margin_trends (gross >50%, op >15%),
    check_financial_health (D/E <1.5, current >1.5).
    Pesi: growth 40%, valuation 25%, margins 15%, insider 10%, health 10%.
    """
    score = 50.0
    rg = safe(m["revenue_growth"])
    if rg is not None:
        if rg > 0.40:   score += 25   # peso 40% = contributo max ~25
        elif rg > 0.20: score += 15
        elif rg > 0.10: score += 7
        elif rg < -0.05: score -= 15

    eg = safe(m["earnings_growth"])
    if eg is not None:
        if eg > 0.40:   score += 12
        elif eg > 0.20: score += 6
        elif eg < -0.10: score -= 8

    gm = safe(m["gross_margin"])
    if gm is not None:
        if gm > 0.50:   score += 8
        elif gm > 0.35: score += 4

    op_m = safe(m["op_margin"])
    if op_m is not None:
        if op_m > 0.15:   score += 7
        elif op_m < 0:    score -= 8

    # Health check
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


# ── Registry ─────────────────────────────────────────────────────────────────

AGENTS = {
    "warren_buffett":        score_warren_buffett,
    "ben_graham":            score_ben_graham,
    "charlie_munger":        score_charlie_munger,
    "bill_ackman":           score_bill_ackman,
    "cathie_wood":           score_cathie_wood,
    "michael_burry":         score_michael_burry,
    "mohnish_pabrai":        score_mohnish_pabrai,
    "peter_lynch":           score_peter_lynch,
    "phil_fisher":           score_phil_fisher,
    "rakesh_jhunjhunwala":   score_rakesh_jhunjhunwala,
    "stanley_druckenmiller": score_stanley_druckenmiller,
    "aswath_damodaran":      score_aswath_damodaran,
    "growth_agent":          score_growth_agent,
}


# ── Forward return da OHLCV ───────────────────────────────────────────────────

def get_forward_returns(ohlcv_data: dict, ticker: str) -> dict:
    df = ohlcv_data.get(ticker)
    if df is None or len(df) < 21:
        return {"fwd_5d": None, "fwd_20d": None, "total_ret": None}
    close = df["Close"].astype(float)
    n = len(close)
    return {
        "fwd_5d":    round(float(close.iloc[min(5,  n-1)] / close.iloc[0] - 1), 4),
        "fwd_20d":   round(float(close.iloc[min(20, n-1)] / close.iloc[0] - 1), 4),
        "total_ret": round(float(close.iloc[-1]            / close.iloc[0] - 1), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 65)
    print("  ATHANOR ALPHA – F4  |  Backtest: Agenti Operativi (Cat. B)")
    print(f"  Eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    print("\n[1/4] Carico dati ...")
    with open(FUND_PATH,  "rb") as f: fund_data  = pickle.load(f)
    with open(OHLCV_PATH, "rb") as f: ohlcv_data = pickle.load(f)
    tickers = list(fund_data.keys())
    print(f"  Ticker: {len(tickers)}  |  Agenti: {len(AGENTS)}")

    print("\n[2/4] Calcolo score per ogni agente x ticker ...")
    rows = []
    for ticker in tickers:
        m         = extract_metrics(fund_data[ticker])
        fwd       = get_forward_returns(ohlcv_data, ticker)
        ema_trend = get_ema_trend(ohlcv_data, ticker)   # Fix 5
        for agent_name, score_fn in AGENTS.items():
            s        = score_fn(m)
            dir_raw  = direction(s)
            dir_filt = apply_ema_filter_backtest(dir_raw, ema_trend)   # Fix 5
            rows.append({
                "agent":              agent_name,
                "ticker":             ticker,
                "sector":             SECTOR_MAP.get(ticker, "Unknown"),
                "score":              s,
                "ema_trend":          ema_trend,
                "direction":          dir_raw,
                "direction_filtered": dir_filt,
                "fwd_5d":             fwd["fwd_5d"],
                "fwd_20d":            fwd["fwd_20d"],
                "total_ret":          fwd["total_ret"],
            })

    df = pd.DataFrame(rows)

    print("\n[3/4] Calcolo metriche per agente ...")
    agent_stats = []
    for agent in AGENTS.keys():
        ag = df[df["agent"] == agent].copy()

        # ── Baseline (senza filtro EMA) ───────────────────────────────────
        n_long    = (ag["direction"] == "LONG").sum()
        n_short   = (ag["direction"] == "SHORT").sum()
        n_neutral = (ag["direction"] == "NEUTRAL").sum()

        ag_sig = ag[ag["direction"].isin(["LONG", "SHORT"])].copy()
        ag_sig["win_5d"]  = np.where(ag_sig["direction"] == "LONG", ag_sig["fwd_5d"] > 0,  ag_sig["fwd_5d"] < 0)
        ag_sig["win_20d"] = np.where(ag_sig["direction"] == "LONG", ag_sig["fwd_20d"] > 0, ag_sig["fwd_20d"] < 0)
        wr_5d  = ag_sig["win_5d"].mean()  * 100 if len(ag_sig) > 0 else 0
        wr_20d = ag_sig["win_20d"].mean() * 100 if len(ag_sig) > 0 else 0
        pnl_long  = ag[ag["direction"] == "LONG"]["fwd_20d"].sum()  * 100
        pnl_short = ag[ag["direction"] == "SHORT"]["fwd_20d"].sum() * -100
        pnl_total = pnl_long + pnl_short

        # ── Fix 5: con filtro EMA ─────────────────────────────────────────
        n_long_f    = (ag["direction_filtered"] == "LONG").sum()
        n_short_f   = (ag["direction_filtered"] == "SHORT").sum()
        n_neutral_f = (ag["direction_filtered"] == "NEUTRAL").sum()

        ag_sig_f = ag[ag["direction_filtered"].isin(["LONG", "SHORT"])].copy()
        ag_sig_f["win_5d_f"]  = np.where(ag_sig_f["direction_filtered"] == "LONG", ag_sig_f["fwd_5d"] > 0,  ag_sig_f["fwd_5d"] < 0)
        ag_sig_f["win_20d_f"] = np.where(ag_sig_f["direction_filtered"] == "LONG", ag_sig_f["fwd_20d"] > 0, ag_sig_f["fwd_20d"] < 0)
        wr_5d_f  = ag_sig_f["win_5d_f"].mean()  * 100 if len(ag_sig_f) > 0 else 0
        wr_20d_f = ag_sig_f["win_20d_f"].mean() * 100 if len(ag_sig_f) > 0 else 0
        pnl_long_f  = ag[ag["direction_filtered"] == "LONG"]["fwd_20d"].sum()  * 100
        pnl_short_f = ag[ag["direction_filtered"] == "SHORT"]["fwd_20d"].sum() * -100
        pnl_total_f = pnl_long_f + pnl_short_f

        bias = ("BULLISH" if n_long > n_short * 2 else
                "BEARISH" if n_short > n_long * 2 else "BALANCED")

        top_sectors = (ag[ag["direction"] == "LONG"]["sector"]
                       .value_counts().head(2).index.tolist() if n_long > 0 else [])

        agent_stats.append({
            "agent":                 agent,
            # Baseline
            "n_long":                n_long,
            "n_short":               n_short,
            "n_neutral":             n_neutral,
            "bias":                  bias,
            "avg_score":             round(ag["score"].mean(), 1),
            "win_rate_5d_pct":       round(wr_5d,  1),
            "win_rate_20d_pct":      round(wr_20d, 1),
            "pnl_simulated_pct":     round(pnl_total, 2),
            # Fix 5: filtrato
            "n_long_ema":            n_long_f,
            "n_short_ema":           n_short_f,
            "n_neutral_ema":         n_neutral_f,
            "win_rate_5d_ema_pct":   round(wr_5d_f,  1),
            "win_rate_20d_ema_pct":  round(wr_20d_f, 1),
            "pnl_ema_pct":           round(pnl_total_f, 2),
            "delta_wr5d_pp":         round(wr_5d_f  - wr_5d,  1),
            "delta_wr20d_pp":        round(wr_20d_f - wr_20d, 1),
            "top_sectors":           ", ".join(top_sectors),
        })
        print(f"  {agent:25s}  "
              f"BASE: L={n_long:2d} S={n_short:2d}  WR5={wr_5d:.0f}% WR20={wr_20d:.0f}%  |  "
              f"EMA:  L={n_long_f:2d} S={n_short_f:2d}  WR5={wr_5d_f:.0f}% WR20={wr_20d_f:.0f}%  "
              f"Δ5d={wr_5d_f-wr_5d:+.0f}pp")

    agent_df = pd.DataFrame(agent_stats).sort_values("win_rate_20d_ema_pct", ascending=False)

    print("\n[4/4] Salvataggio CSV ...")
    scores_path = os.path.join(RESULT_DIR, "05_operative_scores.csv")
    comp_path   = os.path.join(RESULT_DIR, "05_operative_comparison.csv")
    summ_path   = os.path.join(RESULT_DIR, "05_operative_summary.csv")

    df.to_csv(scores_path, index=False, float_format="%.4f")
    agent_df.to_csv(comp_path, index=False, float_format="%.2f")

    best_wr_base = agent_df.sort_values("win_rate_20d_pct",     ascending=False).iloc[0]["agent"]
    best_wr_ema  = agent_df.sort_values("win_rate_20d_ema_pct", ascending=False).iloc[0]["agent"]
    best_pnl     = agent_df.sort_values("pnl_simulated_pct",    ascending=False).iloc[0]["agent"]
    best_pnl_ema = agent_df.sort_values("pnl_ema_pct",          ascending=False).iloc[0]["agent"]

    # Medie aggregate baseline vs filtrato
    avg_wr5_base  = agent_df["win_rate_5d_pct"].mean()
    avg_wr20_base = agent_df["win_rate_20d_pct"].mean()
    avg_wr5_ema   = agent_df["win_rate_5d_ema_pct"].mean()
    avg_wr20_ema  = agent_df["win_rate_20d_ema_pct"].mean()

    pd.DataFrame([
        {"metrica": "Ticker",                      "valore": len(tickers)},
        {"metrica": "Agenti",                      "valore": len(AGENTS)},
        {"metrica": "[BASE] Miglior WR 20d",        "valore": best_wr_base},
        {"metrica": "[BASE] WR 20d medio",          "valore": f"{avg_wr20_base:.1f}%"},
        {"metrica": "[BASE] WR 5d medio",           "valore": f"{avg_wr5_base:.1f}%"},
        {"metrica": "[EMA]  Miglior WR 20d",        "valore": best_wr_ema},
        {"metrica": "[EMA]  WR 20d medio",          "valore": f"{avg_wr20_ema:.1f}%"},
        {"metrica": "[EMA]  WR 5d medio",           "valore": f"{avg_wr5_ema:.1f}%"},
        {"metrica": "Delta WR 5d  (EMA - BASE)",    "valore": f"{avg_wr5_ema - avg_wr5_base:+.1f}pp"},
        {"metrica": "Delta WR 20d (EMA - BASE)",    "valore": f"{avg_wr20_ema - avg_wr20_base:+.1f}pp"},
        {"metrica": "[BASE] Miglior P&L",           "valore": best_pnl},
        {"metrica": "[EMA]  Miglior P&L",           "valore": best_pnl_ema},
        {"metrica": "LONG_THRESH",                  "valore": LONG_THRESH},
        {"metrica": "SHORT_THRESH",                 "valore": SHORT_THRESH},
        {"metrica": "Nota",                         "valore": "Score proxy su metriche yfinance reali, senza LLM"},
    ]).to_csv(summ_path, index=False)

    print(f"\n  {scores_path}")
    print(f"  {comp_path}")
    print(f"  {summ_path}")

    print("\n" + "=" * 80)
    print("  CLASSIFICA AGENTI — BASELINE vs FIX 5 (EMA Filter)")
    print("=" * 80)
    print(f"  {'Agente':25s}  {'WR5_B':>6}{'WR20_B':>7}  {'WR5_E':>6}{'WR20_E':>7}  {'D5':>4}{'D20':>4}  {'L_E':>4}{'S_E':>4}")
    print("  " + "-" * 80)
    for _, r in agent_df.iterrows():
        print(f"  {r['agent']:25s}  "
              f"{r['win_rate_5d_pct']:5.1f}% {r['win_rate_20d_pct']:6.1f}%  "
              f"{r['win_rate_5d_ema_pct']:5.1f}% {r['win_rate_20d_ema_pct']:6.1f}%  "
              f"{r['delta_wr5d_pp']:+3.0f}pp{r['delta_wr20d_pp']:+3.0f}pp  "
              f"{r['n_long_ema']:3.0f} {r['n_short_ema']:3.0f}")

    print(f"\n  Media BASELINE  : WR_5d={avg_wr5_base:.1f}%  WR_20d={avg_wr20_base:.1f}%")
    print(f"  Media EMA Filter: WR_5d={avg_wr5_ema:.1f}%  WR_20d={avg_wr20_ema:.1f}%")
    print(f"  Delta            : WR_5d={avg_wr5_ema-avg_wr5_base:+.1f}pp  WR_20d={avg_wr20_ema-avg_wr20_base:+.1f}pp")
    print(f"\n  Miglior agente EMA (WR 20d): {best_wr_ema}")
    print(f"  Miglior agente EMA (P&L)   : {best_pnl_ema}")
    print(f"\n  Prossimo step: python backtest/06_backtest_report.py")
    print("=" * 80)


if __name__ == "__main__":
    run_backtest()
