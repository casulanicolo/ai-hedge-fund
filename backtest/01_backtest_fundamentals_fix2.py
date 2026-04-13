"""
01_backtest_fundamentals_fix2.py  –  Athanor Alpha | Fix 2 Validation
============================================================
Confronto: score_debt_equity (originale) vs score_debt_to_assets (Fix 2) per ticker Financials.

Testa la logica di scoring SENZA chiamare LLM:
  - score_revenue_growth
  - score_operating_margin
  - score_fcf_yield
  - score_debt_equity
  - score_insider_activity
  - composite score

Output:
  backtest/results/01_fundamentals_scores.csv
  backtest/results/01_fundamentals_summary.csv

Esegui: python backtest/01_backtest_fundamentals.py
"""

import os
import sys
import pickle
from datetime import datetime

import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "backtest", "data")
RESULT_DIR = os.path.join(BASE_DIR, "backtest", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

FUND_PATH = os.path.join(DATA_DIR, "fundamentals_12m.pkl")

LONG_THRESHOLD  = 65.0
SHORT_THRESHOLD = 35.0

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


FINANCIALS_TICKERS: frozenset[str] = frozenset({
    "JPM", "GS", "MS", "BAC", "C", "WFC",
    "V", "MA", "AXP", "PYPL",
    "BLK", "SCHW", "BX",
})


# ── Scoring (identiche a src/agents/fundamentals.py) ─────────────────────────

def score_revenue_growth(incq):
    raw = {}
    try:
        rev = incq.loc["Total Revenue"]
        q0  = float(rev.iloc[0])
        q1  = float(rev.iloc[1])
        q4  = float(rev.iloc[4]) if len(rev) > 4 else None
        qoq = (q0 - q1) / abs(q1) if q1 else None
        yoy = (q0 - q4) / abs(q4) if q4 else None
        raw["revenue_qoq_pct"] = round(qoq * 100, 2) if qoq is not None else None
        raw["revenue_yoy_pct"] = round(yoy * 100, 2) if yoy is not None else None
        score = 50.0
        if qoq is not None:
            score += max(-50, min(50, qoq * 500))
        if yoy is not None:
            score += max(-50, min(50, yoy * 250))
        score = max(0, min(100, score))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0
    return round(score, 1), raw


def score_operating_margin(incq):
    raw = {}
    try:
        op_inc  = incq.loc["Operating Income"]
        rev     = incq.loc["Total Revenue"]
        margins = []
        for i in range(min(4, len(op_inc))):
            r = float(rev.iloc[i])
            o = float(op_inc.iloc[i])
            margins.append(o / r if r else None)
        margins = [m for m in margins if m is not None]
        current = margins[0] if margins else None
        trend   = margins[0] - margins[-1] if len(margins) >= 2 else 0
        raw["op_margin_current_pct"] = round(current * 100, 2) if current is not None else None
        raw["op_margin_trend_pct"]   = round(trend * 100, 2)
        score = 50.0
        if current is not None:
            score += max(-50, min(50, current * 250))
        score += max(-20, min(20, trend * 200))
        score = max(0, min(100, score))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0
    return round(score, 1), raw


def score_fcf_yield(cfq, market_cap):
    raw = {}
    try:
        fcf_vals = cfq.loc["Free Cash Flow"]
        ttm_fcf  = sum(float(fcf_vals.iloc[i]) for i in range(min(4, len(fcf_vals))))
        raw["fcf_yield_pct"] = None
        if market_cap and market_cap > 0:
            yield_ = ttm_fcf / market_cap
            raw["fcf_yield_pct"] = round(yield_ * 100, 2)
            score = max(0, min(100, 50 + yield_ * 1000))
        else:
            score = 50.0
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0
    return round(score, 1), raw


def score_debt_equity(bsq):
    raw = {}
    try:
        debt   = bsq.loc["Total Debt"]
        equity = bsq.loc["Common Stock Equity"]
        ratios = []
        for i in range(min(4, len(debt))):
            d = float(debt.iloc[i])
            e = float(equity.iloc[i])
            ratios.append(d / e if e and e != 0 else None)
        ratios  = [r for r in ratios if r is not None]
        current = ratios[0] if ratios else None
        trend   = ratios[0] - ratios[-1] if len(ratios) >= 2 else 0
        raw["debt_equity_current"] = round(current, 3) if current is not None else None
        raw["debt_equity_trend"]   = round(trend, 3)
        score = (100 - min(100, current * 25)) if current is not None else 50.0
        score -= max(-20, min(20, trend * 20))
        score = max(0, min(100, score))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0
    return round(score, 1), raw


def score_debt_to_assets(bsq):
    """Debt/Assets per Financials (Fix 2). D/A < 0.80 = buono per banche."""
    raw = {}
    try:
        debt   = bsq.loc["Total Debt"]
        assets = bsq.loc["Total Assets"]
        ratios = []
        for i in range(min(4, len(debt))):
            d = float(debt.iloc[i])
            a = float(assets.iloc[i])
            ratios.append(d / a if a and a != 0 else None)
        ratios  = [r for r in ratios if r is not None]
        current = ratios[0] if ratios else None
        trend   = ratios[0] - ratios[-1] if len(ratios) >= 2 else 0
        raw["debt_to_assets_current"] = round(current, 3) if current is not None else None
        raw["debt_to_assets_trend"]   = round(trend, 3)
        score = (100 - max(0, min(100, (current - 0.50) * 400))) if current is not None else 50.0
        score -= max(-15, min(15, trend * 150))
        score = max(0.0, min(100.0, score))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0
    return round(score, 1), raw


def score_insider_activity(info):
    raw = {}
    try:
        held = info.get("heldPercentInsiders")
        raw["held_pct_insiders"] = held
        if held is None:
            return 50.0, raw
        score = max(0, min(100, 50 + min(50, held * 400)))
    except Exception as e:
        raw["error"] = str(e)
        score = 50.0
    return round(score, 1), raw


def direction_from_composite(composite):
    if composite > LONG_THRESHOLD:
        return "LONG"
    elif composite < SHORT_THRESHOLD:
        return "SHORT"
    return "NEUTRAL"


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 65)
    print("  ATHANOR ALPHA – Fix 2 Validation | Fundamentals: debt_to_assets per Financials")
    print(f"  Eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    if not os.path.exists(FUND_PATH):
        print(f"\n  ERRORE: {FUND_PATH} non trovato.")
        print("  Esegui prima:  python backtest/00_download_data.py")
        sys.exit(1)

    print("\n[1/3] Carico dati fondamentali ...")
    with open(FUND_PATH, "rb") as f:
        fund_data = pickle.load(f)
    print(f"  Ticker caricati: {len(fund_data)}")

    print("\n[2/3] Calcolo scores ...")
    rows = []

    for ticker, payload in fund_data.items():
        print(f"  {ticker:6s} ...", end=" ")

        incq       = payload.get("income_stmt_q")
        bsq        = payload.get("balance_sheet_q")
        cfq        = payload.get("cash_flow_q")
        info       = payload.get("info", {})
        market_cap = info.get("marketCap")

        has_income   = isinstance(incq, pd.DataFrame) and not incq.empty
        has_balance  = isinstance(bsq,  pd.DataFrame) and not bsq.empty
        has_cashflow = isinstance(cfq,  pd.DataFrame) and not cfq.empty
        data_quality = sum([has_income, has_balance, has_cashflow, market_cap is not None])

        s_rev, r_rev = score_revenue_growth(incq)        if has_income   else (50.0, {})
        s_mar, r_mar = score_operating_margin(incq)      if has_income   else (50.0, {})
        s_fcf, r_fcf = score_fcf_yield(cfq, market_cap)  if has_cashflow else (50.0, {})
        _is_financial = ticker in FINANCIALS_TICKERS or SECTOR_MAP.get(ticker, "") == "Financials"
        if _is_financial:
            s_de_orig, _ = score_debt_equity(bsq)     if has_balance else (50.0, {})
            s_de,  r_de  = score_debt_to_assets(bsq)  if has_balance else (50.0, {})
        else:
            s_de_orig = None
            s_de,  r_de  = score_debt_equity(bsq)     if has_balance else (50.0, {})
        s_ins, r_ins = score_insider_activity(info)

        composite = round((s_rev + s_mar + s_fcf + s_de + s_ins) / 5, 1)
        direction = direction_from_composite(composite)

        rows.append({
            "ticker":            ticker,
            "sector":            SECTOR_MAP.get(ticker, "Unknown"),
            "direction":         direction,
            "composite_score":   composite,
            "s_revenue_growth":  s_rev,
            "s_op_margin":       s_mar,
            "s_fcf_yield":       s_fcf,
            "s_debt_score":      s_de,
            "s_debt_equity_orig": s_de_orig,   # None se non Financials
            "is_financial":      _is_financial,
            "scoring_method":    "debt_to_assets" if _is_financial else "debt_equity",
            "s_insider":         s_ins,
            "data_quality_0_4":  data_quality,
            "revenue_qoq_pct":   r_rev.get("revenue_qoq_pct"),
            "revenue_yoy_pct":   r_rev.get("revenue_yoy_pct"),
            "op_margin_pct":     r_mar.get("op_margin_current_pct"),
            "fcf_yield_pct":     r_fcf.get("fcf_yield_pct"),
            "debt_equity_ratio": r_de.get("debt_equity_current"),
            "insider_pct":       r_ins.get("held_pct_insiders"),
            "market_cap_B":      round(market_cap / 1e9, 1) if market_cap else None,
        })

        icon = "▲ LONG" if direction == "LONG" else ("▼ SHORT" if direction == "SHORT" else "– NEUT")
        print(f"composite={composite:5.1f}  {icon}")

    print("\n[3/3] Salvataggio risultati ...")
    df = pd.DataFrame(rows)

    scores_path = os.path.join(RESULT_DIR, "01_fundamentals_scores_fix2.csv")
    df.to_csv(scores_path, index=False, float_format="%.2f")
    print(f"  {scores_path}")

    n_total   = len(df)
    n_long    = (df["direction"] == "LONG").sum()
    n_short   = (df["direction"] == "SHORT").sum()
    n_neutral = (df["direction"] == "NEUTRAL").sum()

    summary_rows = [
        {"metrica": "Ticker totali",       "valore": n_total},
        {"metrica": "Direzione LONG",      "valore": f"{n_long} ({n_long/n_total*100:.0f}%)"},
        {"metrica": "Direzione SHORT",     "valore": f"{n_short} ({n_short/n_total*100:.0f}%)"},
        {"metrica": "Direzione NEUTRAL",   "valore": f"{n_neutral} ({n_neutral/n_total*100:.0f}%)"},
        {"metrica": "Dati completi (4/4)", "valore": (df["data_quality_0_4"] == 4).sum()},
        {"metrica": "Composite medio",     "valore": round(df["composite_score"].mean(), 1)},
        {"metrica": "Revenue Growth medio","valore": round(df["s_revenue_growth"].mean(), 1)},
        {"metrica": "Op Margin medio",     "valore": round(df["s_op_margin"].mean(), 1)},
        {"metrica": "FCF Yield medio",     "valore": round(df["s_fcf_yield"].mean(), 1)},
        {"metrica": "Debt/Equity medio",   "valore": round(df["s_debt_score"].mean(), 1)},
        {"metrica": "Insider medio",       "valore": round(df["s_insider"].mean(), 1)},
        {"metrica": "LONG soglia",         "valore": LONG_THRESHOLD},
        {"metrica": "SHORT soglia",        "valore": SHORT_THRESHOLD},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULT_DIR, "01_fundamentals_summary_fix2.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  {summary_path}")

    # ── Report a schermo ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RISULTATI – Fundamentals Agent")
    print("=" * 65)
    print(f"\n  Direzioni:  LONG={n_long}  SHORT={n_short}  NEUTRAL={n_neutral}")
    print(f"\n  Score medi (0=peggio, 100=meglio):")
    print(f"    Composite       : {df['composite_score'].mean():.1f}")
    print(f"    Revenue Growth  : {df['s_revenue_growth'].mean():.1f}")
    print(f"    Op Margin       : {df['s_op_margin'].mean():.1f}")
    print(f"    FCF Yield       : {df['s_fcf_yield'].mean():.1f}")
    print(f"    Debt Score      : {df['s_debt_score'].mean():.1f}  (debt_to_assets per Financials, debt_equity altrimenti)")
    print(f"    Insider         : {df['s_insider'].mean():.1f}")

    print(f"\n  Top 5 ticker (composite piu alto):")
    for _, r in df.nlargest(5, "composite_score").iterrows():
        print(f"    {r['ticker']:6s}  {r['composite_score']:5.1f}  {r['direction']:7s}  {r['sector']}")

    print(f"\n  Bottom 5 ticker (composite piu basso):")
    for _, r in df.nsmallest(5, "composite_score").iterrows():
        print(f"    {r['ticker']:6s}  {r['composite_score']:5.1f}  {r['direction']:7s}  {r['sector']}")

    print(f"\n  Score medio per settore:")
    sect = df.groupby("sector")["composite_score"].mean().sort_values(ascending=False)
    for sector, avg in sect.items():
        print(f"    {sector:20s}  {avg:.1f}")

    print(f"\n  File salvati in backtest/results/")
    # ── Fix 2: Confronto Financials ──────────────────────────────────────────
    fin_df = df[df["is_financial"] == True].copy()
    if not fin_df.empty:
        print(f"\n  FIX 2 – Confronto score Financials (debt_equity → debt_to_assets):")
        print(f"  {'Ticker':<8} {'D/E score (orig)':>16} {'D/A score (fix2)':>16} {'Delta':>8} {'Composite fix2':>14}")
        print(f"  {'-'*66}")
        for _, r in fin_df.iterrows():
            orig = r["s_debt_equity_orig"]
            fix2 = r["s_debt_score"]
            if orig is not None:
                delta = fix2 - orig
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
                print(f"  {r['ticker']:<8} {orig:>16.1f} {fix2:>16.1f} {arrow}{abs(delta):>6.1f} {r['composite_score']:>14.1f}")
    else:
        print("\n  Nessun ticker Financials nel dataset.")

    print(f"\n  Prossimo step: python backtest/02_backtest_risk_manager.py")
    print("=" * 65)


if __name__ == "__main__":
    run_backtest()
