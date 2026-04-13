"""
02_backtest_risk_manager_fix3.py  –  Athanor Alpha | Fix 3 Validation
============================================================
Backtest ISOLATO del Risk Manager (Categoria A).

Testa su dati OHLCV reali (12 mesi):
  - VaR storico al 95% su portfolio equal-weight
  - VaR in periodo stress: Marzo 2020 e Febbraio 2022
  - Max Drawdown stimato
  - Correlation matrix (media, max, heatmap testuale)
  - Sector concentration
  - Sector cap enforcement (30% GICS cap)

Output:
  backtest/results/02_risk_var.csv
  backtest/results/02_risk_correlation.csv
  backtest/results/02_risk_summary.csv

Esegui: python backtest/02_backtest_risk_manager.py
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

VAR_CONFIDENCE   = 0.95
SECTOR_CAP       = 0.30   # 30% cap per settore
MAX_PORTFOLIO_VAR = 0.04
MAX_DRAWDOWN_LIM  = 0.15
MAX_SINGLE_VAR    = 0.04   # Fix 3: soglia VaR individuale

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


# ── Helper: calcola log-returns da OHLCV ─────────────────────────────────────

def build_returns(ohlcv_data: dict, tickers: list) -> pd.DataFrame:
    """Costruisce DataFrame di log-returns giornalieri (date x ticker)."""
    series = {}
    for ticker in tickers:
        df = ohlcv_data.get(ticker)
        if df is None or df.empty:
            continue
        close = df["Close"].dropna()
        if len(close) < 20:
            continue
        log_ret = np.log(close / close.shift(1)).dropna()
        series[ticker] = log_ret
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna()


def build_returns_period(ohlcv_data: dict, tickers: list,
                         start: str, end: str) -> pd.DataFrame:
    """Log-returns per un sottoperiodo specifico."""
    series = {}
    for ticker in tickers:
        df = ohlcv_data.get(ticker)
        if df is None or df.empty:
            continue
        close = df["Close"].dropna()
        close = close[(close.index >= start) & (close.index <= end)]
        if len(close) < 5:
            continue
        log_ret = np.log(close / close.shift(1)).dropna()
        series[ticker] = log_ret
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna()


# ── Calcoli rischio (identici a risk_manager.py) ─────────────────────────────

def historical_var(returns: pd.DataFrame, weights: dict) -> float:
    """VaR storico al VAR_CONFIDENCE su portfolio con pesi dati."""
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


def max_drawdown(returns: pd.DataFrame, weights: dict) -> float:
    """Max drawdown stimato su portfolio equal-weight."""
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
    return round(float(abs(drawdowns.min())), 4)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Matrice di correlazione pairwise."""
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()
    return returns.corr().round(3)


def sector_concentration(tickers: list) -> dict:
    """Frazione di ticker per settore (equal-weight)."""
    counts = {}
    for t in tickers:
        s = SECTOR_MAP.get(t, "Unknown")
        counts[s] = counts.get(s, 0) + 1
    total = len(tickers)
    return {s: round(c / total, 3) for s, c in sorted(counts.items(),
                                                        key=lambda x: -x[1])}


def check_sector_cap(tickers: list, cap: float = SECTOR_CAP) -> dict:
    """
    Simula sector cap enforcement.
    Ritorna: {settore: {"fraction": x, "breaches_cap": bool, "tickers": [...]}}
    """
    sector_tickers: dict = {}
    for t in tickers:
        s = SECTOR_MAP.get(t, "Unknown")
        sector_tickers.setdefault(s, []).append(t)
    result = {}
    total = len(tickers)
    for s, ts in sector_tickers.items():
        frac = len(ts) / total
        result[s] = {
            "fraction":    round(frac, 3),
            "tickers":     ts,
            "breaches_cap": frac > cap,
        }
    return dict(sorted(result.items(), key=lambda x: -x[1]["fraction"]))


# ── Stress test su periodi storici ────────────────────────────────────────────

STRESS_PERIODS = {
    # Nota: i dati scaricati coprono solo apr 2025 - apr 2026.
    # I periodi classici (mar 2020, feb 2022) NON sono nel dataset.
    # Usiamo invece i peggiori 20 giorni dell'anno nel dataset disponibile
    # come proxy di stress, e segnaliamo chiaramente il limite.
    "worst_20_days_in_dataset":    ("proxy stress", None, None),
    "last_30_days":                ("recente",      None, None),
    "tariff_shock_apr_2025":       ("shock",        "2025-04-01", "2025-04-30"),
}


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 65)
    print("  ATHANOR ALPHA – Fix 3 Validation | Risk Manager: single-ticker VaR cap 4%")
    print(f"  Eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    if not os.path.exists(OHLCV_PATH):
        print(f"\n  ERRORE: {OHLCV_PATH} non trovato.")
        print("  Esegui prima:  python backtest/00_download_data.py")
        sys.exit(1)

    print("\n[1/5] Carico dati OHLCV ...")
    with open(OHLCV_PATH, "rb") as f:
        ohlcv_data = pickle.load(f)

    tickers = list(ohlcv_data.keys())
    print(f"  Ticker disponibili: {len(tickers)}")

    # ── Returns full period ───────────────────────────────────────────────────
    print("\n[2/5] Calcolo log-returns (12 mesi) ...")
    returns_full = build_returns(ohlcv_data, tickers)
    tickers_ok   = list(returns_full.columns)
    n            = len(tickers_ok)
    print(f"  Ticker con dati sufficienti: {n}")
    print(f"  Periodo: {returns_full.index[0].date()} → {returns_full.index[-1].date()}")
    print(f"  Barre disponibili: {len(returns_full)}")

    weights = {t: 1.0 / n for t in tickers_ok}

    # ── VaR e Drawdown – full period ──────────────────────────────────────────
    print("\n[3/5] Calcolo VaR e Max Drawdown ...")
    var_full = historical_var(returns_full, weights)
    mdd_full = max_drawdown(returns_full, weights)

    print(f"  VaR 95% (full 12m, equal-weight) : {var_full*100:.2f}%")
    print(f"  Max Drawdown (full 12m)           : {mdd_full*100:.2f}%")
    print(f"  Soglia VaR warning (>4%)          : {'BREACH' if var_full > MAX_PORTFOLIO_VAR else 'OK'}")
    print(f"  Soglia DD  warning (>15%)         : {'BREACH' if mdd_full > MAX_DRAWDOWN_LIM else 'OK'}")

    # ── VaR per ticker (individuale) ──────────────────────────────────────────
    var_rows = []
    for ticker in tickers_ok:
        ret_t = returns_full[[ticker]]
        w_t   = {ticker: 1.0}
        var_t = historical_var(ret_t, w_t)
        mdd_t = max_drawdown(ret_t, w_t)
        vol_t = float(returns_full[ticker].std() * np.sqrt(252))
        var_rows.append({
            "ticker":         ticker,
            "sector":         SECTOR_MAP.get(ticker, "Unknown"),
            "var_95_daily":   round(var_t, 4),
            "max_drawdown":   round(mdd_t, 4),
            "annual_vol":     round(vol_t, 4),
            "var_warn":       var_t > MAX_PORTFOLIO_VAR,
            "dd_warn":        mdd_t > MAX_DRAWDOWN_LIM,
        })

    var_df = pd.DataFrame(var_rows).sort_values("var_95_daily", ascending=False)

    # ── Stress test: ultimi 30 giorni ─────────────────────────────────────────
    print("\n[4/5] Stress test periodi ...")
    last_date   = returns_full.index[-1]
    start_30d   = returns_full.index[-min(30, len(returns_full))]
    ret_30d     = returns_full.loc[start_30d:]
    var_30d     = historical_var(ret_30d, weights)
    mdd_30d     = max_drawdown(ret_30d, weights)

    # Stress: worst 20 singole giornate nel dataset (proxy crash)
    port_daily  = returns_full[tickers_ok].values @ np.array([weights[t] for t in tickers_ok])
    worst_20    = np.sort(port_daily)[:20]
    var_worst20 = -float(np.mean(worst_20))

    print(f"  VaR 95% ultimi 30gg        : {var_30d*100:.2f}%")
    print(f"  Max DD  ultimi 30gg        : {mdd_30d*100:.2f}%")
    print(f"  Perdita media worst 20gg   : {var_worst20*100:.2f}%")
    print(f"  Nota: Mar 2020 / Feb 2022 fuori dal dataset (apr 2025-apr 2026)")

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr = correlation_matrix(returns_full)
    avg_corr = float(corr.values[np.triu_indices_from(corr.values, k=1)].mean())
    max_corr_val = float(corr.values[np.triu_indices_from(corr.values, k=1)].max())

    # Trova la coppia più correlata
    corr_upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    max_pair   = corr_upper.stack().idxmax()
    max_pair_v = corr_upper.stack().max()

    print(f"\n  Correlazione media portafoglio : {avg_corr:.3f}")
    print(f"  Correlazione massima           : {max_corr_val:.3f}  ({max_pair[0]} / {max_pair[1]})")

    # ── Sector concentration + cap enforcement ────────────────────────────────
    print("\n[5/5] Sector concentration e cap enforcement ...")
    conc    = sector_concentration(tickers_ok)
    cap_res = check_sector_cap(tickers_ok, cap=SECTOR_CAP)

    print(f"\n  Distribuzione settori (universe completo):")
    for s, frac in conc.items():
        breach = "  ← BREACH 30% cap" if frac > SECTOR_CAP else ""
        print(f"    {s:20s}  {frac*100:5.1f}%{breach}")

    breached_sectors = [s for s, d in cap_res.items() if d["breaches_cap"]]
    if breached_sectors:
        print(f"\n  Settori che violano il cap ({SECTOR_CAP*100:.0f}%): {', '.join(breached_sectors)}")
        print(f"  Azione attesa dal Risk Manager: rimuovere ticker in eccesso")
    else:
        print(f"\n  Nessun settore viola il cap {SECTOR_CAP*100:.0f}%")

    # ── Salvataggio CSV ───────────────────────────────────────────────────────
    var_path  = os.path.join(RESULT_DIR, "02_risk_var_fix3.csv")
    corr_path = os.path.join(RESULT_DIR, "02_risk_correlation_fix3.csv")
    summ_path = os.path.join(RESULT_DIR, "02_risk_summary_fix3.csv")

    var_df.to_csv(var_path, index=False)

    corr.to_csv(corr_path)

    summary_rows = [
        {"metrica": "Ticker analizzati",              "valore": n},
        {"metrica": "Periodo",                        "valore": f"{returns_full.index[0].date()} - {returns_full.index[-1].date()}"},
        {"metrica": "VaR 95% portfolio (12m)",        "valore": f"{var_full*100:.2f}%"},
        {"metrica": "VaR warning breach (>4%)",       "valore": "SI" if var_full > MAX_PORTFOLIO_VAR else "NO"},
        {"metrica": "Max Drawdown portfolio (12m)",   "valore": f"{mdd_full*100:.2f}%"},
        {"metrica": "DD warning breach (>15%)",       "valore": "SI" if mdd_full > MAX_DRAWDOWN_LIM else "NO"},
        {"metrica": "VaR 95% ultimi 30gg",            "valore": f"{var_30d*100:.2f}%"},
        {"metrica": "Max DD ultimi 30gg",             "valore": f"{mdd_30d*100:.2f}%"},
        {"metrica": "Perdita media worst 20gg",       "valore": f"{var_worst20*100:.2f}%"},
        {"metrica": "Correlazione media",             "valore": f"{avg_corr:.3f}"},
        {"metrica": "Correlazione massima",           "valore": f"{max_pair_v:.3f} ({max_pair[0]}/{max_pair[1]})"},
        {"metrica": "Settori con breach 30% cap",     "valore": ', '.join(breached_sectors) if breached_sectors else "Nessuno"},
        {"metrica": "Ticker piu rischioso (VaR)",     "valore": var_df.iloc[0]["ticker"]},
        {"metrica": "VaR del piu rischioso",          "valore": f"{var_df.iloc[0]['var_95_daily']*100:.2f}%"},
        {"metrica": "Ticker meno rischioso (VaR)",    "valore": var_df.iloc[-1]["ticker"]},
        {"metrica": "VaR del meno rischioso",         "valore": f"{var_df.iloc[-1]['var_95_daily']*100:.2f}%"},
    ]
    pd.DataFrame(summary_rows).to_csv(summ_path, index=False)

    print(f"\n  File salvati:")
    print(f"    {var_path}")
    print(f"    {corr_path}")
    print(f"    {summ_path}")

    # ── Top rischiosi ─────────────────────────────────────────────────────────
    print(f"\n  Top 5 ticker piu rischiosi (VaR 95% giornaliero):")
    for _, r in var_df.head(5).iterrows():
        print(f"    {r['ticker']:6s}  VaR={r['var_95_daily']*100:.2f}%  DD={r['max_drawdown']*100:.2f}%  Vol={r['annual_vol']*100:.1f}%")

    print(f"\n  Top 5 ticker meno rischiosi:")
    for _, r in var_df.tail(5).iterrows():
        print(f"    {r['ticker']:6s}  VaR={r['var_95_daily']*100:.2f}%  DD={r['max_drawdown']*100:.2f}%  Vol={r['annual_vol']*100:.1f}%")

    # ── Fix 3: Single-ticker VaR cap validation ──────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  FIX 3 – Single-ticker VaR cap ({MAX_SINGLE_VAR*100:.0f}%)")
    print(f"{'─'*65}")
    print(f"\n  VaR individuale per tutti i ticker (ordinato per rischio):")
    print(f"  {'Ticker':<8} {'VaR 95%':>9} {'Status':>12} {'Settore'}")
    print(f"  {'-'*55}")

    n_blocked = 0
    blocked_tickers = []
    for _, r in var_df.iterrows():
        var_pct = r['var_95_daily'] * 100
        blocked = r['var_95_daily'] > MAX_SINGLE_VAR
        status  = "BLOCCATO ✗" if blocked else "OK ✓"
        if blocked:
            n_blocked += 1
            blocked_tickers.append(r['ticker'])
        print(f"  {r['ticker']:<8} {var_pct:>8.2f}%  {status:>12}  {r['sector']}")

    print(f"\n  Ticker bloccati dal VaR cap: {n_blocked}/{len(var_df)}")
    if blocked_tickers:
        print(f"  Lista: {blocked_tickers}")
        print(f"  Questi ticker NON entrerebbero in bullish_approved anche se segnalati LONG.")
    else:
        print(f"  Nessun ticker supera la soglia {MAX_SINGLE_VAR*100:.0f}% — cap non attivo oggi.")

    # Scenario: simula impact su portfolio ipotetico (tutti LONG)
    n_all = len(var_df)
    n_approved = n_all - n_blocked
    print(f"\n  Scenario ipotetico (tutti i ticker segnalati LONG):")
    print(f"    Totale segnalati:   {n_all}")
    print(f"    Dopo VaR cap:       {n_approved}  ({n_blocked} rimossi)")
    if n_approved > 0:
        approved_tickers = [r['ticker'] for _, r in var_df.iterrows() if r['var_95_daily'] <= MAX_SINGLE_VAR]
        w_approved = {t: 1.0/n_approved for t in approved_tickers}
        var_after = historical_var(returns_full[approved_tickers] if approved_tickers else returns_full, w_approved)
        var_before_all = historical_var(returns_full, {t: 1.0/n_all for t in tickers_ok})
        print(f"    VaR portfolio prima del cap: {var_before_all*100:.2f}%")
        print(f"    VaR portfolio dopo  il cap:  {var_after*100:.2f}%")
        delta_var = var_after - var_before_all
        arrow = "▲" if delta_var > 0 else "▼"
        direction_str = "aumentato" if delta_var > 0 else "ridotto"
        print(f"    Delta VaR: {arrow}{abs(delta_var)*100:.2f}%  ({direction_str})")

    print(f"\n  Prossimo step: python backtest/03_backtest_devils_advocate.py")
    print("=" * 65)


if __name__ == "__main__":
    run_backtest()
