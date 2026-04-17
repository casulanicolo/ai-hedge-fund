"""
00_download_data.py  —  Athanor Alpha | Backtest F4 + Fix B1
============================================================
Scarica dati OHLCV + metriche fondamentali per 30 ticker.

Fix B1: finestra temporale estesa da Apr 2025 a Jan 2020.
  - Include: COVID crash Mar 2020, bear market 2022, recovery 2023-2024, bull 2025-2026
  - Per ticker senza storia completa da Jan 2020, si usa la prima data disponibile
  - Il CSV backtest/results/00_ticker_history.csv documenta la copertura per ticker

Salva tutto in:  backtest/data/ohlcv_5y.pkl          (era ohlcv_12m.pkl)
                 backtest/data/fundamentals_12m.pkl   (invariato — snapshot statico)
                 backtest/results/00_ticker_history.csv

Esegui UNA VOLTA prima di tutti gli altri script di backtest.
Tempo stimato: 4-8 minuti.
"""

import os
import sys
import pickle
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# ── Configurazione ────────────────────────────────────────────────────────────

TICKERS = [
    # Large-cap tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    # Consumer / industriali
    "AMZN", "TSLA", "HD", "NKE", "WMT",
    # Financials
    "JPM", "V", "MA", "BRK-B", "GS",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK",
    # Energia / Materiali
    "XOM", "CVX", "LIN", "APD", "FCX",
    # Growth / small cap
    "SMCI", "MELI", "COIN", "MSTR", "PLTR",
]

# Fix B1: esteso da Apr 2025 a Jan 2020
# Include COVID crash (Mar 2020), bear market 2022, recovery 2023-2024, bull 2025-2026
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = "2020-01-01"

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "data")
RESULT_DIR   = os.path.join(os.path.dirname(__file__), "results")
OHLCV_PATH   = os.path.join(OUTPUT_DIR, "ohlcv_5y.pkl")       # Fix B1: rinominato da ohlcv_12m.pkl
FUND_PATH    = os.path.join(OUTPUT_DIR, "fundamentals_12m.pkl")
HISTORY_PATH = os.path.join(RESULT_DIR, "00_ticker_history.csv")

os.makedirs(RESULT_DIR, exist_ok=True)

# Fix B1: sotto-periodi per aggregazione risultati backtest
# Usati anche da 05_backtest_operative_agents.py
SUB_PERIODS = {
    "2020-2021": ("2020-01-01", "2021-12-31"),  # bull market post-COVID
    "2022":      ("2022-01-01", "2022-12-31"),  # bear market / rate hike Fed
    "2023-2024": ("2023-01-01", "2024-12-31"),  # recovery
    "2025-2026": ("2025-01-01", "2026-12-31"),  # bull corrente
}


# ── Funzioni ──────────────────────────────────────────────────────────────────

def download_ohlcv(tickers: list, start: str, end: str) -> dict:
    """
    Scarica OHLCV giornaliero per tutti i ticker dal 2020-01-01.
    Per ticker senza dati dalla data richiesta, usa la prima data disponibile
    e lo documenta nel CSV 00_ticker_history.csv.
    """
    result = {}
    history_rows = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:2d}/{total}] Scarico OHLCV: {ticker} ...", end=" ")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                print("VUOTO - saltato")
                history_rows.append({
                    "ticker":           ticker,
                    "requested_start":  start,
                    "actual_start":     None,
                    "actual_end":       None,
                    "n_bars":           0,
                    "coverage_ok":      False,
                    "note":             "nessun dato disponibile",
                })
                continue

            # Flatten colonne MultiIndex se presenti
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

            actual_start = df.index[0].strftime("%Y-%m-%d")
            actual_end   = df.index[-1].strftime("%Y-%m-%d")
            coverage_ok  = df.index[0] <= pd.Timestamp("2020-06-01")  # tolleranza 5 mesi

            note = ""
            if not coverage_ok:
                # Fix B1: ticker senza storia completa — documentato, usato dalla prima data disponibile
                note = f"dati disponibili solo da {actual_start} (richiesto {start})"
                print(f"PARZIALE ({actual_start} → {actual_end}, {len(df)} barre) ⚠")
            else:
                print(f"OK ({actual_start} → {actual_end}, {len(df)} barre)")

            result[ticker] = df
            history_rows.append({
                "ticker":           ticker,
                "requested_start":  start,
                "actual_start":     actual_start,
                "actual_end":       actual_end,
                "n_bars":           len(df),
                "coverage_ok":      coverage_ok,
                "note":             note,
            })

        except Exception as e:
            print(f"ERRORE: {e}")
            history_rows.append({
                "ticker":           ticker,
                "requested_start":  start,
                "actual_start":     None,
                "actual_end":       None,
                "n_bars":           0,
                "coverage_ok":      False,
                "note":             f"errore: {e}",
            })

    # Salva CSV copertura ticker
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(HISTORY_PATH, index=False)
    print(f"\n  Copertura ticker salvata: {HISTORY_PATH}")

    # Report copertura a schermo
    n_ok      = history_df["coverage_ok"].sum()
    n_partial = (~history_df["coverage_ok"] & history_df["actual_start"].notna()).sum()
    n_miss    = history_df["actual_start"].isna().sum()
    print(f"  Copertura Jan 2020: OK={n_ok}  PARZIALE={n_partial}  MANCANTI={n_miss}")
    if n_partial > 0:
        print("  Ticker con storia parziale:")
        for _, row in history_df[~history_df["coverage_ok"] & history_df["actual_start"].notna()].iterrows():
            # Fix B1: per ticker senza storia completa, la prima data disponibile è documentata qui
            print(f"    {row['ticker']:8s}  prima data: {row['actual_start']}  barre: {row['n_bars']}")

    return result


def download_fundamentals(tickers: list) -> dict:
    """Scarica dati fondamentali via yfinance per ogni ticker (snapshot statico)."""
    result = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:2d}/{total}] Scarico fondamentali: {ticker} ...", end=" ")
        try:
            t = yf.Ticker(ticker)
            payload = {
                "info":            t.info or {},
                "income_stmt_q":   t.quarterly_income_stmt,
                "balance_sheet_q": t.quarterly_balance_sheet,
                "cash_flow_q":     t.quarterly_cashflow,
            }
            result[ticker] = payload
            print("OK")
        except Exception as e:
            print(f"ERRORE: {e}")
            result[ticker] = {}
    return result


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Salvato: {path}  ({size_kb:.1f} KB)")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  ATHANOR ALPHA — F4 Data Download  |  Fix B1: finestra 2020-oggi")
    print(f"  Periodo: {START_DATE}  →  {END_DATE}")
    print(f"  Ticker:  {len(TICKERS)}")
    print(f"  Sotto-periodi: {list(SUB_PERIODS.keys())}")
    print("=" * 65)

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    print("\n[1/2] Download OHLCV giornaliero (Fix B1: Jan 2020 → oggi) ...")
    if os.path.exists(OHLCV_PATH):
        print(f"  File già presente: {OHLCV_PATH}")
        ans = input("  Riscarico? (s/N): ").strip().lower()
        if ans != "s":
            print("  Salto download OHLCV.")
            ohlcv_data = load_pickle(OHLCV_PATH)
        else:
            ohlcv_data = download_ohlcv(TICKERS, START_DATE, END_DATE)
            save_pickle(ohlcv_data, OHLCV_PATH)
    else:
        # Fix B1: controlla anche il vecchio file ohlcv_12m.pkl
        old_path = os.path.join(OUTPUT_DIR, "ohlcv_12m.pkl")
        if os.path.exists(old_path):
            print(f"  Trovato vecchio file {old_path} (12 mesi).")
            print(f"  Fix B1 richiede storia da Jan 2020 — riscarico necessario.")
        ohlcv_data = download_ohlcv(TICKERS, START_DATE, END_DATE)
        save_pickle(ohlcv_data, OHLCV_PATH)

    # ── Fundamentals ──────────────────────────────────────────────────────────
    print("\n[2/2] Download fondamentali ...")
    if os.path.exists(FUND_PATH):
        print(f"  File già presente: {FUND_PATH}")
        ans = input("  Riscarico? (s/N): ").strip().lower()
        if ans != "s":
            print("  Salto download fondamentali.")
        else:
            fund_data = download_fundamentals(TICKERS)
            save_pickle(fund_data, FUND_PATH)
    else:
        fund_data = download_fundamentals(TICKERS)
        save_pickle(fund_data, FUND_PATH)

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DOWNLOAD COMPLETATO  —  Fix B1")
    ohlcv_loaded = load_pickle(OHLCV_PATH)
    ticker_ok  = [t for t, df in ohlcv_loaded.items() if isinstance(df, pd.DataFrame) and not df.empty]
    ticker_nok = [t for t in TICKERS if t not in ticker_ok]

    print(f"  Ticker OK:      {len(ticker_ok)} / {len(TICKERS)}")
    if ticker_nok:
        print(f"  Ticker saltati: {', '.join(ticker_nok)}")

    # Verifica copertura sotto-periodi
    print(f"\n  Verifica copertura sotto-periodi:")
    for period_name, (p_start, p_end) in SUB_PERIODS.items():
        p_ts = pd.Timestamp(p_start)
        n_covered = sum(
            1 for df in ohlcv_loaded.values()
            if isinstance(df, pd.DataFrame) and not df.empty and df.index[0] <= p_ts
        )
        print(f"    {period_name:12s}  ticker con dati: {n_covered}/{len(ticker_ok)}")

    print(f"\n  File salvati in: {OUTPUT_DIR}")
    print(f"    ohlcv_5y.pkl         — dati OHLCV Jan 2020 → oggi")
    print(f"    fundamentals_12m.pkl — dati fondamentali (snapshot statico)")
    print(f"    results/00_ticker_history.csv — copertura per ticker")
    print(f"\n  Nota Fix B1: gli script backtest ora usano ohlcv_5y.pkl")
    print(f"  Prossimo step: esegui  01_backtest_fundamentals_fix2.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
