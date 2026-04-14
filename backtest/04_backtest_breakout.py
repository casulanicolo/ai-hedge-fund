"""
04_backtest_breakout.py  -  Athanor Alpha | Backtest F4
========================================================
Backtest ISOLATO del Breakout Momentum Agent (Categoria A).

Testa la logica tecnica SENZA LLM su 12 mesi di dati OHLCV reali:
  1. Volume Surge Ratio      (volume vs media 20gg)
  2. ATR Expansion           (ATR recente vs 20gg fa)
  3. 52-Week High Proximity  (distanza % dal massimo)
  4. Resistance Breakout     (prezzo > max 20gg con volume)
  5. Momentum Score          (ROC 5gg e 10gg)
  6. EMA Trend               (EMA8 vs EMA21)
  7. RSI-14                  (Fix 4: penalita overbought/oversold)

Fix 4 rispetto alla versione precedente:
  - rsi_series(): calcola RSI-14 con Wilder smoothing su tutta la serie
  - compute_breakout_score(): penalita -8 se breakout_up e RSI > 70
  - compute_breakout_score(): penalita -8 se breakdown e RSI < 30
  - Peso breakdown SHORT: da +15 -> +18 (bilancia bias LONG strutturale)
  - Colonna "rsi" aggiunta all'output CSV

Output:
  backtest/results/04_breakout_signals.csv
  backtest/results/04_breakout_forward_returns.csv
  backtest/results/04_breakout_summary.csv

Esegui: python backtest/04_backtest_breakout.py
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

# Parametri (identici a breakout_momentum.py)
VOLUME_SURGE_THRESHOLD  = 2.0
ATR_EXPANSION_THRESHOLD = 1.3
BREAKOUT_LOOKBACK       = 20
HIGH_PROXIMITY_PCT      = 0.05
RSI_OVERBOUGHT          = 70    # Fix 4
RSI_OVERSOLD            = 30    # Fix 4

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


# ── Metriche tecniche ─────────────────────────────────────────────────────────

def volume_surge_ratio(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Volume corrente / media volume ultimi N giorni. Applicato a tutta la serie."""
    vol = df["Volume"].astype(float)
    avg_vol = vol.rolling(window=lookback).mean().shift(1)
    return (vol / avg_vol).round(2)


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR con smoothing EWM (Wilder)."""
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def atr_expansion_series(df: pd.DataFrame, period: int = 14, lookback: int = 20) -> pd.Series:
    """Rapporto ATR corrente / ATR di N giorni fa."""
    atr = atr_series(df, period)
    return (atr / atr.shift(lookback)).round(2)


def high_proximity_series(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Distanza % dal massimo degli ultimi N barre. 0 = AT the high."""
    close    = df["Close"].astype(float)
    roll_max = close.rolling(window=window, min_periods=20).max()
    return ((roll_max - close) / roll_max).round(4)


def resistance_breakout_series(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """1 se close > max degli ultimi N giorni, 0 altrimenti."""
    close    = df["Close"].astype(float)
    prev_max = close.rolling(window=lookback).max().shift(1)
    return (close > prev_max).astype(int)


def resistance_breakdown_series(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """1 se close < min degli ultimi N giorni, 0 altrimenti."""
    close    = df["Close"].astype(float)
    prev_min = close.rolling(window=lookback).min().shift(1)
    return (close < prev_min).astype(int)


def momentum_roc(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Rate of Change su N giorni."""
    close = df["Close"].astype(float)
    return ((close / close.shift(period)) - 1).round(4)


def ema_trend_series(df: pd.DataFrame, fast: int = 8, slow: int = 21) -> pd.Series:
    """1 se EMA fast > EMA slow (uptrend), -1 altrimenti."""
    close = df["Close"].astype(float)
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    return np.where(ema_f > ema_s, 1, -1)


def rsi_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    RSI classico (Wilder smoothing) su tutta la serie daily.
    Fix 4: usato per penalizzare breakout in overbought e breakdown in oversold.
    Returns: Series float in [0, 100], NaN nei primi `period` periodi.
    """
    close = df["Close"].astype(float)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).round(2)


def compute_breakout_score(row) -> float:
    """
    Score composito 0-100 da 7 metriche.

    Fix 4 rispetto alla versione precedente:
      - Peso breakdown SHORT: +15 -> +18 (bilancia bias LONG)
      - Penalita -8 se resistance_breakout=1 e RSI > RSI_OVERBOUGHT (70)
      - Penalita -8 se resistance_breakdown=1 e RSI < RSI_OVERSOLD (30)
      (scala proporzionale: -0.20 su [0,1] ~ -8 su base 50)
    """
    score = 50.0  # neutro di partenza

    # 1. Volume surge (+15 se anomalo)
    vsr = row.get("volume_surge", 1.0)
    if pd.notna(vsr):
        if vsr >= VOLUME_SURGE_THRESHOLD:
            score += 15
        elif vsr < 0.5:
            score -= 10

    # 2. ATR expansion (+10 se espansione)
    atr_exp = row.get("atr_expansion", 1.0)
    if pd.notna(atr_exp):
        if atr_exp >= ATR_EXPANSION_THRESHOLD:
            score += 10
        elif atr_exp < 0.7:
            score -= 5

    # 3. High proximity (vicino al massimo = bullish)
    prox = row.get("high_proximity", 0.5)
    if pd.notna(prox):
        if prox < HIGH_PROXIMITY_PCT:       # entro 5% dal max
            score += 15
        elif prox > 0.20:                   # oltre 20% dal max
            score -= 10

    # 4. Resistance breakout UP (+15)
    brk = row.get("resistance_breakout", 0)
    if pd.notna(brk):
        score += brk * 15

    # 5. Resistance breakdown SHORT (+18, era +15 — Fix 4: bilancia bias LONG)
    brkdn = row.get("resistance_breakdown", 0)
    if pd.notna(brkdn) and brkdn == 1:
        score -= 18   # segnale SHORT: abbassa lo score verso la zona SHORT (<35)

    # 6. Momentum ROC 5gg
    roc5 = row.get("roc_5d", 0.0)
    if pd.notna(roc5):
        score += max(-10, min(10, roc5 * 200))

    # 7. EMA trend (+5 uptrend, -5 downtrend)
    ema = row.get("ema_trend", 0)
    if pd.notna(ema):
        score += ema * 5

    # ── Penalita RSI (Fix 4) ──────────────────────────────────────────────────
    rsi_val = row.get("rsi", 50.0)
    if pd.notna(rsi_val):
        # Breakout UP in overbought: rischio pullback immediato
        if brk == 1 and rsi_val > RSI_OVERBOUGHT:
            score -= 8
        # Breakdown in oversold estremo: rischio rimbalzo tecnico
        if brkdn == 1 and rsi_val < RSI_OVERSOLD:
            score += 8   # riporta verso neutral (non favorire SHORT estremo)

    return round(max(0, min(100, score)), 1)


def direction_from_score(score: float) -> str:
    if score >= 65:
        return "LONG"
    elif score <= 35:
        return "SHORT"
    return "NEUTRAL"


# ── Backtest per ticker ───────────────────────────────────────────────────────

def backtest_ticker(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola metriche giorno per giorno e forward returns.
    Ritorna DataFrame con una riga per ogni giorno.
    """
    df = df.copy()
    df["volume_surge"]          = volume_surge_ratio(df)
    df["atr_expansion"]         = atr_expansion_series(df)
    df["high_proximity"]        = high_proximity_series(df)
    df["resistance_breakout"]   = resistance_breakout_series(df)
    df["resistance_breakdown"]  = resistance_breakdown_series(df)   # Fix 4
    df["roc_5d"]                = momentum_roc(df, 5)
    df["roc_10d"]               = momentum_roc(df, 10)
    df["ema_trend"]             = ema_trend_series(df)
    df["rsi"]                   = rsi_series(df)                    # Fix 4

    rows = []
    for i, (date, row) in enumerate(df.iterrows()):
        score     = compute_breakout_score(row)
        direction = direction_from_score(score)

        close_now = row["Close"]
        fwd_3d  = (df["Close"].iloc[i + 3]  / close_now - 1) if i + 3  < len(df) else np.nan
        fwd_5d  = (df["Close"].iloc[i + 5]  / close_now - 1) if i + 5  < len(df) else np.nan
        fwd_10d = (df["Close"].iloc[i + 10] / close_now - 1) if i + 10 < len(df) else np.nan

        rows.append({
            "date":               date,
            "ticker":             ticker,
            "sector":             SECTOR_MAP.get(ticker, "Unknown"),
            "close":              round(float(close_now), 2),
            "score":              score,
            "direction":          direction,
            "volume_surge":       round(float(row["volume_surge"]), 2)       if pd.notna(row["volume_surge"])        else None,
            "atr_expansion":      round(float(row["atr_expansion"]), 2)      if pd.notna(row["atr_expansion"])       else None,
            "high_proximity_pct": round(float(row["high_proximity"]) * 100, 1) if pd.notna(row["high_proximity"])   else None,
            "resistance_breakout":int(row["resistance_breakout"])             if pd.notna(row["resistance_breakout"]) else 0,
            "resistance_breakdown":int(row["resistance_breakdown"])           if pd.notna(row["resistance_breakdown"]) else 0,
            "rsi":                round(float(row["rsi"]), 1)                if pd.notna(row["rsi"])                 else None,
            "roc_5d_pct":         round(float(row["roc_5d"]) * 100, 2)      if pd.notna(row["roc_5d"])              else None,
            "ema_trend":          int(row["ema_trend"])                       if pd.notna(row["ema_trend"])           else 0,
            "fwd_return_3d":      round(float(fwd_3d), 4)  if pd.notna(fwd_3d)  else None,
            "fwd_return_5d":      round(float(fwd_5d), 4)  if pd.notna(fwd_5d)  else None,
            "fwd_return_10d":     round(float(fwd_10d), 4) if pd.notna(fwd_10d) else None,
        })

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 65)
    print("  ATHANOR ALPHA - F4  |  Backtest: Breakout Momentum Agent")
    print(f"  Fix 4: RSI overbought penalty + SHORT weight increase")
    print(f"  Eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    if not os.path.exists(OHLCV_PATH):
        print(f"\n  ERRORE: {OHLCV_PATH} non trovato.")
        sys.exit(1)

    print("\n[1/4] Carico dati OHLCV ...")
    with open(OHLCV_PATH, "rb") as f:
        ohlcv_data = pickle.load(f)
    print(f"  Ticker: {len(ohlcv_data)}")

    print("\n[2/4] Calcolo metriche giornaliere per ogni ticker ...")
    all_dfs = []
    for ticker, df in ohlcv_data.items():
        print(f"  {ticker:6s} ...", end=" ")
        if df is None or len(df) < 30:
            print("SALTATO (dati insufficienti)")
            continue
        ticker_df = backtest_ticker(ticker, df)
        all_dfs.append(ticker_df)
        n_long  = (ticker_df["direction"] == "LONG").sum()
        n_short = (ticker_df["direction"] == "SHORT").sum()
        print(f"LONG={n_long}  SHORT={n_short}  NEUTRAL={len(ticker_df)-n_long-n_short}")

    full_df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n[3/4] Calcolo win rate e forward returns ...")

    sig_df = full_df[full_df["direction"].isin(["LONG", "SHORT"])].copy()

    for h in ["3d", "5d", "10d"]:
        col     = f"fwd_return_{h}"
        win_col = f"win_{h}"
        sig_df[win_col] = np.where(
            sig_df["direction"] == "LONG",
            sig_df[col] > 0,
            sig_df[col] < 0,
        )

    win_3d  = sig_df["win_3d"].mean()  * 100
    win_5d  = sig_df["win_5d"].mean()  * 100
    win_10d = sig_df["win_10d"].mean() * 100

    avg_ret_long_5d  = sig_df[sig_df["direction"] == "LONG"]["fwd_return_5d"].mean()  * 100
    avg_ret_short_5d = sig_df[sig_df["direction"] == "SHORT"]["fwd_return_5d"].mean() * 100

    # ── Analisi RSI split (Fix 4) ─────────────────────────────────────────────
    long_df   = sig_df[sig_df["direction"] == "LONG"].copy()
    short_df  = sig_df[sig_df["direction"] == "SHORT"].copy()

    long_overbought  = long_df[long_df["rsi"] > RSI_OVERBOUGHT]
    long_normal      = long_df[long_df["rsi"] <= RSI_OVERBOUGHT]
    short_oversold   = short_df[short_df["rsi"] < RSI_OVERSOLD]
    short_normal     = short_df[short_df["rsi"] >= RSI_OVERSOLD]

    wr_long_ob   = (long_overbought["win_5d"].mean()  * 100) if len(long_overbought)  > 0 else float("nan")
    wr_long_norm = (long_normal["win_5d"].mean()       * 100) if len(long_normal)      > 0 else float("nan")
    wr_short_os  = (short_oversold["win_5d"].mean()   * 100) if len(short_oversold)   > 0 else float("nan")
    wr_short_norm= (short_normal["win_5d"].mean()     * 100) if len(short_normal)     > 0 else float("nan")

    print(f"\n  Win rate segnali (LONG + SHORT combinati):")
    print(f"    a 3gg  : {win_3d:.1f}%")
    print(f"    a 5gg  : {win_5d:.1f}%")
    print(f"    a 10gg : {win_10d:.1f}%")

    print(f"\n  Return medio a 5gg:")
    print(f"    LONG signals : {avg_ret_long_5d:+.2f}%")
    print(f"    SHORT signals: {avg_ret_short_5d:+.2f}%")

    print(f"\n  [Fix 4] Analisi RSI split (win rate 5gg):")
    print(f"    LONG  con RSI > {RSI_OVERBOUGHT} (overbought): {wr_long_ob:.1f}%  (n={len(long_overbought)})")
    print(f"    LONG  con RSI <= {RSI_OVERBOUGHT} (normale)  : {wr_long_norm:.1f}%  (n={len(long_normal)})")
    print(f"    SHORT con RSI < {RSI_OVERSOLD}  (oversold) : {wr_short_os:.1f}%  (n={len(short_oversold)})")
    print(f"    SHORT con RSI >= {RSI_OVERSOLD}  (normale)  : {wr_short_norm:.1f}%  (n={len(short_normal)})")
    print(f"    -> Delta LONG overbought vs normale: {wr_long_ob - wr_long_norm:+.1f}pp")

    # Win rate per ticker
    print(f"\n  Win rate a 5gg per ticker (top e bottom 5):")
    wr_by_ticker = sig_df.groupby("ticker")["win_5d"].mean() * 100
    wr_by_ticker = wr_by_ticker.sort_values(ascending=False)
    print("  Top 5:")
    for t, wr in wr_by_ticker.head(5).items():
        print(f"    {t:6s}  {wr:.1f}%")
    print("  Bottom 5:")
    for t, wr in wr_by_ticker.tail(5).items():
        print(f"    {t:6s}  {wr:.1f}%")

    # Resistenza breakout confermati
    brk_days  = full_df[full_df["resistance_breakout"] == 1]
    brk_long  = brk_days[brk_days["direction"] == "LONG"]
    brk_wr    = (brk_long["fwd_return_5d"] > 0).mean() * 100 if len(brk_long) > 0 else 0

    # Breakout UP filtrati per RSI (quanti segnali soppressi dalla penalita)
    brk_overbought = brk_days[brk_days["rsi"] > RSI_OVERBOUGHT]
    brk_normal     = brk_days[(brk_days["rsi"] <= RSI_OVERBOUGHT) & (brk_days["direction"] == "LONG")]
    wr_brk_ob   = (brk_overbought["fwd_return_5d"] > 0).mean() * 100 if len(brk_overbought) > 0 else 0
    wr_brk_norm = (brk_normal["fwd_return_5d"] > 0).mean()     * 100 if len(brk_normal)     > 0 else 0

    print(f"\n  Resistance breakout confermati (score LONG + brk=1):")
    print(f"    Totale occorrenze breakout: {len(brk_days)}")
    print(f"    Di cui RSI > {RSI_OVERBOUGHT} (penalizzati Fix 4): {len(brk_overbought)}  WR 5gg: {wr_brk_ob:.1f}%")
    print(f"    RSI <= {RSI_OVERBOUGHT} (segnali normali)          : {len(brk_normal)}  WR 5gg: {wr_brk_norm:.1f}%")
    print(f"    Win rate breakout LONG complessivo: {brk_wr:.1f}%")

    print(f"\n[4/4] Salvataggio CSV ...")
    signals_path = os.path.join(RESULT_DIR, "04_breakout_signals.csv")
    fwd_path     = os.path.join(RESULT_DIR, "04_breakout_forward_returns.csv")
    summ_path    = os.path.join(RESULT_DIR, "04_breakout_summary.csv")

    full_df.to_csv(signals_path, index=False, float_format="%.4f")
    sig_df[["ticker", "date", "direction", "score", "rsi",
            "fwd_return_3d", "fwd_return_5d", "fwd_return_10d",
            "win_3d", "win_5d", "win_10d"]].to_csv(fwd_path, index=False)

    summary_rows = [
        {"metrica": "Ticker analizzati",                    "valore": full_df["ticker"].nunique()},
        {"metrica": "Giorni totali analizzati",             "valore": len(full_df)},
        {"metrica": "Segnali LONG totali",                  "valore": (full_df["direction"] == "LONG").sum()},
        {"metrica": "Segnali SHORT totali",                 "valore": (full_df["direction"] == "SHORT").sum()},
        {"metrica": "Segnali NEUTRAL totali",               "valore": (full_df["direction"] == "NEUTRAL").sum()},
        {"metrica": "Win rate a 3gg",                       "valore": f"{win_3d:.1f}%"},
        {"metrica": "Win rate a 5gg",                       "valore": f"{win_5d:.1f}%"},
        {"metrica": "Win rate a 10gg",                      "valore": f"{win_10d:.1f}%"},
        {"metrica": "Avg return LONG a 5gg",                "valore": f"{avg_ret_long_5d:+.2f}%"},
        {"metrica": "Avg return SHORT a 5gg",               "valore": f"{avg_ret_short_5d:+.2f}%"},
        {"metrica": "[Fix4] WR LONG overbought (RSI>70)",   "valore": f"{wr_long_ob:.1f}%"},
        {"metrica": "[Fix4] WR LONG normale (RSI<=70)",     "valore": f"{wr_long_norm:.1f}%"},
        {"metrica": "[Fix4] WR SHORT oversold (RSI<30)",    "valore": f"{wr_short_os:.1f}%"},
        {"metrica": "[Fix4] WR SHORT normale (RSI>=30)",    "valore": f"{wr_short_norm:.1f}%"},
        {"metrica": "[Fix4] Delta LONG ob vs normale",      "valore": f"{wr_long_ob - wr_long_norm:+.1f}pp"},
        {"metrica": "Breakout confermati (LONG)",           "valore": len(brk_long)},
        {"metrica": "Win rate breakout confermati",         "valore": f"{brk_wr:.1f}%"},
        {"metrica": "Miglior ticker (win rate 5d)",         "valore": wr_by_ticker.index[0]},
        {"metrica": "Peggior ticker (win rate 5d)",         "valore": wr_by_ticker.index[-1]},
    ]
    pd.DataFrame(summary_rows).to_csv(summ_path, index=False)

    print(f"  {signals_path}")
    print(f"  {fwd_path}")
    print(f"  {summ_path}")
    print(f"\n  Prossimo step: python backtest/05_backtest_operative_agents.py")
    print("=" * 65)


if __name__ == "__main__":
    run_backtest()
