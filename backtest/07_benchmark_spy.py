"""
07_benchmark_spy.py — Athanor Alpha F4 / B4
Benchmark SPY e metriche di rischio per agente.

Per ogni agente costruisce una equity curve giornaliera aggregando i segnali
attivi (LONG/SHORT) con rendimento 20d normalizzato a giornaliero (÷20),
poi calcola: Sharpe, Sortino, Max Drawdown, Calmar ratio.
Confronta ogni metrica con SPY buy-and-hold nello stesso periodo.

Input  : backtest/results/05_operative_scores.csv
Output : backtest/results/07_metrics_vs_spy.csv
         backtest/results/07_equity_curve_vs_spy.png

Usage  : python backtest/07_benchmark_spy.py
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Percorsi ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
SCORES_CSV   = os.path.join(RESULTS_DIR, "05_operative_scores.csv")
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "07_metrics_vs_spy.csv")
OUTPUT_CHART = os.path.join(RESULTS_DIR, "07_equity_curve_vs_spy.png")

# ── Parametri ─────────────────────────────────────────────────────────────────
RISK_FREE_ANNUAL = 0.045        # 4.5% annuo
TRADING_DAYS     = 252
RF_DAILY         = (1 + RISK_FREE_ANNUAL) ** (1 / TRADING_DAYS) - 1

# Colori per il grafico
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
]
SPY_COLOR = "#000000"


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_scores(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] File non trovato: {path}")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["signal_date"])
    print(f"[INFO] Caricate {len(df):,} righe da {os.path.basename(path)}")
    return df


def download_spy(start: str, end: str) -> pd.Series:
    """Scarica SPY e restituisce serie di rendimenti giornalieri."""
    print(f"[INFO] Scarico SPY da yfinance ({start} → {end}) ...")
    ticker = yf.Ticker("SPY")
    raw = ticker.history(start=start, end=end, auto_adjust=True)
    if raw.empty:
        print("[ERROR] Impossibile scaricare SPY da yfinance.")
        sys.exit(1)
    closes = raw["Close"].sort_index()
    rets   = closes.pct_change().dropna()
    rets.index = rets.index.tz_localize(None)
    print(f"[INFO] SPY: {len(rets)} giorni di trading scaricati")
    return rets


# ══════════════════════════════════════════════════════════════════════════════
# Equity curve builder
# ══════════════════════════════════════════════════════════════════════════════

def build_agent_daily_rets(df: pd.DataFrame, agent: str) -> pd.Series:
    """
    Costruisce una serie di rendimenti giornalieri per un agente.

    Ogni segnale su (date, ticker) genera un rendimento 20d netto.
    Lo approssimazione a giornaliero è: ret_daily = fwd_20d_net / 20
    (semplicità — alternativa: distribuire su 20 giorni).
    I segnali sullo stesso giorno vengono mediati (portafoglio equi-pesato).
    """
    active = df[
        (df["agent"] == agent) &
        (df["direction_filtered"].isin(["LONG", "SHORT"]))
    ].copy()

    if active.empty:
        return pd.Series(dtype=float)

    # Rendimento giornaliero approssimato per segnale
    active["daily_ret"] = active["fwd_20d_net"] / 20.0

    # Media dei segnali attivi per data (portafoglio equi-pesato giornaliero)
    daily = active.groupby("signal_date")["daily_ret"].mean()
    daily = daily.sort_index()
    return daily


def build_portfolio_daily_rets(df: pd.DataFrame) -> pd.Series:
    """
    Equity curve del portafoglio aggregato: media di tutti gli agenti per data.
    """
    active = df[df["direction_filtered"].isin(["LONG", "SHORT"])].copy()
    active["daily_ret"] = active["fwd_20d_net"] / 20.0
    daily = active.groupby("signal_date")["daily_ret"].mean()
    return daily.sort_index()


def equity_curve(daily_rets: pd.Series, start_value: float = 100.0) -> pd.Series:
    """Da rendimenti giornalieri a equity curve cumulativa."""
    return (1 + daily_rets).cumprod() * start_value


# ══════════════════════════════════════════════════════════════════════════════
# Risk metrics
# ══════════════════════════════════════════════════════════════════════════════

def sharpe(daily_rets: pd.Series) -> float:
    excess = daily_rets - RF_DAILY
    if excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino(daily_rets: pd.Series) -> float:
    excess    = daily_rets - RF_DAILY
    downside  = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(daily_rets: pd.Series) -> float:
    """Ritorna max drawdown come valore negativo (es. -0.25 = -25%)."""
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    return float(dd.min())


def calmar(daily_rets: pd.Series) -> float:
    ann_ret = float((1 + daily_rets.mean()) ** TRADING_DAYS - 1)
    mdd     = max_drawdown(daily_rets)
    if mdd == 0:
        return np.nan
    return ann_ret / abs(mdd)


def annual_return(daily_rets: pd.Series) -> float:
    return float((1 + daily_rets.mean()) ** TRADING_DAYS - 1)


def compute_metrics(daily_rets: pd.Series, label: str) -> dict:
    if daily_rets.empty:
        return {
            "agente": label, "n_days": 0,
            "ann_return_pct": np.nan, "sharpe": np.nan,
            "sortino": np.nan, "max_drawdown_pct": np.nan, "calmar": np.nan,
        }
    return {
        "agente"           : label,
        "n_days"           : len(daily_rets),
        "ann_return_pct"   : round(annual_return(daily_rets) * 100, 2),
        "sharpe"           : round(sharpe(daily_rets), 3),
        "sortino"          : round(sortino(daily_rets), 3),
        "max_drawdown_pct" : round(max_drawdown(daily_rets) * 100, 2),
        "calmar"           : round(calmar(daily_rets), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_equity_curves(
    agent_curves: dict[str, pd.Series],
    portfolio_curve: pd.Series,
    spy_rets: pd.Series,
    output_path: str,
) -> None:
    # Allinea SPY al comune date index
    all_dates = portfolio_curve.index
    spy_aligned = spy_rets.reindex(all_dates).fillna(0.0)
    spy_curve   = equity_curve(spy_aligned)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0d1b2a",
    )
    fig.suptitle(
        "Athanor Alpha — Equity Curve vs SPY (orizzonte 20d, netto costi)",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Pannello superiore: equity curves per agente ──────────────────────────
    ax1.set_facecolor("#0d1b2a")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#334")

    for i, (agent, curve) in enumerate(agent_curves.items()):
        ax1.plot(
            curve.index, curve.values,
            color=COLORS[i % len(COLORS)],
            linewidth=0.8, alpha=0.55, label=agent,
        )

    # Portfolio aggregato in evidenza
    ax1.plot(
        portfolio_curve.index, portfolio_curve.values,
        color="#f5a623", linewidth=2.5, alpha=0.95,
        label="Portfolio (aggregato)", zorder=5,
    )
    # SPY
    ax1.plot(
        spy_curve.index, spy_curve.values,
        color=SPY_COLOR, linewidth=2.0, alpha=0.9,
        linestyle="--", label="SPY B&H", zorder=5,
    )

    ax1.axhline(100, color="#556", linewidth=0.6, linestyle=":")
    ax1.set_ylabel("Equity (base 100)", color="white", fontsize=10)
    ax1.yaxis.label.set_color("white")
    ax1.legend(
        loc="upper left", fontsize=7, ncol=3,
        facecolor="#1a2a3a", labelcolor="white", edgecolor="#334",
        framealpha=0.8,
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.get_xticklabels(), color="white", fontsize=8)
    plt.setp(ax1.get_yticklabels(), color="white", fontsize=8)

    # ── Pannello inferiore: drawdown portafoglio vs SPY ───────────────────────
    ax2.set_facecolor("#0d1b2a")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#334")

    # Drawdown portfolio
    port_dd = (portfolio_curve / portfolio_curve.cummax() - 1) * 100
    spy_dd  = (spy_curve / spy_curve.cummax() - 1) * 100

    ax2.fill_between(port_dd.index, port_dd.values, 0,
                     color="#f5a623", alpha=0.35, label="Portfolio DD")
    ax2.plot(port_dd.index, port_dd.values, color="#f5a623", linewidth=1.0)
    ax2.fill_between(spy_dd.index, spy_dd.values, 0,
                     color="#888", alpha=0.20, label="SPY DD")
    ax2.plot(spy_dd.index, spy_dd.values, color="#888",
             linewidth=1.0, linestyle="--")

    ax2.set_ylabel("Drawdown %", color="white", fontsize=9)
    ax2.legend(fontsize=7, facecolor="#1a2a3a", labelcolor="white",
               edgecolor="#334", loc="lower left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.get_xticklabels(), color="white", fontsize=8)
    plt.setp(ax2.get_yticklabels(), color="white", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1b2a")
    plt.close()
    print(f"[INFO] Grafico salvato: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Print table
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics_table(result: pd.DataFrame) -> None:
    print()
    print("=" * 90)
    print("  ATHANOR ALPHA — B4: Metriche vs SPY Benchmark")
    print("=" * 90)
    header = (
        f"  {'Agente':<25} {'AnnRet%':>8}  {'Sharpe':>7}  {'Sortino':>8}  "
        f"{'MaxDD%':>7}  {'Calmar':>7}"
    )
    print(header)
    print("  " + "-" * 85)

    # SPY prima come benchmark
    spy_row = result[result["agente"] == "SPY"].iloc[0]
    print(
        f"  {'► SPY (benchmark)':<25} "
        f"{spy_row['ann_return_pct']:>8.2f}  "
        f"{spy_row['sharpe']:>7.3f}  "
        f"{spy_row['sortino']:>8.3f}  "
        f"{spy_row['max_drawdown_pct']:>7.2f}  "
        f"{spy_row['calmar']:>7.3f}"
    )
    print("  " + "-" * 85)

    # Poi portafoglio aggregato
    port_row = result[result["agente"] == "Portfolio_Aggregato"].iloc[0]
    sharpe_vs = "✅" if port_row["sharpe"] > spy_row["sharpe"] else "❌"
    print(
        f"  {'★ Portfolio Aggregato':<25} "
        f"{port_row['ann_return_pct']:>8.2f}  "
        f"{port_row['sharpe']:>7.3f}{sharpe_vs} "
        f"{port_row['sortino']:>8.3f}  "
        f"{port_row['max_drawdown_pct']:>7.2f}  "
        f"{port_row['calmar']:>7.3f}"
    )
    print("  " + "-" * 85)

    # Agenti individuali
    agents_df = result[~result["agente"].isin(["SPY", "Portfolio_Aggregato"])]
    agents_df = agents_df.sort_values("sharpe", ascending=False)
    for _, row in agents_df.iterrows():
        sharpe_sym = "✅" if row["sharpe"] > spy_row["sharpe"] else "❌"
        print(
            f"  {row['agente']:<25} "
            f"{row['ann_return_pct']:>8.2f}  "
            f"{row['sharpe']:>7.3f}{sharpe_sym} "
            f"{row['sortino']:>8.3f}  "
            f"{row['max_drawdown_pct']:>7.2f}  "
            f"{row['calmar']:>7.3f}"
        )

    print("  " + "-" * 85)
    n_beat = (agents_df["sharpe"] > spy_row["sharpe"]).sum()
    print(f"\n  Agenti con Sharpe > SPY: {n_beat}/{len(agents_df)}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("[B4] Benchmark SPY + Metriche di Rischio — Athanor Alpha")
    print(f"     Input : {SCORES_CSV}")
    print(f"     Output: {OUTPUT_CSV}")
    print(f"            {OUTPUT_CHART}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Carica scores ─────────────────────────────────────────────────────────
    df = load_scores(SCORES_CSV)
    start_date = df["signal_date"].min().strftime("%Y-%m-%d")
    end_date   = df["signal_date"].max().strftime("%Y-%m-%d")
    print(f"[INFO] Periodo backtest: {start_date} → {end_date}")

    # ── Scarica SPY ───────────────────────────────────────────────────────────
    spy_rets = download_spy(start_date, end_date)

    # ── Equity curves agenti ──────────────────────────────────────────────────
    agents = sorted(df["agent"].unique())
    agent_curves = {}
    rows = []

    for agent in agents:
        daily = build_agent_daily_rets(df, agent)
        if daily.empty:
            continue
        curve = equity_curve(daily)
        agent_curves[agent] = curve
        rows.append(compute_metrics(daily, agent))

    # ── Portfolio aggregato ───────────────────────────────────────────────────
    port_daily = build_portfolio_daily_rets(df)
    port_curve = equity_curve(port_daily)
    rows.append(compute_metrics(port_daily, "Portfolio_Aggregato"))

    # ── SPY metriche ──────────────────────────────────────────────────────────
    # Allinea SPY al periodo dei segnali
    common_start = port_daily.index.min()
    common_end   = port_daily.index.max()
    spy_aligned  = spy_rets.reindex(
        pd.date_range(common_start, common_end, freq="B")
    ).dropna()
    rows.append(compute_metrics(spy_aligned, "SPY"))

    # ── Salva CSV ─────────────────────────────────────────────────────────────
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] CSV salvato: {OUTPUT_CSV}")

    # ── Stampa tabella ────────────────────────────────────────────────────────
    print_metrics_table(result)

    # ── Grafico ───────────────────────────────────────────────────────────────
    plot_equity_curves(agent_curves, port_curve, spy_rets, OUTPUT_CHART)

    print("[B4] Completato.\n")


if __name__ == "__main__":
    main()
