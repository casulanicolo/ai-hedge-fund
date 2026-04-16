"""
06_expected_value_agents.py — Athanor Alpha F4 / B3
Expected Value netto per agente a orizzonte 20d.

Formula:
  EV = win_rate × avg_win_pct − loss_rate × avg_loss_pct

Calcola sia EV lordo (fwd_20d) che EV netto di costi (fwd_20d_net).
I costi round-trip sono già incorporati in fwd_20d_net dal backtest precedente,
quindi EV_netto è direttamente calcolabile su fwd_20d_net.

Input  : backtest/results/05_operative_scores.csv
Output : backtest/results/06_ev_per_agent.csv
         stampa tabella a schermo

Usage  : python backtest/06_expected_value_agents.py
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

# ── Percorsi ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
SCORES_CSV  = os.path.join(RESULTS_DIR, "05_operative_scores.csv")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "06_ev_per_agent.csv")

# ── Ticker ad alta volatilità (costo 0.30% round-trip) ───────────────────────
HIGH_VOL_TICKERS = {"BTC-USD", "COIN", "ETH-USD", "MSTR", "SMCI", "SOL-USD"}
COST_EQUITY  = 0.0010   # 0.10% round-trip
COST_HIGHVOL = 0.0030   # 0.30% round-trip


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_scores(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] File non trovato: {path}")
        print("  Assicurati di aver eseguito prima i backtest 01-05.")
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"agent", "ticker", "direction_filtered", "fwd_20d", "fwd_20d_net"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Colonne mancanti in {path}: {missing}")
        sys.exit(1)
    print(f"[INFO] Caricate {len(df):,} righe da {os.path.basename(path)}")
    return df


def compute_ev_stats(rets: pd.Series) -> dict:
    """
    Dato un array di rendimenti (float), calcola EV e statistiche associate.
    Un trade è vincente se il rendimento è > 0.
    """
    n = len(rets)
    if n == 0:
        return dict(n=0, win_rate=np.nan, avg_win_pct=np.nan,
                    avg_loss_pct=np.nan, ev=np.nan)

    wins   = rets[rets > 0]
    losses = rets[rets <= 0]

    win_rate  = len(wins) / n
    loss_rate = 1.0 - win_rate
    avg_win   = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss  = float(losses.mean()) if len(losses) > 0 else 0.0  # negativo

    # EV algebrico: win_rate * avg_win + loss_rate * avg_loss
    ev = win_rate * avg_win + loss_rate * avg_loss

    return dict(
        n            = n,
        win_rate     = win_rate,
        avg_win_pct  = avg_win  * 100,
        avg_loss_pct = abs(avg_loss) * 100,   # positivo per leggibilità
        ev           = ev * 100,               # in %
    )


# ══════════════════════════════════════════════════════════════════════════════
# Core computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_expected_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni agente calcola EV lordo e EV netto.
    Filtra solo segnali attivi (direction_filtered == LONG o SHORT).
    """
    # direction_filtered usa LONG / SHORT / NEUTRAL
    active = df[df["direction_filtered"].isin(["LONG", "SHORT"])].copy()

    rows = []
    for agent, grp in active.groupby("agent", sort=True):

        # ── EV lordo ──────────────────────────────────────────────────────────
        stats_gross = compute_ev_stats(grp["fwd_20d"].dropna())

        # ── EV netto (costi già incorporati in fwd_20d_net) ───────────────────
        stats_net   = compute_ev_stats(grp["fwd_20d_net"].dropna())

        # ── Costo medio round-trip per questo agente (informativo) ────────────
        n_total = len(grp)
        n_hv    = int(grp["ticker"].isin(HIGH_VOL_TICKERS).sum())
        n_eq    = n_total - n_hv
        avg_cost_pct = round(
            (n_hv * COST_HIGHVOL + n_eq * COST_EQUITY) / n_total * 100, 3
        ) if n_total > 0 else round(COST_EQUITY * 100, 3)

        verdict = "KEEP" if (not np.isnan(stats_net["ev"]) and stats_net["ev"] > 0) else "REMOVE"

        rows.append({
            "agente"          : agent,
            "n_trades"        : stats_gross["n"],
            "win_rate"        : round(stats_gross["win_rate"] * 100, 1),
            "avg_win_pct"     : round(stats_gross["avg_win_pct"], 3),
            "avg_loss_pct"    : round(stats_gross["avg_loss_pct"], 3),
            "EV_lordo"        : round(stats_gross["ev"], 4),
            "avg_cost_rt_pct" : avg_cost_pct,
            "EV_netto_costi"  : round(stats_net["ev"], 4),
            "verdict"         : verdict,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def print_table(result: pd.DataFrame) -> None:
    sorted_df = result.sort_values("EV_netto_costi", ascending=False)

    print()
    print("=" * 100)
    print("  ATHANOR ALPHA — B3: Expected Value per Agente (orizzonte 20d)")
    print("=" * 100)
    header = (
        f"  {'Agente':<25} {'N':>5}  {'WR%':>6}  {'AvgWin%':>8}  {'AvgLoss%':>9}  "
        f"{'EV_lordo%':>10}  {'EV_netto%':>10}  {'Verdict':>10}"
    )
    print(header)
    print("  " + "-" * 96)

    for _, row in sorted_df.iterrows():
        verdict_sym = "✅ KEEP  " if row["verdict"] == "KEEP" else "❌ REMOVE"
        print(
            f"  {row['agente']:<25} "
            f"{int(row['n_trades']):>5}  "
            f"{row['win_rate']:>6.1f}  "
            f"{row['avg_win_pct']:>8.3f}  "
            f"{row['avg_loss_pct']:>9.3f}  "
            f"{row['EV_lordo']:>+10.4f}  "
            f"{row['EV_netto_costi']:>+10.4f}  "
            f"{verdict_sym}"
        )

    print("  " + "-" * 96)
    keep   = (result["verdict"] == "KEEP").sum()
    remove = (result["verdict"] == "REMOVE").sum()
    print(f"\n  KEEP: {keep} agenti   |   REMOVE: {remove} agenti\n")

    positivi = sorted_df[sorted_df["EV_netto_costi"] > 0]["agente"].tolist()
    negativi = sorted_df[sorted_df["EV_netto_costi"] <= 0]["agente"].tolist()
    if positivi:
        print(f"  EV netto > 0  →  {', '.join(positivi)}")
    if negativi:
        print(f"  EV netto ≤ 0  →  {', '.join(negativi)}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("[B3] Expected Value per Agente — Athanor Alpha")
    print(f"     Input : {SCORES_CSV}")
    print(f"     Output: {OUTPUT_CSV}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df     = load_scores(SCORES_CSV)
    result = compute_expected_value(df)

    # Salva CSV
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] CSV salvato: {OUTPUT_CSV}")

    # Stampa tabella
    print_table(result)

    # Info aggiuntive
    active_n = len(df[df["direction_filtered"].isin(["LONG", "SHORT"])])
    print(f"[INFO] Segnali attivi totali: {active_n:,}  "
          f"(periodo: {df['signal_date'].min()} → {df['signal_date'].max()})")
    print(f"[INFO] Costi applicati: equity={COST_EQUITY*100:.2f}% RT, "
          f"high-vol={COST_HIGHVOL*100:.2f}% RT  (già inclusi in fwd_20d_net)")
    print("[B3] Completato.\n")


if __name__ == "__main__":
    main()
