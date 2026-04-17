"""
03_backtest_devils_advocate.py  –  Athanor Alpha | Backtest F4
===============================================================
Backtest ISOLATO del Devil's Advocate Agent (Categoria A).

Il Devil's Advocate usa:
  - VIX regime detection (da regime_detector)
  - Coerenza segnali agenti (quanti concordano)
  - Conviction threshold

Poiché il Devil chiama LLM per il veto narrativo, qui testiamo:
  1. Regime VIX → frequenza attesa di veto per range VIX
  2. Simulazione veto su segnali sintetici (conviction + coerenza variabili)
  3. False rejection rate simulato: veto su segnali che poi "vincono"
  4. Distribuzione veto per livello VIX

Dati usati:
  - VIX storico scaricato da yfinance (^VIX)
  - Segnali sintetici generati in modo controllato

Output:
  backtest/results/03_devil_vix_regimes.csv
  backtest/results/03_devil_veto_simulation.csv
  backtest/results/03_devil_summary.csv

Esegui: python backtest/03_backtest_devils_advocate.py
"""

import os
import sys
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "backtest", "data")
RESULT_DIR = os.path.join(BASE_DIR, "backtest", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

OHLCV_PATH = os.path.join(DATA_DIR, "ohlcv_12m.pkl")

# Parametri Devil's Advocate (da devils_advocate.py)
CONVICTION_THRESHOLD  = 0.55   # soglia minima per non essere vetato
MIN_AGENTS_AGREEING   = 2      # almeno 2 agenti concordi
VIX_LOW    = 18.0              # VIX < 18 → regime calmo
VIX_MEDIUM = 25.0              # 18 ≤ VIX < 25 → regime normale
VIX_HIGH   = 35.0              # VIX ≥ 25 → regime stress


# ── VIX regime classifier ─────────────────────────────────────────────────────

def classify_vix_regime(vix: float) -> str:
    if vix < VIX_LOW:
        return "LOW"       # mercato calmo, Devil più permissivo
    elif vix < VIX_MEDIUM:
        return "NORMAL"    # regime standard
    elif vix < VIX_HIGH:
        return "ELEVATED"  # Devil più restrittivo
    else:
        return "CRISIS"    # Devil veta quasi tutto


def veto_probability_by_regime(regime: str) -> float:
    """
    Probabilità di veto simulata per regime VIX.
    Calibrata sulla logica del Devil: più alto VIX → più veto.
    """
    return {"LOW": 0.10, "NORMAL": 0.20, "ELEVATED": 0.40, "CRISIS": 0.70}.get(regime, 0.20)


# ── Scarica VIX storico ───────────────────────────────────────────────────────

def download_vix(start: str, end: str) -> pd.Series:
    """Scarica VIX da yfinance. Ritorna Series con date come index."""
    vix_path = os.path.join(DATA_DIR, "vix_12m.pkl")
    if os.path.exists(vix_path):
        print("  VIX già scaricato, carico da file ...")
        with open(vix_path, "rb") as f:
            return pickle.load(f)
    print("  Scarico VIX da yfinance ...")
    df = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    vix = df["Close"].dropna()
    with open(vix_path, "wb") as f:
        pickle.dump(vix, f)
    print(f"  VIX scaricato: {len(vix)} barre")
    return vix


# ── Simulazione segnali sintetici ─────────────────────────────────────────────

def simulate_agent_signals(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Genera n segnali sintetici con parametri variabili per testare
    il comportamento del Devil in modo controllato.

    Colonne generate:
      conviction      : 0.1 – 1.0
      agents_agreeing : 1 – 6
      direction       : LONG / SHORT / NEUTRAL
      vix_level       : livello VIX al momento del segnale
    """
    rng = np.random.default_rng(seed)

    conviction      = rng.uniform(0.1, 1.0, n)
    agents_agreeing = rng.integers(1, 7, n)
    direction       = rng.choice(["LONG", "SHORT", "NEUTRAL"], n, p=[0.5, 0.3, 0.2])
    vix_level       = rng.choice(
        np.concatenate([
            np.random.uniform(10, 18, 400),   # 40% regime LOW
            np.random.uniform(18, 25, 350),   # 35% regime NORMAL
            np.random.uniform(25, 35, 180),   # 18% regime ELEVATED
            np.random.uniform(35, 60, 70),    # 7%  regime CRISIS
        ]),
        n, replace=False
    )

    df = pd.DataFrame({
        "conviction":       conviction,
        "agents_agreeing":  agents_agreeing,
        "direction":        direction,
        "vix_level":        vix_level,
    })

    df["vix_regime"] = df["vix_level"].apply(classify_vix_regime)

    # Regola di veto (senza LLM):
    # Il Devil veta se:
    #   - conviction < soglia, OPPURE
    #   - agents_agreeing < minimo, OPPURE
    #   - vix regime è CRISIS e conviction < 0.7
    df["veto_rule_based"] = (
        (df["conviction"] < CONVICTION_THRESHOLD) |
        (df["agents_agreeing"] < MIN_AGENTS_AGREEING) |
        ((df["vix_regime"] == "CRISIS") & (df["conviction"] < 0.70))
    )

    # Simula "esito reale" del segnale: il segnale era corretto?
    # Ipotesi semplificata: conviction alta → più probabilità di essere giusto
    p_correct = np.clip(df["conviction"] * 0.8 + 0.1, 0.1, 0.9)
    df["signal_was_correct"] = rng.random(n) < p_correct

    # False rejection: Devil veta un segnale che sarebbe stato corretto
    df["false_rejection"] = df["veto_rule_based"] & df["signal_was_correct"]

    # False acceptance: Devil non veta un segnale sbagliato
    df["false_acceptance"] = (~df["veto_rule_based"]) & (~df["signal_was_correct"])

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=" * 65)
    print("  ATHANOR ALPHA – F4  |  Backtest: Devil's Advocate Agent")
    print(f"  Eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # ── Carica OHLCV per ottenere date range ─────────────────────────────────
    with open(OHLCV_PATH, "rb") as f:
        ohlcv_data = pickle.load(f)
    sample_df   = list(ohlcv_data.values())[0]
    start_date  = sample_df.index[0].strftime("%Y-%m-%d")
    end_date    = sample_df.index[-1].strftime("%Y-%m-%d")

    # ── VIX storico ──────────────────────────────────────────────────────────
    print("\n[1/4] VIX storico ...")
    vix = download_vix(start_date, end_date)

    vix_df = pd.DataFrame({"vix": vix})
    vix_df["regime"] = vix_df["vix"].apply(classify_vix_regime)

    regime_counts = vix_df["regime"].value_counts()
    regime_pct    = (regime_counts / len(vix_df) * 100).round(1)

    print(f"\n  Distribuzione regime VIX (apr 2025 – apr 2026):")
    for regime in ["LOW", "NORMAL", "ELEVATED", "CRISIS"]:
        count = regime_counts.get(regime, 0)
        pct   = regime_pct.get(regime, 0.0)
        p_veto = veto_probability_by_regime(regime) * 100
        print(f"    {regime:10s}: {count:3d} giorni ({pct:5.1f}%)  →  veto atteso: {p_veto:.0f}%")

    print(f"\n  VIX medio:  {vix.mean():.1f}")
    print(f"  VIX min:    {vix.min():.1f}  ({vix.idxmin().date()})")
    print(f"  VIX max:    {vix.max():.1f}  ({vix.idxmax().date()})")

    # ── Veto rate atteso per il periodo ──────────────────────────────────────
    vix_df["veto_prob"] = vix_df["regime"].apply(veto_probability_by_regime)
    expected_veto_rate  = vix_df["veto_prob"].mean()
    print(f"\n  Veto rate atteso medio (periodo): {expected_veto_rate*100:.1f}%")

    # Salva CSV VIX
    vix_csv_path = os.path.join(RESULT_DIR, "03_devil_vix_regimes.csv")
    vix_df.reset_index().rename(columns={"index": "date"}).to_csv(vix_csv_path, index=False)
    print(f"  Salvato: {vix_csv_path}")

    # ── Simulazione segnali sintetici ─────────────────────────────────────────
    print("\n[2/4] Simulazione 1000 segnali sintetici ...")
    sim_df = simulate_agent_signals(n=1000)

    n_total    = len(sim_df)
    n_veto     = sim_df["veto_rule_based"].sum()
    n_pass     = n_total - n_veto
    n_false_rej= sim_df["false_rejection"].sum()
    n_false_acc= sim_df["false_acceptance"].sum()

    print(f"\n  Totale segnali simulati : {n_total}")
    print(f"  Segnali vetati           : {n_veto} ({n_veto/n_total*100:.1f}%)")
    print(f"  Segnali passati          : {n_pass} ({n_pass/n_total*100:.1f}%)")
    print(f"  False rejection rate     : {n_false_rej/n_total*100:.1f}%  (vetato ma era giusto)")
    print(f"  False acceptance rate    : {n_false_acc/n_total*100:.1f}%  (passato ma era sbagliato)")

    # Veto rate per regime
    print(f"\n  Veto rate per regime VIX (simulazione):")
    veto_by_regime = sim_df.groupby("vix_regime")["veto_rule_based"].agg(["sum", "count"])
    veto_by_regime["veto_pct"] = (veto_by_regime["sum"] / veto_by_regime["count"] * 100).round(1)
    for regime in ["LOW", "NORMAL", "ELEVATED", "CRISIS"]:
        if regime in veto_by_regime.index:
            row = veto_by_regime.loc[regime]
            print(f"    {regime:10s}: {row['veto_pct']:.1f}%  (n={int(row['count'])})")

    # Veto rate per conviction bucket
    print(f"\n  Veto rate per bucket conviction:")
    sim_df["conv_bucket"] = pd.cut(sim_df["conviction"],
                                    bins=[0, 0.3, 0.55, 0.7, 1.0],
                                    labels=["Bassa (0-0.3)", "Media (0.3-0.55)",
                                            "Buona (0.55-0.7)", "Alta (0.7-1.0)"])
    veto_by_conv = sim_df.groupby("conv_bucket", observed=True)["veto_rule_based"].agg(["sum", "count"])
    veto_by_conv["veto_pct"] = (veto_by_conv["sum"] / veto_by_conv["count"] * 100).round(1)
    for label, row in veto_by_conv.iterrows():
        print(f"    {str(label):20s}: {row['veto_pct']:.1f}%  (n={int(row['count'])})")

    # Salva CSV simulazione
    sim_path = os.path.join(RESULT_DIR, "03_devil_veto_simulation.csv")
    sim_df.to_csv(sim_path, index=False, float_format="%.3f")
    print(f"\n  Salvato: {sim_path}")

    # ── Analisi coerenza agenti ───────────────────────────────────────────────
    print("\n[3/4] Analisi coerenza agenti ...")
    print(f"\n  Veto rate per numero agenti concordi:")
    veto_by_agents = sim_df.groupby("agents_agreeing")["veto_rule_based"].agg(["sum", "count"])
    veto_by_agents["veto_pct"] = (veto_by_agents["sum"] / veto_by_agents["count"] * 100).round(1)
    for n_ag, row in veto_by_agents.iterrows():
        note = "  ← sotto MIN_AGENTS" if n_ag < MIN_AGENTS_AGREEING else ""
        print(f"    {n_ag} agenti: {row['veto_pct']:.1f}%{note}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[4/4] Salvataggio summary ...")
    summary_rows = [
        {"metrica": "Giorni analizzati (VIX)",        "valore": len(vix_df)},
        {"metrica": "VIX medio",                      "valore": f"{vix.mean():.1f}"},
        {"metrica": "VIX min",                        "valore": f"{vix.min():.1f}"},
        {"metrica": "VIX max",                        "valore": f"{vix.max():.1f}"},
        {"metrica": "Regime LOW (%)",                 "valore": f"{regime_pct.get('LOW', 0):.1f}%"},
        {"metrica": "Regime NORMAL (%)",              "valore": f"{regime_pct.get('NORMAL', 0):.1f}%"},
        {"metrica": "Regime ELEVATED (%)",            "valore": f"{regime_pct.get('ELEVATED', 0):.1f}%"},
        {"metrica": "Regime CRISIS (%)",              "valore": f"{regime_pct.get('CRISIS', 0):.1f}%"},
        {"metrica": "Veto rate atteso medio",         "valore": f"{expected_veto_rate*100:.1f}%"},
        {"metrica": "Segnali simulati",               "valore": n_total},
        {"metrica": "Veto rate simulazione",          "valore": f"{n_veto/n_total*100:.1f}%"},
        {"metrica": "False rejection rate",           "valore": f"{n_false_rej/n_total*100:.1f}%"},
        {"metrica": "False acceptance rate",          "valore": f"{n_false_acc/n_total*100:.1f}%"},
        {"metrica": "CONVICTION_THRESHOLD",           "valore": CONVICTION_THRESHOLD},
        {"metrica": "MIN_AGENTS_AGREEING",            "valore": MIN_AGENTS_AGREEING},
    ]
    summ_path = os.path.join(RESULT_DIR, "03_devil_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summ_path, index=False)
    print(f"  Salvato: {summ_path}")

    print("\n  Prossimo step: python backtest/04_backtest_breakout.py")
    print("=" * 65)


if __name__ == "__main__":
    run_backtest()
