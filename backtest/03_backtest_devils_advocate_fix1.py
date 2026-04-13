"""
03_backtest_devils_advocate_fix1.py  –  Athanor Alpha | Fix 1 Validation
=========================================================================
Test comparativo: soglia fissa (0.50) vs soglia dinamica VIX-based.

Domanda a cui risponde:
  "La soglia dinamica riduce le false rejection in regime LOW
   senza aprire troppo il filtro in ELEVATED/CRISIS?"

Logica soglie dinamiche (da Fix 1):
  VIX < 18  (LOW):      0.45
  VIX 18-25 (NORMAL):   0.55
  VIX 25-35 (ELEVATED): 0.65
  VIX > 35  (CRISIS):   0.75

Esegui: python backtest/03_backtest_devils_advocate_fix1.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ── Parametri ─────────────────────────────────────────────────────────────────

FIXED_THRESHOLD = 0.50      # soglia originale (pre-fix)
MIN_AGENTS = 2              # minimo agenti concordi (invariato)

# Soglie dinamiche Fix 1 (opzione B: ELEVATED e CRISIS ammorbidite)
DYNAMIC_THRESHOLDS = {
    "LOW":      0.45,
    "NORMAL":   0.55,
    "ELEVATED": 0.60,
    "CRISIS":   0.65,
}

# VIX breakpoint per classify_vix_regime (Fix 1)
VIX_BREAKS = {"LOW": 18.0, "NORMAL": 25.0, "ELEVATED": 35.0}

N_SIGNALS = 1000
SEED = 42


# ── VIX regime classifier (Fix 1 breakpoints) ─────────────────────────────────

def classify_vix_regime(vix: float) -> str:
    if vix < VIX_BREAKS["LOW"]:
        return "LOW"
    elif vix < VIX_BREAKS["NORMAL"]:
        return "NORMAL"
    elif vix < VIX_BREAKS["ELEVATED"]:
        return "ELEVATED"
    else:
        return "CRISIS"


# ── Generazione segnali sintetici ─────────────────────────────────────────────

def simulate_signals(n: int = N_SIGNALS, seed: int = SEED) -> pd.DataFrame:
    """
    Genera n segnali sintetici con distribuzione VIX realistica
    (calibrata sul periodo apr 2025 – apr 2026).

    Per ogni segnale:
      - conviction:      forza del segnale (0.1 – 1.0)
      - agents_agreeing: n. agenti concordi (1–6)
      - vix_level:       livello VIX al momento del segnale
      - signal_was_correct: il segnale era giusto? (P ≈ conviction × 0.8 + 0.1)
    """
    rng = np.random.default_rng(seed)

    conviction = rng.uniform(0.1, 1.0, n)
    agents_agreeing = rng.integers(1, 7, n)

    # Distribuzione VIX realistica (apr 2025 – apr 2026)
    # Periodo dominato da VIX elevato per spike tariffari
    vix_pool = np.concatenate([
        rng.uniform(10, 18, 220),   # 22% regime LOW
        rng.uniform(18, 25, 310),   # 31% regime NORMAL
        rng.uniform(25, 35, 310),   # 31% regime ELEVATED
        rng.uniform(35, 60, 160),   # 16% regime CRISIS
    ])
    rng.shuffle(vix_pool)
    vix_level = vix_pool[:n]

    df = pd.DataFrame({
        "conviction":      conviction,
        "agents_agreeing": agents_agreeing,
        "vix_level":       vix_level,
    })

    df["vix_regime"] = df["vix_level"].apply(classify_vix_regime)

    # Esito reale: P(corretto) proporzionale alla conviction
    p_correct = np.clip(df["conviction"] * 0.8 + 0.1, 0.1, 0.9)
    df["signal_was_correct"] = rng.random(n) < p_correct.values

    return df


# ── Logica di veto ────────────────────────────────────────────────────────────

def apply_veto_fixed(df: pd.DataFrame) -> pd.Series:
    """Soglia fissa originale (0.50). Regime CRISIS non aveva protezione extra."""
    return (
        (df["conviction"] < FIXED_THRESHOLD) |
        (df["agents_agreeing"] < MIN_AGENTS)
    )


def apply_veto_dynamic(df: pd.DataFrame) -> pd.Series:
    """Soglia dinamica Fix 1: varia per regime VIX."""
    threshold = df["vix_regime"].map(DYNAMIC_THRESHOLDS)
    return (
        (df["conviction"] < threshold) |
        (df["agents_agreeing"] < MIN_AGENTS)
    )


# ── Metriche per regime ───────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, veto_col: str) -> pd.DataFrame:
    """
    Per ogni regime VIX calcola:
      - n_signals:         segnali totali nel regime
      - veto_rate:         % vetati
      - false_rejection:   % vetati che erano corretti (errore tipo I)
      - false_acceptance:  % passati che erano sbagliati (errore tipo II)
      - precision:         % dei passati che erano corretti
    """
    rows = []
    for regime in ["LOW", "NORMAL", "ELEVATED", "CRISIS"]:
        sub = df[df["vix_regime"] == regime]
        if len(sub) == 0:
            continue

        vetoed  = sub[veto_col]
        correct = sub["signal_was_correct"]

        n       = len(sub)
        n_veto  = vetoed.sum()
        n_pass  = n - n_veto
        fr      = (vetoed & correct).sum()        # false rejection
        fa      = (~vetoed & ~correct).sum()      # false acceptance
        prec    = (correct & ~vetoed).sum() / n_pass if n_pass > 0 else float("nan")

        rows.append({
            "regime":            regime,
            "n_signals":         n,
            "veto_rate_%":       round(n_veto / n * 100, 1),
            "false_rej_%":       round(fr / n * 100, 1),
            "false_acc_%":       round(fa / n * 100, 1),
            "precision_%":       round(prec * 100, 1),
        })

    return pd.DataFrame(rows).set_index("regime")


# ── Stampa tabella comparativa ────────────────────────────────────────────────

def print_comparison(fixed_m: pd.DataFrame, dyn_m: pd.DataFrame) -> None:
    regimes = ["LOW", "NORMAL", "ELEVATED", "CRISIS"]
    metrics = ["veto_rate_%", "false_rej_%", "false_acc_%", "precision_%"]
    labels  = ["Veto rate", "False rejection", "False acceptance", "Precision segnali passati"]

    print(f"\n{'Regime':<10} {'Metrica':<26} {'Fissa 0.50':>12} {'Dinamica':>12} {'Delta':>10}")
    print("-" * 74)
    for regime in regimes:
        if regime not in fixed_m.index or regime not in dyn_m.index:
            continue
        thresh = DYNAMIC_THRESHOLDS[regime]
        print(f"\n  {regime}  (threshold dinamica: {thresh:.2f})")
        for col, label in zip(metrics, labels):
            f_val = fixed_m.loc[regime, col]
            d_val = dyn_m.loc[regime, col]
            delta = d_val - f_val
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            # Per false_rejection il delta negativo è positivo (meno errori)
            good = ""
            if col == "false_rej_%" and delta < -0.5:
                good = " ✓"
            elif col == "false_rej_%" and delta > 0.5:
                good = " ✗"
            elif col == "false_acc_%" and delta > 0.5 and regime in ("ELEVATED", "CRISIS"):
                good = " ✗"
            elif col == "false_acc_%" and delta < -0.5 and regime in ("ELEVATED", "CRISIS"):
                good = " ✓"
            elif col == "precision_%" and delta > 0.5:
                good = " ✓"
            print(f"    {'':8} {label:<26} {f_val:>10.1f}%  {d_val:>10.1f}%  {arrow}{abs(delta):>6.1f}%{good}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 74)
    print("  ATHANOR ALPHA – Fix 1 Validation | Devil's Advocate")
    print("  Soglia fissa (0.50) vs Soglia dinamica VIX-based")
    print(f"  Eseguito: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 74)

    # Genera segnali (stesso seed → deterministico)
    print(f"\n[1/3] Generazione {N_SIGNALS} segnali sintetici (seed={SEED}) ...")
    df = simulate_signals()

    dist = df["vix_regime"].value_counts()
    print(f"\n  Distribuzione VIX regime nel campione:")
    for r in ["LOW", "NORMAL", "ELEVATED", "CRISIS"]:
        n = dist.get(r, 0)
        th_fixed   = FIXED_THRESHOLD
        th_dynamic = DYNAMIC_THRESHOLDS[r]
        delta_str  = f"  (soglia: {th_fixed:.2f} → {th_dynamic:.2f})"
        print(f"    {r:<10}: {n:>4} segnali{delta_str}")

    # Applica logica veto
    print("\n[2/3] Applicazione regole di veto ...")
    df["veto_fixed"]   = apply_veto_fixed(df)
    df["veto_dynamic"] = apply_veto_dynamic(df)

    # Calcola metriche
    fixed_m = compute_metrics(df, "veto_fixed")
    dyn_m   = compute_metrics(df, "veto_dynamic")

    # Stampa comparazione per regime
    print("\n[3/3] Confronto metriche per regime VIX:")
    print_comparison(fixed_m, dyn_m)

    # ── Sommario globale ───────────────────────────────────────────────────────
    n = len(df)
    print("\n" + "─" * 74)
    print("  SOMMARIO GLOBALE (tutti i regimi)")
    print("─" * 74)
    for label, col in [("Veto rate", "veto_fixed"), ("", "veto_dynamic")]:
        veto_col = col
        n_veto = df[veto_col].sum()
        fr = (df[veto_col] & df["signal_was_correct"]).sum()
        fa = (~df[veto_col] & ~df["signal_was_correct"]).sum()
        n_pass = n - n_veto
        prec = (df["signal_was_correct"] & ~df[veto_col]).sum() / n_pass if n_pass > 0 else 0
        tag = "FISSA 0.50  " if col == "veto_fixed" else "DINAMICA   "
        print(f"\n  [{tag}]")
        print(f"    Segnali vetati    : {n_veto:>4} / {n}  ({n_veto/n*100:.1f}%)")
        print(f"    False rejection   : {fr:>4} / {n}  ({fr/n*100:.1f}%)")
        print(f"    False acceptance  : {fa:>4} / {n}  ({fa/n*100:.1f}%)")
        print(f"    Precision passati : {prec*100:.1f}%")

    # ── Valutazione finale (per-regime) ─────────────────────────────────────
    # Il verdict globale non è significativo perché vetare di più in
    # ELEVATED/CRISIS per definizione aumenta la false rejection globale.
    # Criteri corretti (valutati per regime):
    #   1. LOW:              false rejection Δ ≤ +1%
    #   2. ELEVATED:         false acceptance migliora (Δ < 0)
    #   3. ELEVATED:         precision migliora (Δ > 0)
    #   4. ELEVATED:         false rejection Δ < +8%
    #   5. CRISIS:           false acceptance migliora (Δ < 0)
    #   6. CRISIS:           precision migliora (Δ > 0)
    #   7. CRISIS:           false rejection Δ < +8%

    print("\n" + "─" * 74)
    print("  VERDICT (valutazione per-regime)")
    print("─" * 74)

    checks = {}
    for regime in ["LOW", "NORMAL", "ELEVATED", "CRISIS"]:
        if regime not in fixed_m.index:
            continue
        fr_d = dyn_m.loc[regime, "false_rej_%"]  - fixed_m.loc[regime, "false_rej_%"]
        fa_d = dyn_m.loc[regime, "false_acc_%"]  - fixed_m.loc[regime, "false_acc_%"]
        pr_d = dyn_m.loc[regime, "precision_%"]  - fixed_m.loc[regime, "precision_%"]
        checks[regime] = {"fr_delta": fr_d, "fa_delta": fa_d, "pr_delta": pr_d}

    c1 = checks["LOW"]["fr_delta"]       <= 1.0
    c2 = checks["ELEVATED"]["fa_delta"]  <  0
    c3 = checks["ELEVATED"]["pr_delta"]  >  0
    c4 = checks["ELEVATED"]["fr_delta"]  <  8.0
    c5 = checks["CRISIS"]["fa_delta"]    <  0
    c6 = checks["CRISIS"]["pr_delta"]    >  0
    c7 = checks["CRISIS"]["fr_delta"]    <  8.0

    def tick(ok): return "✓" if ok else "✗"

    print(f"\n  Criterio 1 — LOW:      false rejection  Δ ≤ +1%   Δ={checks['LOW']['fr_delta']:+.1f}%  {tick(c1)}")
    print(f"  Criterio 2 — ELEVATED: false acceptance Δ <  0    Δ={checks['ELEVATED']['fa_delta']:+.1f}%  {tick(c2)}")
    print(f"  Criterio 3 — ELEVATED: precision        Δ >  0    Δ={checks['ELEVATED']['pr_delta']:+.1f}%  {tick(c3)}")
    print(f"  Criterio 4 — ELEVATED: false rejection  Δ < +8%   Δ={checks['ELEVATED']['fr_delta']:+.1f}%  {tick(c4)}")
    print(f"  Criterio 5 — CRISIS:   false acceptance Δ <  0    Δ={checks['CRISIS']['fa_delta']:+.1f}%  {tick(c5)}")
    print(f"  Criterio 6 — CRISIS:   precision        Δ >  0    Δ={checks['CRISIS']['pr_delta']:+.1f}%  {tick(c6)}")
    print(f"  Criterio 7 — CRISIS:   false rejection  Δ < +8%   Δ={checks['CRISIS']['fr_delta']:+.1f}%  {tick(c7)}")

    n_ok  = sum([c1, c2, c3, c4, c5, c6, c7])
    all_ok = all([c1, c2, c3, c4, c5, c6, c7])

    print()
    if all_ok:
        print("  ✅ Fix 1 VALIDATO: tutti i criteri per-regime soddisfatti.")
        print("     Meno false rejection in LOW, maggiore protezione in ELEVATED/CRISIS.")
    elif n_ok >= 5:
        print(f"  ⚠️  Fix 1 PARZIALE: {n_ok}/7 criteri soddisfatti — accettabile con riserva.")
    else:
        print(f"  ❌ Fix 1 NON VALIDATO: solo {n_ok}/7 criteri soddisfatti.")

    # ── Salva CSV ─────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "03_devil_fix1_comparison.csv")

    combined = fixed_m.add_suffix("_fixed").join(dyn_m.add_suffix("_dynamic"))
    combined.to_csv(out_path)
    print(f"\n  CSV salvato: {out_path}")
    print("=" * 74)


if __name__ == "__main__":
    run()
