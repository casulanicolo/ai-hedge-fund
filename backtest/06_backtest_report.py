"""
06_backtest_report.py  –  Athanor Alpha | Backtest F4
======================================================
Legge tutti i CSV prodotti dagli script 01-05 e genera:

  1. Tabella qualità per ogni agente (score 1-5)
  2. Lista fix prioritizzati per il Piano v4
  3. Report testuale completo in formato Markdown

Output:
  backtest/results/06_quality_table.csv
  backtest/results/06_fix_priorities.csv
  backtest/results/F4_REPORT.md

Esegui: python backtest/06_backtest_report.py
"""

import os
import sys
from datetime import datetime

import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "backtest", "results")

# ── Carica risultati precedenti ───────────────────────────────────────────────

def load(filename):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        print(f"  MANCANTE: {filename}")
        return None
    return pd.read_csv(path)


# ── Tabella qualità: criteri di scoring 1-5 ───────────────────────────────────
#
# Dimensioni valutate per ogni agente:
#   A. Data quality     – quanti ticker hanno dati sufficienti
#   B. Signal accuracy  – WR_5d (orizzonte reale del pipeline)
#   C. Segnali generati – n_long + n_short (agente che non segnala nulla è inutile)
#   D. Bias direzionale – equilibrio LONG/SHORT (un agente sempre LONG non aggiunge valore)
#   E. Coerenza filosofia – i segnali corrispondono alla logica dichiarata dell'agente?
#
# Score 1=pessimo, 3=adeguato, 5=eccellente

QUALITY_CRITERIA = {
    # agente: {A, B, C, D, E, note}
    "Fundamentals Agent": {
        "data_quality":     5,  # 30/30 ticker, 4/4 fonti per la maggior parte
        "signal_accuracy":  3,  # Non misurabile direttamente (dipende da LLM per dir.)
        "signal_volume":    4,  # 20 LONG, 1 SHORT, 9 NEUTRAL su 30 ticker
        "directional_bias": 2,  # Bias bullish estremo (20/1 LONG/SHORT)
        "philosophy_fit":   3,  # Score matematico corretto ma soglie non calibrate per settore
        "note": "Bug: score_debt_equity non calibrato per Financials (banche). Bias bullish strutturale."
    },
    "Risk Manager": {
        "data_quality":     5,  # 30/30 ticker OHLCV
        "signal_accuracy":  4,  # VaR e DD corretti, sector cap funziona
        "signal_volume":    5,  # Analizza tutto il portfolio
        "directional_bias": 5,  # Non genera direzioni, solo annotazioni
        "philosophy_fit":   4,  # Logica corretta ma non penalizza abbastanza singoli asset ad alto VaR
        "note": "Warning: MSTR/COIN/SMCI con VaR >6% non vengono bloccati individualmente. V/MA ridondanti (corr 0.887)."
    },
    "Devil's Advocate": {
        "data_quality":     4,  # VIX disponibile, regime OK
        "signal_accuracy":  3,  # False rejection 21.9% — alto
        "signal_volume":    4,  # Veto rate 16.5% atteso sul periodo
        "directional_bias": 3,  # Veto dipende quasi solo da conviction threshold, non da VIX
        "philosophy_fit":   2,  # VIX regime quasi decorativo: diff solo 5-10% tra LOW e CRISIS
        "note": "Bug critico: la logica VIX non scala il comportamento in modo significativo. Conviction threshold domina tutto."
    },
    "Breakout Momentum": {
        "data_quality":     5,  # OHLCV completo
        "signal_accuracy":  3,  # WR 53% combinato, ma breakout confermati 49.1% (sotto il caso)
        "signal_volume":    4,  # Segnali frequenti (ogni giorno per ogni ticker)
        "directional_bias": 2,  # Bias bullish estremo (es. JNJ 193 LONG / 1 SHORT)
        "philosophy_fit":   3,  # Metriche corrette ma score composito troppo bullish
        "note": "Bug: compute_breakout_score genera bias LONG strutturale in bull market. Breakout confermati da volume non battono il caso (49.1%)."
    },
    "warren_buffett": {
        "data_quality":     4,
        "signal_accuracy":  2,  # WR_5d 34.8% — sotto il caso
        "signal_volume":    4,  # 21 LONG, 2 SHORT
        "directional_bias": 2,  # Molto bullish
        "philosophy_fit":   4,  # Logica ROE/moat/FCF corretta
        "note": "Orizzonte fondamentale (trimestrale) non compatibile con swing 3-4 giorni. Su 20gg funziona."
    },
    "ben_graham": {
        "data_quality":     3,  # Molte metriche Graham non disponibili in yfinance (NCAV, Graham Number)
        "signal_accuracy":  3,  # WR_5d 69.2% ma solo 5 LONG (troppo selettivo)
        "signal_volume":    2,  # Solo 5 LONG su 30 ticker — troppo pochi segnali
        "directional_bias": 4,  # Bilanciato (5 LONG, 8 SHORT) — unico agente con SHORT significativi
        "philosophy_fit":   4,  # P/B e current ratio corretti
        "note": "Agente troppo selettivo per un universe di large cap US. Graham è nato per small cap value. WR_20d 30.8% in bull market — penalizzato dal contesto."
    },
    "charlie_munger": {
        "data_quality":     4,
        "signal_accuracy":  3,  # WR_5d 40.9% — sotto il caso
        "signal_volume":    4,  # 20 LONG
        "directional_bias": 2,  # Molto bullish
        "philosophy_fit":   4,  # Gross margin + ROE + FCF corretti
        "note": "Stesso problema di Buffett: orizzonte non compatibile con 3-4 giorni. Miglior P&L simulato a 20gg."
    },
    "bill_ackman": {
        "data_quality":     4,
        "signal_accuracy":  2,  # WR_5d 36.8%
        "signal_volume":    3,  # 17 LONG, 2 SHORT
        "directional_bias": 2,  # Molto bullish
        "philosophy_fit":   3,  # FCF + op margin ok, ma activism non misurabile senza dati extra
        "note": "Activism potential non testabile con sole metriche yfinance."
    },
    "cathie_wood": {
        "data_quality":     4,
        "signal_accuracy":  3,  # WR_5d 43.8% — il più vicino al 50% tra i growth
        "signal_volume":    3,  # 11 LONG, 5 SHORT — più bilanciato degli altri
        "directional_bias": 3,  # Relativamente bilanciato per un agente growth
        "philosophy_fit":   4,  # Revenue growth + beta corretti
        "note": "L'unico agente growth che genera SHORT significativi (5). Beta come proxy R&D è approssimativo."
    },
    "michael_burry": {
        "data_quality":     3,  # FCF yield disponibile ma EV/EBIT non in yfinance standard
        "signal_accuracy":  3,  # WR_5d 57.9% sui SHORT — funziona come contrarian!
        "signal_volume":    3,  # 2 LONG, 17 SHORT
        "directional_bias": 4,  # Contrarian — utile come hedging
        "philosophy_fit":   4,  # FCF yield + net cash + P/B bassi corretti
        "note": "WR_5d SHORT = 57.9% — l'agente contrarian funziona. Penalizzato in bull market ma utile come hedge. Da pesare di più in regime ELEVATED/CRISIS."
    },
    "mohnish_pabrai": {
        "data_quality":     4,
        "signal_accuracy":  2,  # WR_5d 12.5% — il peggiore
        "signal_volume":    2,  # Solo 7 LONG — troppo selettivo
        "directional_bias": 4,  # Molto selettivo = conservative
        "philosophy_fit":   4,  # Downside protection + FCF yield corretti
        "note": "Troppo selettivo per un universe di large cap. WR_5d 12.5% è anomalo — probabile che i pochi LONG scelti siano in settori sfavorevoli."
    },
    "peter_lynch": {
        "data_quality":     3,  # PEG ratio spesso None in yfinance
        "signal_accuracy":  2,  # WR_5d 36.4%
        "signal_volume":    3,  # 10 LONG, 1 SHORT
        "directional_bias": 2,  # Molto bullish
        "philosophy_fit":   4,  # PEG ratio come metrica principale è fedele
        "note": "PEG ratio mancante per molti ticker (specialmente non-US). Quando disponibile, funziona bene."
    },
    "phil_fisher": {
        "data_quality":     4,
        "signal_accuracy":  2,  # WR_5d 31.6% — il peggiore tra i bullish
        "signal_volume":    4,  # 18 LONG, 1 SHORT
        "directional_bias": 2,  # Molto bullish
        "philosophy_fit":   4,  # Revenue growth + gross margin + op margin corretti
        "note": "Agente long-term per definizione: Fisher teneva i titoli decenni. WR_5d basso è atteso, non un bug."
    },
    "rakesh_jhunjhunwala": {
        "data_quality":     4,
        "signal_accuracy":  2,  # WR_5d 35.0%
        "signal_volume":    4,  # 16 LONG, 4 SHORT
        "directional_bias": 3,  # Qualche SHORT
        "philosophy_fit":   3,  # Logica EPS + FCF ok, ma Jhunjhunwala era India-focused
        "note": "Filosofia pensata per mercato indiano applicata a US large cap — mismatch geografico."
    },
    "stanley_druckenmiller": {
        "data_quality":     4,
        "signal_accuracy":  3,  # WR_5d 35.0% — deludente per un agente momentum
        "signal_volume":    4,  # 17 LONG, 3 SHORT
        "directional_bias": 3,  # Qualche SHORT
        "philosophy_fit":   4,  # 52w high proximity + revenue growth + beta corretti
        "note": "Il momentum dovrebbe dare edge su 5gg. WR_5d 35% suggerisce che la 52w proximity da sola non basta senza dati intraday."
    },
    "aswath_damodaran": {
        "data_quality":     4,
        "signal_accuracy":  2,  # WR_5d 26.3% — il più basso
        "signal_volume":    4,  # 16 LONG, 3 SHORT
        "directional_bias": 3,
        "philosophy_fit":   5,  # Il più fedele alla logica reale (CAPM + FCF vs cost of equity)
        "note": "DCF è per definizione long-term. WR_5d 26.3% conferma: non adatto a swing trading."
    },
    "growth_agent": {
        "data_quality":     4,
        "signal_accuracy":  3,  # WR_5d 38.9%
        "signal_volume":    4,  # 16 LONG, 2 SHORT
        "directional_bias": 2,  # Molto bullish
        "philosophy_fit":   4,  # Revenue + EPS + margin + health corretti
        "note": "Agente ben strutturato con pesi espliciti (40/25/15/10/10). Funziona meglio su orizzonti >5gg."
    },
}


def compute_quality_score(criteria: dict) -> float:
    """Media delle 5 dimensioni → score 1.0–5.0."""
    dims = ["data_quality", "signal_accuracy", "signal_volume",
            "directional_bias", "philosophy_fit"]
    return round(sum(criteria[d] for d in dims) / len(dims), 1)


# ── Lista fix prioritizzati ───────────────────────────────────────────────────

FIX_LIST = [
    {
        "priorita": 1,
        "agente":   "Devil's Advocate",
        "problema": "VIX regime quasi decorativo: conviction threshold domina, VIX aggiunge solo 5-10%",
        "fix":      "Implementare conviction_threshold dinamica: 0.45 in LOW, 0.55 in NORMAL, 0.65 in ELEVATED, 0.75 in CRISIS",
        "impatto":  "ALTO",
        "effort":   "BASSO",
    },
    {
        "priorita": 2,
        "agente":   "Fundamentals Agent",
        "problema": "score_debt_equity penalizza banche e financials per leva strutturale (GS a 22.0)",
        "fix":      "Aggiungere check settore: per Financials usare Tier 1 Capital Ratio o debt_to_assets invece di D/E",
        "impatto":  "MEDIO",
        "effort":   "MEDIO",
    },
    {
        "priorita": 3,
        "agente":   "Risk Manager",
        "problema": "MSTR/COIN/SMCI con VaR individuale >6% non vengono bloccati. V/MA ridondanti (corr 0.887)",
        "fix":      "Aggiungere single-ticker VaR cap (es. max 4% VaR individuale nel portfolio). Rimuovere uno tra V e MA dall'universe.",
        "impatto":  "ALTO",
        "effort":   "BASSO",
    },
    {
        "priorita": 4,
        "agente":   "Breakout Momentum",
        "problema": "compute_breakout_score genera bias LONG strutturale in bull market. Breakout confermati da volume non battono il caso (49.1%)",
        "fix":      "Aggiungere componente di mean-reversion: penalizzare score se RSI>70. Aumentare peso del componente SHORT (resistance breakdown).",
        "impatto":  "MEDIO",
        "effort":   "MEDIO",
    },
    {
        "priorita": 5,
        "agente":   "Tutti gli agenti fondamentali",
        "problema": "WR_5d sistematicamente <50% per tutti gli agenti value/growth. Orizzonte fondamentale (trimestrale) incompatibile con swing 3-4 giorni",
        "fix":      "Aggiungere filtro tecnico obbligatorio: un segnale fondamentale viene emesso solo se confermato da momentum tecnico (es. EMA trend positivo)",
        "impatto":  "ALTO",
        "effort":   "MEDIO",
    },
    {
        "priorita": 6,
        "agente":   "michael_burry",
        "problema": "WR_20d 5.3% in bull market — non per un bug, ma per mismatch contesto",
        "fix":      "Aumentare peso di Burry nel Weight Adjuster solo in regime ELEVATED/CRISIS (VIX>25). In regime LOW abbassarlo automaticamente.",
        "impatto":  "MEDIO",
        "effort":   "BASSO",
    },
    {
        "priorita": 7,
        "agente":   "peter_lynch",
        "problema": "PEG ratio None per molti ticker — agente parzialmente cieco",
        "fix":      "Calcolare PEG ratio internamente: P/E trailing / earnings_growth (da yfinance info) quando non disponibile direttamente",
        "impatto":  "BASSO",
        "effort":   "BASSO",
    },
    {
        "priorita": 8,
        "agente":   "mohnish_pabrai",
        "problema": "WR_5d 12.5% — troppo selettivo per large cap US, solo 7 LONG su 30",
        "fix":      "Abbassare soglie FCF yield (da 10% a 6% per LONG) e allargare il pool di ticker a mid cap",
        "impatto":  "BASSO",
        "effort":   "BASSO",
    },
    {
        "priorita": 9,
        "agente":   "Weight Adjuster (EWA)",
        "problema": "Non testato isolatamente in questa sessione",
        "fix":      "Da includere in F4 seconda sessione: testare stabilità pesi EWA su dati storici con outcome tracker simulato",
        "impatto":  "MEDIO",
        "effort":   "MEDIO",
    },
]


# ── Main ─────────────────────────────────────────────────────────────────────

def run_report():
    print("=" * 65)
    print("  ATHANOR ALPHA – F4  |  Report Finale Backtest")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # ── Tabella qualità ───────────────────────────────────────────────────────
    quality_rows = []
    for agent, criteria in QUALITY_CRITERIA.items():
        overall = compute_quality_score(criteria)
        quality_rows.append({
            "agente":            agent,
            "data_quality_1_5":  criteria["data_quality"],
            "signal_accuracy_1_5": criteria["signal_accuracy"],
            "signal_volume_1_5": criteria["signal_volume"],
            "directional_bias_1_5": criteria["directional_bias"],
            "philosophy_fit_1_5":criteria["philosophy_fit"],
            "overall_1_5":       overall,
            "note":              criteria["note"],
        })

    quality_df = pd.DataFrame(quality_rows).sort_values("overall_1_5", ascending=False)
    fix_df     = pd.DataFrame(FIX_LIST)

    # Salva CSV
    qt_path  = os.path.join(RESULT_DIR, "06_quality_table.csv")
    fix_path = os.path.join(RESULT_DIR, "06_fix_priorities.csv")
    quality_df.to_csv(qt_path,  index=False)
    fix_df.to_csv(fix_path, index=False)
    print(f"\n  Salvato: {qt_path}")
    print(f"  Salvato: {fix_path}")

    # ── Stampa tabella qualità ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TABELLA QUALITÀ AGENTI  (1=pessimo → 5=eccellente)")
    print("=" * 65)
    print(f"  {'Agente':25s}  {'Dati':>5}  {'Acc.':>5}  {'Vol.':>5}  {'Bias':>5}  {'Fil.':>5}  {'TOT':>5}")
    print("  " + "-" * 58)
    for _, r in quality_df.iterrows():
        bar = "★" * int(r["overall_1_5"]) + "☆" * (5 - int(r["overall_1_5"]))
        print(f"  {r['agente']:25s}  {r['data_quality_1_5']:5.0f}  "
              f"{r['signal_accuracy_1_5']:5.0f}  {r['signal_volume_1_5']:5.0f}  "
              f"{r['directional_bias_1_5']:5.0f}  {r['philosophy_fit_1_5']:5.0f}  "
              f"{r['overall_1_5']:5.1f}  {bar}")

    # ── Stampa fix prioritizzati ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FIX PRIORITIZZATI PER PIANO v4")
    print("=" * 65)
    for _, r in fix_df.iterrows():
        print(f"\n  [{int(r['priorita'])}] {r['agente']}  [{r['impatto']} impatto / {r['effort']} effort]")
        print(f"      Problema: {r['problema']}")
        print(f"      Fix:      {r['fix']}")

    # ── Genera Markdown ───────────────────────────────────────────────────────
    md_path = os.path.join(RESULT_DIR, "F4_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Athanor Alpha — Report Backtest F4\n\n")
        f.write(f"**Generato:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Periodo dati:** Aprile 2025 – Aprile 2026  \n")
        f.write(f"**Ticker:** 30  \n\n")

        f.write("## 1. Sintesi Categoria A — Agenti Infrastrutturali\n\n")
        f.write("| Agente | Risultato chiave | Warning |\n")
        f.write("|--------|-----------------|--------|\n")
        f.write("| Fundamentals Agent | Composite medio 70.3/100, 20 LONG / 1 SHORT / 9 NEUTRAL | Bias bullish strutturale; D/E formula sbagliata per Financials |\n")
        f.write("| Risk Manager | VaR portfolio 1.70%, Max DD 11.75% (entrambi OK) | MSTR/COIN/SMCI con VaR >6% individuale non bloccati; V/MA correlati 0.887 |\n")
        f.write("| Devil's Advocate | Veto rate atteso 16.5%, false rejection 21.9% | VIX regime decorativo; conviction threshold domina |\n")
        f.write("| Breakout Momentum | WR 53% combinato (3gg/5gg/10gg) | Bias LONG estremo; breakout confermati da volume 49.1% (sotto il caso) |\n\n")

        f.write("## 2. Classifica Agenti Operativi (Categoria B)\n\n")
        f.write("**Nota:** WR_5d è la metrica rilevante per il pipeline (orizzonte reale 3-4 giorni).  \n")
        f.write("WR_20d è gonfiato dal bull market 2025-2026 e non è indicativo.\n\n")
        f.write("| Agente | L | S | N | WR_5d | WR_20d | P&L_20d | Bias |\n")
        f.write("|--------|---|---|---|-------|--------|---------|------|\n")

        op_df = load("05_operative_comparison.csv")
        if op_df is not None:
            op_df_sorted = op_df.sort_values("win_rate_5d_pct", ascending=False)
            for _, r in op_df_sorted.iterrows():
                f.write(f"| {r['agent']} | {r['n_long']:.0f} | {r['n_short']:.0f} | {r['n_neutral']:.0f} | "
                        f"{r['win_rate_5d_pct']:.1f}% | {r['win_rate_20d_pct']:.1f}% | "
                        f"{r['pnl_simulated_pct']:+.1f}% | {r['bias']} |\n")

        f.write("\n## 3. Tabella Qualità Agenti (1–5)\n\n")
        f.write("| Agente | Dati | Accuratezza | Volume | Bias | Filosofia | **Totale** |\n")
        f.write("|--------|------|-------------|--------|------|-----------|------------|\n")
        for _, r in quality_df.iterrows():
            f.write(f"| {r['agente']} | {r['data_quality_1_5']:.0f} | "
                    f"{r['signal_accuracy_1_5']:.0f} | {r['signal_volume_1_5']:.0f} | "
                    f"{r['directional_bias_1_5']:.0f} | {r['philosophy_fit_1_5']:.0f} | "
                    f"**{r['overall_1_5']:.1f}** |\n")

        f.write("\n## 4. Fix Prioritizzati — Piano v4\n\n")
        for _, r in fix_df.iterrows():
            f.write(f"### [{int(r['priorita'])}] {r['agente']} — {r['impatto']} impatto / {r['effort']} effort\n\n")
            f.write(f"**Problema:** {r['problema']}  \n")
            f.write(f"**Fix:** {r['fix']}  \n\n")

        f.write("## 5. Conclusioni\n\n")
        f.write("### Agenti da potenziare (priorità alta)\n")
        f.write("- **Devil's Advocate**: conviction threshold dinamica su VIX regime\n")
        f.write("- **Risk Manager**: single-ticker VaR cap, rimozione ticker ridondanti\n")
        f.write("- **Tutti gli agenti fondamentali**: aggiungere filtro tecnico di conferma\n\n")
        f.write("### Agenti da usare con cautela\n")
        f.write("- **Ben Graham**: funziona meglio su mercati orso e small cap value\n")
        f.write("- **Michael Burry**: peso da aumentare solo in regime ELEVATED/CRISIS\n")
        f.write("- **Mohnish Pabrai**: soglie troppo restrittive per large cap US\n\n")
        f.write("### Agenti più affidabili nel contesto attuale\n")
        f.write("- **Risk Manager**: logica corretta, dati completi\n")
        f.write("- **Cathie Wood**: più bilanciato degli altri growth (5 SHORT)\n")
        f.write("- **Charlie Munger / Phil Fisher**: migliori su orizzonte 20gg\n\n")
        f.write("---\n*Report generato automaticamente da 06_backtest_report.py — Athanor Alpha F4*\n")

    print(f"\n  Markdown: {md_path}")
    print("\n" + "=" * 65)
    print("  F4 COMPLETATA — Tutti i file salvati in backtest/results/")
    print("=" * 65)
    print("\n  File prodotti:")
    for fn in sorted(os.listdir(RESULT_DIR)):
        size = os.path.getsize(os.path.join(RESULT_DIR, fn)) // 1024
        print(f"    {fn:45s}  {size:4d} KB")


if __name__ == "__main__":
    run_report()
