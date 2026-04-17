# Athanor Alpha — Report Backtest F4

**Generato:** 2026-04-11 13:02  
**Periodo dati:** Aprile 2025 – Aprile 2026  
**Ticker:** 30  

## 1. Sintesi Categoria A — Agenti Infrastrutturali

| Agente | Risultato chiave | Warning |
|--------|-----------------|--------|
| Fundamentals Agent | Composite medio 70.3/100, 20 LONG / 1 SHORT / 9 NEUTRAL | Bias bullish strutturale; D/E formula sbagliata per Financials |
| Risk Manager | VaR portfolio 1.70%, Max DD 11.75% (entrambi OK) | MSTR/COIN/SMCI con VaR >6% individuale non bloccati; V/MA correlati 0.887 |
| Devil's Advocate | Veto rate atteso 16.5%, false rejection 21.9% | VIX regime decorativo; conviction threshold domina |
| Breakout Momentum | WR 53% combinato (3gg/5gg/10gg) | Bias LONG estremo; breakout confermati da volume 49.1% (sotto il caso) |

## 2. Classifica Agenti Operativi (Categoria B)

**Nota:** WR_5d è la metrica rilevante per il pipeline (orizzonte reale 3-4 giorni).  
WR_20d è gonfiato dal bull market 2025-2026 e non è indicativo.

| Agente | L | S | N | WR_5d | WR_20d | P&L_20d | Bias |
|--------|---|---|---|-------|--------|---------|------|
| ben_graham | 5 | 8 | 17 | 69.2% | 30.8% | -22.3% | BALANCED |
| michael_burry | 2 | 17 | 11 | 57.9% | 5.3% | -227.9% | BEARISH |
| cathie_wood | 11 | 5 | 14 | 43.8% | 75.0% | +164.0% | BULLISH |
| charlie_munger | 20 | 2 | 8 | 40.9% | 90.9% | +201.5% | BULLISH |
| growth_agent | 16 | 2 | 12 | 38.9% | 83.3% | +178.6% | BULLISH |
| bill_ackman | 17 | 2 | 11 | 36.8% | 84.2% | +118.8% | BULLISH |
| peter_lynch | 10 | 1 | 19 | 36.4% | 90.9% | +127.9% | BULLISH |
| stanley_druckenmiller | 17 | 3 | 10 | 35.0% | 90.0% | +183.7% | BULLISH |
| rakesh_jhunjhunwala | 16 | 4 | 10 | 35.0% | 85.0% | +153.3% | BULLISH |
| warren_buffett | 21 | 2 | 7 | 34.8% | 87.0% | +145.6% | BULLISH |
| phil_fisher | 18 | 1 | 11 | 31.6% | 94.7% | +194.1% | BULLISH |
| aswath_damodaran | 16 | 3 | 11 | 26.3% | 84.2% | +115.1% | BULLISH |
| mohnish_pabrai | 7 | 1 | 22 | 12.5% | 75.0% | +102.7% | BULLISH |

## 3. Tabella Qualità Agenti (1–5)

| Agente | Dati | Accuratezza | Volume | Bias | Filosofia | **Totale** |
|--------|------|-------------|--------|------|-----------|------------|
| Risk Manager | 5 | 4 | 5 | 5 | 4 | **4.6** |
| stanley_druckenmiller | 4 | 3 | 4 | 3 | 4 | **3.6** |
| aswath_damodaran | 4 | 2 | 4 | 3 | 5 | **3.6** |
| Fundamentals Agent | 5 | 3 | 4 | 2 | 3 | **3.4** |
| Breakout Momentum | 5 | 3 | 4 | 2 | 3 | **3.4** |
| growth_agent | 4 | 3 | 4 | 2 | 4 | **3.4** |
| michael_burry | 3 | 3 | 3 | 4 | 4 | **3.4** |
| cathie_wood | 4 | 3 | 3 | 3 | 4 | **3.4** |
| charlie_munger | 4 | 3 | 4 | 2 | 4 | **3.4** |
| mohnish_pabrai | 4 | 2 | 2 | 4 | 4 | **3.2** |
| ben_graham | 3 | 3 | 2 | 4 | 4 | **3.2** |
| Devil's Advocate | 4 | 3 | 4 | 3 | 2 | **3.2** |
| warren_buffett | 4 | 2 | 4 | 2 | 4 | **3.2** |
| rakesh_jhunjhunwala | 4 | 2 | 4 | 3 | 3 | **3.2** |
| phil_fisher | 4 | 2 | 4 | 2 | 4 | **3.2** |
| bill_ackman | 4 | 2 | 3 | 2 | 3 | **2.8** |
| peter_lynch | 3 | 2 | 3 | 2 | 4 | **2.8** |

## 4. Fix Prioritizzati — Piano v4

### [1] Devil's Advocate — ALTO impatto / BASSO effort

**Problema:** VIX regime quasi decorativo: conviction threshold domina, VIX aggiunge solo 5-10%  
**Fix:** Implementare conviction_threshold dinamica: 0.45 in LOW, 0.55 in NORMAL, 0.65 in ELEVATED, 0.75 in CRISIS  

### [2] Fundamentals Agent — MEDIO impatto / MEDIO effort

**Problema:** score_debt_equity penalizza banche e financials per leva strutturale (GS a 22.0)  
**Fix:** Aggiungere check settore: per Financials usare Tier 1 Capital Ratio o debt_to_assets invece di D/E  

### [3] Risk Manager — ALTO impatto / BASSO effort

**Problema:** MSTR/COIN/SMCI con VaR individuale >6% non vengono bloccati. V/MA ridondanti (corr 0.887)  
**Fix:** Aggiungere single-ticker VaR cap (es. max 4% VaR individuale nel portfolio). Rimuovere uno tra V e MA dall'universe.  

### [4] Breakout Momentum — MEDIO impatto / MEDIO effort

**Problema:** compute_breakout_score genera bias LONG strutturale in bull market. Breakout confermati da volume non battono il caso (49.1%)  
**Fix:** Aggiungere componente di mean-reversion: penalizzare score se RSI>70. Aumentare peso del componente SHORT (resistance breakdown).  

### [5] Tutti gli agenti fondamentali — ALTO impatto / MEDIO effort

**Problema:** WR_5d sistematicamente <50% per tutti gli agenti value/growth. Orizzonte fondamentale (trimestrale) incompatibile con swing 3-4 giorni  
**Fix:** Aggiungere filtro tecnico obbligatorio: un segnale fondamentale viene emesso solo se confermato da momentum tecnico (es. EMA trend positivo)  

### [6] michael_burry — MEDIO impatto / BASSO effort

**Problema:** WR_20d 5.3% in bull market — non per un bug, ma per mismatch contesto  
**Fix:** Aumentare peso di Burry nel Weight Adjuster solo in regime ELEVATED/CRISIS (VIX>25). In regime LOW abbassarlo automaticamente.  

### [7] peter_lynch — BASSO impatto / BASSO effort

**Problema:** PEG ratio None per molti ticker — agente parzialmente cieco  
**Fix:** Calcolare PEG ratio internamente: P/E trailing / earnings_growth (da yfinance info) quando non disponibile direttamente  

### [8] mohnish_pabrai — BASSO impatto / BASSO effort

**Problema:** WR_5d 12.5% — troppo selettivo per large cap US, solo 7 LONG su 30  
**Fix:** Abbassare soglie FCF yield (da 10% a 6% per LONG) e allargare il pool di ticker a mid cap  

### [9] Weight Adjuster (EWA) — MEDIO impatto / MEDIO effort

**Problema:** Non testato isolatamente in questa sessione  
**Fix:** Da includere in F4 seconda sessione: testare stabilità pesi EWA su dati storici con outcome tracker simulato  

## 5. Conclusioni

### Agenti da potenziare (priorità alta)
- **Devil's Advocate**: conviction threshold dinamica su VIX regime
- **Risk Manager**: single-ticker VaR cap, rimozione ticker ridondanti
- **Tutti gli agenti fondamentali**: aggiungere filtro tecnico di conferma

### Agenti da usare con cautela
- **Ben Graham**: funziona meglio su mercati orso e small cap value
- **Michael Burry**: peso da aumentare solo in regime ELEVATED/CRISIS
- **Mohnish Pabrai**: soglie troppo restrittive per large cap US

### Agenti più affidabili nel contesto attuale
- **Risk Manager**: logica corretta, dati completi
- **Cathie Wood**: più bilanciato degli altri growth (5 SHORT)
- **Charlie Munger / Phil Fisher**: migliori su orizzonte 20gg

---
*Report generato automaticamente da 06_backtest_report.py — Athanor Alpha F4*
