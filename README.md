# Athanor Alpha

**AI-powered multi-agent hedge fund simulation** — proof-of-concept educativo che replica il processo decisionale di un fondo quantistico tramite LangGraph, agenti LLM a personalità filosofiche, e un sistema di feedback loop basato su SQLite.

> **Disclaimer**: Questo progetto è esclusivamente a fini educativi e di ricerca. Non costituisce consulenza finanziaria. Non usare in produzione con denaro reale.

---

## Indice

1. [Panoramica](#panoramica)
2. [Architettura generale](#architettura-generale)
3. [Requisiti e installazione](#requisiti-e-installazione)
4. [Configurazione](#configurazione)
5. [Modalità di esecuzione](#modalità-di-esecuzione)
6. [Struttura del progetto — file per file](#struttura-del-progetto--file-per-file)
   - [Entry point](#entry-point)
   - [Grafo LangGraph (`src/graph/`)](#grafo-langgraph-srcgraph)
   - [Agenti (`src/agents/`)](#agenti-srcagents)
   - [Data layer (`src/data/`)](#data-layer-srcdata)
   - [Database (`src/db/`)](#database-srcdb)
   - [Feedback loop (`src/feedback/`)](#feedback-loop-srcfeedback)
   - [Utilità (`src/utils/`)](#utilità-srcutils)
   - [API & Tools (`src/tools/`)](#api--tools-srctools)
   - [LLM routing (`src/llm/`)](#llm-routing-srcllm)
   - [Indicatori tecnici (`src/indicators/`)](#indicatori-tecnici-srcindicators)
   - [Alert & Monitoring (`src/alerts/`, `src/monitor/`)](#alert--monitoring-srcalerts-srcmonitor)
   - [CLI (`src/cli/`)](#cli-srccli)
   - [Portfolio (`src/portfolio/`)](#portfolio-srcportfolio)
   - [Backtesting (`src/backtesting/`)](#backtesting-srcbacktesting)
   - [Test (`tests/`)](#test-tests)
   - [Config (`config/`)](#config-config)
   - [Database & log runtime (`db/`, `logs/`, `cache/`)](#database--log-runtime-db-logs-cache)
7. [Topologia del grafo](#topologia-del-grafo)
8. [Dimensioni di analisi (FIX C2)](#dimensioni-di-analisi-fix-c2)
9. [Agenti filosofici — dettaglio](#agenti-filosofici--dettaglio)
10. [Sistema di risk management](#sistema-di-risk-management)
11. [Portfolio manager — logica di aggregazione](#portfolio-manager--logica-di-aggregazione)
12. [Backtesting e walk-forward](#backtesting-e-walk-forward)
13. [Feedback loop e weight adjustment](#feedback-loop-e-weight-adjustment)
14. [Schema database SQLite](#schema-database-sqlite)
15. [Variabili d'ambiente](#variabili-dampiente)
16. [Limitazioni e stato dell'arte](#limitazioni-e-stato-dellarte)

---

## Panoramica

Athanor Alpha è un sistema multi-agente che simula il processo di analisi e decisione di un hedge fund. Il flusso prevede:

1. **Prefetch** batch di dati (yfinance, SEC EDGAR, macro) per tutti i ticker.
2. **Analisi parallela** da parte di 15+ agenti con personalità ispirate a investitori reali.
3. **Aggregazione** ortogonale su 4 dimensioni: `FUNDAMENTALS`, `TECHNICAL`, `SENTIMENT`, `MACRO`.
4. **Risk management** deterministico (VaR, correlazioni, concentrazione settoriale).
5. **Devil's advocate** — veto automatico basato su regime VIX e coerenza del segnale.
6. **Portfolio manager** — conviction scoring, Kelly-sizing, raccomandazioni finali.
7. **Feedback loop** — tracking esiti reali, aggiornamento pesi agenti via EWA.
8. **Backtesting** — engine storico con metriche Sharpe/Sortino/max-drawdown + walk-forward IS/OOS.

**Stack tecnico**: Python 3.11+, LangGraph, LangChain, Pydantic, SQLite, yfinance, pandas, numpy, Rich.

---

## Architettura generale

```
┌─────────────────────────────────────────────────────────────────────┐
│                          LangGraph Pipeline                         │
│                                                                     │
│  START → data_prefetch                                              │
│              │                                                      │
│    ┌─────────┼──────────────────────────────────┐                  │
│    │   Agenti paralleli (fan-out)                │                  │
│    │  warren_buffett  ben_graham  charlie_munger │                  │
│    │  michael_burry   bill_ackman cathie_wood    │                  │
│    │  phil_fisher     peter_lynch mohnish_pabrai │                  │
│    │  damodaran       druckenmiller              │                  │
│    │  fundamentals    valuation   growth         │                  │
│    │  technicals      breakout_momentum          │                  │
│    │  sentiment       macro                      │                  │
│    └─────────┬──────────────────────────────────┘                  │
│              │                                                      │
│         devils_advocate (veto rule-based)                           │
│              │                                                      │
│         risk_manager (VaR, correlazioni, ATR levels)                │
│              │                                                      │
│         portfolio_manager (aggregazione + sizing)                   │
│              │                                                      │
│         prediction_log (scrivi su SQLite)                           │
│              │                                                      │
│         time_exit (alert email posizioni aperte ≥4gg)               │
│              │                                                      │
│            END                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Ogni nodo può essere saltato tramite flag in `state["metadata"]`.

---

## Requisiti e installazione

```bash
# Python 3.11+
git clone https://github.com/<org>/athanor-alpha.git
cd athanor-alpha

# Installa dipendenze
pip install -r requirements.txt

# Oppure con poetry
poetry install

# Copia e configura le variabili d'ambiente
cp .env.example .env
# Edita .env con le tue chiavi API

# Inizializza il database SQLite
python -m src.db.init_db
```

### Dipendenze principali (`requirements.txt`)

| Pacchetto | Uso |
|-----------|-----|
| `langgraph` | Orchestrazione grafo agenti |
| `langchain` | Wrapper LLM, prompt templates |
| `yfinance` | Dati OHLCV, financials, news |
| `pandas`, `numpy` | Calcolo indicatori, metriche |
| `pydantic` | Validazione modelli dati |
| `requests` | SEC EDGAR API, Financial Datasets API |
| `python-dotenv` | Caricamento `.env` |
| `rich`, `tabulate`, `colorama` | Output terminale |
| `questionary` | CLI interattiva |
| `matplotlib` | Grafico equity curve (backtest) |
| `schedule` | Scheduling cron daemon |
| `pytest` | Test suite |

---

## Configurazione

### `config/tickers.yaml`

Lista ticker monitorati dal pipeline produzione:

```yaml
tickers:
  - AAPL
  - MSFT
  - NVDA
  - TSLA
  - MSTR
  - COIN
  - SMCI
  - MELI
  - BTC-USD
  - ETH-USD
  - SOL-USD
  - SPY       # benchmark
```

I ticker crypto (BTC-USD, ETH-USD, SOL-USD) vengono automaticamente esclusi dal fetch SEC EDGAR (nessun CIK disponibile).

### `config/risk_params.yaml`

Parametri di rischio caricati da `risk_manager.py` e `portfolio_manager.py`:

| Parametro | Valore default | Descrizione |
|-----------|---------------|-------------|
| `target_daily_vol` | 1% | Volatilità target giornaliera per vol-targeting |
| `kelly_fraction` | 0.25 | Quarter-Kelly per position sizing |
| `min_position_pct` | 2% | Soglia minima posizione |
| `max_position_pct` | 15% | Soglia massima posizione (hard cap) |
| `wc_threshold` | 0.8% | Min weighted conviction per aprire trade |
| `sector_cap_pct` | 30% | Concentrazione massima per settore |
| `vix_risk_off_threshold` | — | VIX oltre cui regime = RISK_OFF |
| `vix_caution_threshold` | — | VIX oltre cui regime = CAUTION |

---

## Modalità di esecuzione

### Pipeline produzione giornaliera

```bash
# Tutti gli agenti, ticker da config/tickers.yaml
python -m src.run_pipeline

# Modalità light (solo technicals + sentiment)
python -m src.run_pipeline --mode light

# Review posizioni aperte (time-exit check)
python -m src.run_pipeline --mode review

# Ticker specifici, senza email
python -m src.run_pipeline AAPL MSFT NVDA --mode full --no-email
```

### Esecuzione interattiva (legacy)

```bash
# CLI interattiva — seleziona analisti, modello, date
python src/main.py --ticker AAPL,MSFT
```

### Backtest storico

```bash
python src/backtester.py \
  --tickers AAPL MSFT NVDA \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --initial-cash 1000000
```

### Walk-forward IS/OOS

```bash
python src/walk_forward_backtest.py \
  --tickers AAPL \
  --is-start 2023-01-01 \
  --is-end 2023-12-31 \
  --oos-start 2024-01-01 \
  --oos-end 2024-12-31
```

---

## Struttura del progetto — file per file

```
athanor-alpha/
├── src/
│   ├── main.py                    # Entry point legacy interattivo
│   ├── run_pipeline.py            # Entry point produzione giornaliera
│   ├── backtester.py              # CLI backtest storico
│   ├── walk_forward_backtest.py   # CLI walk-forward IS/OOS
│   ├── graph/
│   │   ├── state.py               # AgentState TypedDict + reducers
│   │   └── graph.py               # Builder grafo LangGraph
│   ├── agents/
│   │   ├── data_prefetch.py       # Batch fetch dati (no LLM)
│   │   ├── warren_buffett.py      # Agente: moat + FCF
│   │   ├── ben_graham.py          # Agente: net-net + Graham Number
│   │   ├── charlie_munger.py      # Agente: quality at fair price
│   │   ├── michael_burry.py       # Agente: deep value contrarian
│   │   ├── bill_ackman.py         # Agente: activist investor
│   │   ├── cathie_wood.py         # Agente: disruptive innovation
│   │   ├── phil_fisher.py         # Agente: scuttlebutt + margins
│   │   ├── peter_lynch.py         # Agente: PEG + 10-bagger
│   │   ├── mohnish_pabrai.py      # Agente: Dhandho + downside
│   │   ├── rakesh_jhunjhunwala.py # Agente: growth acceleration
│   │   ├── aswath_damodaran.py    # Agente: FCFF DCF + CAPM
│   │   ├── stanley_druckenmiller.py # Agente: macro + momentum
│   │   ├── fundamentals.py        # Agente: screening ratios
│   │   ├── valuation.py           # Agente: intrinsic value DCF
│   │   ├── growth_agent.py        # Agente: growth trends
│   │   ├── technicals.py          # Agente: pattern tecnico LLM
│   │   ├── breakout_momentum.py   # Agente: breakout + volume
│   │   ├── sentiment.py           # Agente: news + 10-Q MD&A
│   │   ├── macro_agent.py         # Agente: VIX + yield curve
│   │   ├── devils_advocate.py     # Agente: veto rule-based
│   │   ├── risk_manager.py        # Agente: VaR + correlazioni
│   │   ├── portfolio_manager.py   # Agente: aggregazione finale
│   │   └── time_exit_agent.py     # Agente: exit temporali
│   ├── data/
│   │   ├── prefetch.py            # DataPrefetcher singleton
│   │   ├── sec_edgar.py           # SEC EDGAR REST API + cache
│   │   ├── macro_fetcher.py       # VIX + yields macro data
│   │   ├── state_reader.py        # Accessor type-safe su state
│   │   ├── models.py              # Pydantic modelli dati
│   │   ├── cache.py               # TTL filesystem cache
│   │   ├── ttl_cache.py           # TTL constants
│   │   ├── signal_cache.py        # Cache segnali 24h per agente
│   │   ├── cache_guard.py         # Validità cache (price move, 8-K)
│   │   └── market_events.py       # Rilevamento eventi (earnings, split)
│   ├── db/
│   │   ├── init_db.py             # SQLite init + CRUD helpers
│   │   ├── models.py              # Pydantic modelli DB
│   │   └── schema.sql             # DDL tabelle
│   ├── feedback/
│   │   ├── outcome_tracker.py     # Calcolo rendimenti reali T+1/5/20d
│   │   ├── weight_adjuster.py     # EWA weight update
│   │   ├── prompt_injector.py     # Inject performance in LLM prompt
│   │   └── logger.py              # prediction_log_node
│   ├── utils/
│   │   ├── llm.py                 # call_llm() con retry + JSON mode
│   │   ├── analysts.py            # ANALYST_CONFIG registry
│   │   ├── progress.py            # Progress tracker terminale
│   │   ├── display.py             # Pretty-print output terminale
│   │   ├── email_report.py        # HTML digest email builder
│   │   ├── trade_levels.py        # ATR-based entry/SL/TP
│   │   ├── ema_filter.py          # EMA filter (disabilitato default)
│   │   ├── exit_checker.py        # Controllo segnali exit
│   │   ├── api_key.py             # Estrazione chiavi da .env
│   │   ├── ollama.py              # Integrazione Ollama locale
│   │   └── visualize.py           # Visualizzazione grafo PNG
│   ├── tools/
│   │   ├── api.py                 # Financial Datasets API wrapper
│   │   └── api_shim.py            # Shim compatibilità + state context
│   ├── llm/
│   │   ├── models.py              # Provider registry + ModelInfo
│   │   └── agent_router.py        # Per-agent model override
│   ├── indicators/
│   │   ├── technical_indicators.py # RSI, MACD, BB, ATR, Ichimoku...
│   │   ├── regime_detector.py      # Classificazione regime mercato
│   │   └── multi_timeframe.py      # Allineamento multi-timeframe
│   ├── alerts/
│   │   ├── email_sender.py        # SMTP sender con rate limit
│   │   └── templates.py           # Template HTML email
│   ├── monitor/
│   │   ├── daemon.py              # Daemon monitoring continuo
│   │   ├── price_checker.py       # Rilevamento movimento prezzi
│   │   ├── news_checker.py        # Monitoring news real-time
│   │   └── alert_builder.py       # Costruzione payload alert
│   ├── cli/
│   │   └── input.py               # CLI questionary interattiva
│   ├── portfolio/
│   │   └── manager.py             # Tracker posizioni aperte (SQLite)
│   └── backtesting/
│       ├── engine.py              # BacktestEngine principale
│       ├── portfolio.py           # Portfolio state (cash, positions)
│       ├── trader.py              # TradeExecutor (BUY/SELL/SHORT)
│       ├── controller.py          # AgentController per backtest
│       ├── metrics.py             # Sharpe, Sortino, max drawdown
│       ├── types.py               # TypedDict: AgentDecision, PerformanceMetrics
│       ├── valuation.py           # Calcolo valore portfolio + exposure
│       ├── output.py              # Formattazione risultati
│       ├── benchmarks.py          # SPY benchmark per alpha/beta
│       ├── walk_forward.py        # IS/OOS walk-forward analyzer
│       └── cli.py                 # CLI argparse backtest
├── tests/
│   ├── backtesting/
│   │   ├── test_portfolio.py
│   │   ├── test_execution.py
│   │   ├── test_metrics.py
│   │   ├── test_controller.py
│   │   ├── test_valuation.py
│   │   ├── test_results.py
│   │   └── integration/
│   │       ├── test_integration_long_only.py
│   │       ├── test_integration_long_short.py
│   │       └── test_integration_short_only.py
│   └── test_api_rate_limiting.py
├── config/
│   ├── tickers.yaml
│   └── risk_params.yaml
├── db/
│   ├── hedge_fund.db              # SQLite DB (creato al primo run)
│   └── schema.sql                 # DDL reference
├── logs/
│   └── runs.jsonl                 # Append-only log di ogni run
├── cache/
│   └── sec_edgar/                 # Cache filing SEC (7 giorni)
├── requirements.txt
└── .env                           # Variabili d'ambiente (non in git)
```

---

## Entry point

### `src/main.py` — Runner interattivo legacy

CLI interattiva che guida l'utente nella selezione di:
- Ticker da analizzare
- Periodo di analisi
- Analisti da attivare
- Modello LLM da usare

Costruisce `initial_state` via `make_initial_state()` e invoca il grafo LangGraph compilato. Usato per test manuali e debug. Funzioni principali: `run_hedge_fund()`, `create_workflow()`.

### `src/run_pipeline.py` — Pipeline produzione

Entry point principale per esecuzione quotidiana. Supporta tre modalità:
- **`full`**: tutti gli agenti attivi
- **`light`**: solo `technicals` + `sentiment` (più veloce, meno costoso)
- **`review`**: solo portfolio review + time-exit check

Flusso: carica ticker da YAML → costruisce state → invoca grafo → salva predizioni su DB → aggiorna pesi → invia digest email.

### `src/backtester.py` — Backtest CLI

Wrapper argparse per `BacktestEngine`. Accetta: `--tickers`, `--start-date`, `--end-date`, `--initial-cash`, `--model`, selezione analisti.

### `src/walk_forward_backtest.py` — Walk-forward CLI

Wrapper per `WalkForwardAnalyzer`. Richiede periodi IS e OOS separati. Stampa report di diagnostica overfitting.

---

## Grafo LangGraph (`src/graph/`)

### `src/graph/state.py`

Definisce `AgentState` — il contratto condiviso tra tutti i nodi del grafo.

**Struttura `AgentState`**:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # Log append-only
    data: Annotated[dict, _merge_dicts]        # Dati condivisi (merge parallelo-safe)
    metadata: Annotated[dict, _keep_first]     # Config immutabile (scritto una volta)
```

Il reducer `_merge_dicts` permette a nodi paralleli di scrivere su `data` senza race condition (deep merge). `_keep_first` garantisce che `metadata` non venga sovrascritto.

**`AgentOutput`** (Pydantic):
```python
class AgentOutput(BaseModel):
    signal: Literal["LONG", "SHORT", "NEUTRAL"]
    confidence: float         # 0.0 – 1.0
    expected_return: float    # stima rendimento atteso
    reasoning: str            # spiegazione LLM
```

**Factory `make_initial_state()`**: costruisce lo state iniziale con metadata (tickers, modello, flag nodi, run_id).

**Helpers**: `get_prefetched()`, `get_weight()`, `get_feedback()`, `set_analyst_signal()`.

### `src/graph/graph.py`

Costruisce e compila il grafo LangGraph a topologia statica.

**`_ALL_ANALYST_NODES`**: registry centralizzato di tutti i nodi analisti — mappa `node_name → (module_path, function_name)`.

**`_make_conditional_analyst()`**: wrapper che salta il nodo se non presente in `active_analyst_nodes` (da metadata). Permette di selezionare sottoinsiemi di analisti senza ricompilare il grafo.

**`_make_conditional_node()`**: wrapper generico per nodi infrastrutturali (devils_advocate, risk_manager, ecc.) — salta in base a flag metadata.

**`_import_or_stub()`**: importazione lazy con fallback stub se il modulo non esiste — permette deploy parziali.

Il grafo viene compilato una volta al momento dell'import (`compiled_graph`).

---

## Agenti (`src/agents/`)

### `src/agents/data_prefetch.py` — Prefetch dati

Nodo infrastrutturale senza LLM. Esegue batch fetch per tutti i ticker prima che gli analisti partano. Usa `DataPrefetcher.get_instance()` (singleton) per evitare fetch multipli nella stessa run.

Scrive in `state["data"]["prefetched_data"]`:
- OHLCV (1 anno, frequenza giornaliera)
- Financials (income statement, balance sheet, cash flow, 4 trimestri)
- Info aziendali (settore, dipendenti, market cap, descrizione)
- Holders (istituzionali, insider)
- News (7 giorni)
- Filing SEC (10-K, 10-Q, 8-K)

Scrive in `state["data"]["macro_data"]`: VIX, yield 10Y/3M/5Y, spread, trend.

### `src/agents/devils_advocate.py` — Devil's Advocate

Agente deterministico (nessuna chiamata LLM). Applica veto basato su:
- **Regime VIX**: moltiplicatore di sizing (0.4 in RISK_OFF → 1.0 in RISK_ON)
- **Coerenza segnale**: vetoed se segnali fortemente contrastanti senza sufficiente consensus
- **Soglie di confidence**: filtra segnali sotto threshold minima

Scrive `state["data"]["devils_advocate_output"]`.

### `src/agents/risk_manager.py` — Risk Manager

Agente deterministico. Calcola:
- **Matrice di correlazione** tra ticker in portafoglio
- **VaR-95% parametrico** (distribuzione normale, finestra 252gg)
- **Concentrazione settoriale** vs cap (30%)
- **Max drawdown** posizioni aperte
- **ATR-based trade levels**: entry, stop-loss (±1×ATR14), take-profit (±2×ATR14, R/R 1:2)

Legge `risk_params.yaml` per soglie. Scrive `state["data"]["risk_output"]`.

### `src/agents/portfolio_manager.py` — Portfolio Manager

Nodo di aggregazione finale. Include 1 chiamata LLM per reasoning narrativo.

**Logica di aggregazione**:
1. Raggruppa segnali per dimensione (FUNDAMENTALS/TECHNICAL/SENTIMENT/MACRO)
2. Score per dimensione = media pesata segnali × confidence × agent_weight
3. Score finale = media delle 4 dimensioni (pesi uguali 25% ciascuna)
4. Se `|score| < NET_SCORE_THRESHOLD` (0.25) → NEUTRAL
5. Se `consensus_ratio < MIN_CONSENSUS_RATIO` (0.65) → NEUTRAL
6. Se `|weighted_conviction| < WC_THRESHOLD` (0.8%) → no trade

**Position sizing**:
- Vol-targeting: `size = target_vol / realized_vol`
- Kelly: `size *= kelly_fraction` (0.25)
- Cap: `min(size, MAX_SINGLE_POSITION)` (20%)
- Regime cap: RISK_ON 20%, CAUTION 12%, RISK_OFF 5%

Scrive `state["data"]["portfolio_recommendations"]`, appende a `logs/runs.jsonl`, salva su SQLite.

### `src/agents/time_exit_agent.py` — Time Exit

Attivo solo in modalità `review`. Controlla posizioni aperte da `positions.json`. Se una posizione è aperta da ≥4 sessioni senza segnale di chiusura, invia alert email.

---

## Agenti filosofici — dettaglio

### Dimensione FUNDAMENTALS

| Agente | File | Filosofia | Metriche chiave |
|--------|------|-----------|-----------------|
| **Warren Buffett** | `warren_buffett.py` | Moat, FCF, qualità business | PE, ROE, FCF yield, earnings consistency, book value growth |
| **Ben Graham** | `ben_graham.py` | Net-net, Graham Number, margin of safety | Current ratio, P/B < 1.2, Debt/Equity, Graham Number vs prezzo |
| **Charlie Munger** | `charlie_munger.py` | Quality at fair price, ROIC | ROIC > 15%, moat sostenibile, management quality |
| **Michael Burry** | `michael_burry.py` | Deep value contrarian, tail risk | Distressed asset ratio, debt coverage, market dislocation score |
| **Bill Ackman** | `bill_ackman.py` | Activist, concentrated bets | Quality score, financial discipline, activism potential |
| **Cathie Wood** | `cathie_wood.py` | Disruptive innovation, TAM expansion | R&D/Revenue ratio, growth acceleration, TAM addressable |
| **Phil Fisher** | `phil_fisher.py` | Scuttlebutt, margin stability | Margin trend (10 trimestri), profit quality, management depth |
| **Peter Lynch** | `peter_lynch.py` | PEG ratio, 10-bagger | PEG < 1.0, earnings growth acceleration, market cap vs growth |
| **Mohnish Pabrai** | `mohnish_pabrai.py` | Dhandho, asimmetria rischio | Margin of safety, debt basso, profitability stabile |
| **Rakesh Jhunjhunwala** | `rakesh_jhunjhunwala.py` | Growth acceleration, conviction | Revenue growth trend, profit growth, sector momentum |
| **Aswath Damodaran** | `aswath_damodaran.py` | FCFF DCF, CAPM, story+numbers | DCF intrinsic value, WACC, relative multiples |
| **Stanley Druckenmiller** | `stanley_druckenmiller.py` | Macro themes, momentum, insider | Macro trend score, insider trades, sentiment shift |
| **Fundamentals** | `fundamentals.py` | Screening sistematico ratios | Tutti i ratio finanziari (profitability, leverage, liquidity, efficiency) |
| **Valuation** | `valuation.py` | Owner earnings DCF | Owner earnings normalizzati, DCF multi-stage, normalized P/E |
| **Growth** | `growth_agent.py` | Accelerazione crescita | CAGR revenue/EPS 3Y, accelerazione marginale, forward estimates |

### Dimensione TECHNICAL

| Agente | File | Specializzazione | Indicatori |
|--------|------|-----------------|------------|
| **Technicals** | `technicals.py` | Pattern LLM + supporti/resistenze | RSI-14, MACD (12/26/9), Bollinger Bands (20,2σ), ATR-14, VWAP, OBV, Ichimoku (9/26/52), Fibonacci |
| **Breakout Momentum** | `breakout_momentum.py` | Breakout + volume surge | Volume surge (>2× media 20gg), ATR expansion, prossimità 52-week high, ROC, RSI penalty |

### Dimensione SENTIMENT

| Agente | File | Fonti dati | Output |
|--------|------|-----------|--------|
| **Sentiment** | `sentiment.py` | yfinance news (7gg), SEC 8-K, 10-Q MD&A | Classificazione news (bullish/bearish/neutral), urgency score, tone MD&A |

### Dimensione MACRO

| Agente | File | Dati | Output |
|--------|------|------|--------|
| **Macro** | `macro_agent.py` | VIX, 10Y/3M yield, yield spread, trend | Regime macroeconomico, sensitivity scaling per ticker |

---

## Data layer (`src/data/`)

### `src/data/prefetch.py` — DataPrefetcher

Singleton che coordina tutti i fetch. Prima chiamata esegue il fetch; chiamate successive restituiscono la cache in memoria.

Struttura `TickerPayload` in `state["data"]["prefetched_data"][ticker]`:
```python
{
    "ohlcv": pd.DataFrame,          # OHLCV 1 anno
    "financials": {
        "income_stmt": pd.DataFrame,
        "balance_sheet": pd.DataFrame,
        "cash_flow": pd.DataFrame,
    },
    "info": dict,                   # yfinance .info
    "holders": dict,                # institutional, insider
    "news": list[dict],             # 7 giorni
    "sec_filings": {
        "10-K": list,
        "10-Q": list,
        "8-K": list,
    }
}
```

### `src/data/sec_edgar.py` — SECFetcher

Interfaccia con SEC EDGAR Public API (`data.sec.gov`). Rate limit interno: max 10 req/sec. Cache disco: 7 giorni (`cache/sec_edgar/`).

Funzioni principali:
- `get_ticker_cik(ticker)` — risolve ticker → CIK (Codice identificativo SEC)
- `fetch_filings(ticker, form_types)` — recupera lista filing per tipo (10-K, 10-Q, 8-K)
- `fetch_submission_history(cik)` — storico completo submission

Ticker crypto vengono automaticamente skippati (nessun CIK).

### `src/data/macro_fetcher.py` — MacroFetcher

Scarica tramite yfinance:
- `^VIX` — CBOE Volatility Index
- `^TNX` — 10Y Treasury yield
- `^IRX` — 3M T-bill yield
- `^FVX` — 5Y Treasury yield

Calcola campi derivati:
- `yield_spread_10y3m` = TNX - IRX (indicatore inversione curva)
- `rate_trend_20d` = tendenza tassi ultimi 20 giorni
- `vix_trend_10d` = tendenza VIX ultimi 10 giorni

### `src/data/state_reader.py` — Accessor type-safe

Helper functions che leggono `state["data"]["prefetched_data"]` con fallback sicuri:
- `get_ohlcv(state, ticker)` → `pd.DataFrame | None`
- `get_fundamentals(state, ticker)` → dict
- `get_info(state, ticker)` → dict
- `get_news(state, ticker)` → list
- `get_sec_filings(state, ticker)` → dict

### `src/data/models.py` — Modelli Pydantic

```python
class FinancialMetrics(BaseModel): ...    # Tutti i ratio finanziari
class LineItem(BaseModel): ...            # Voce conto economico/SP
class Price(BaseModel): ...               # OHLCV singolo giorno
class CompanyNews(BaseModel): ...         # Articolo news
class InsiderTrade(BaseModel): ...        # Trade insider
class MacroData(BaseModel): ...           # Snapshot macro
```

### `src/data/cache.py` e `src/data/ttl_cache.py` — TTL Cache

Cache filesystem con TTL configurabile:
- **Prezzi**: 1 ora (`TTL_PRICES = 3600`)
- **Financials**: 7 giorni (`TTL_FINANCIALS = 604800`)

### `src/data/signal_cache.py` e `src/data/cache_guard.py` — Signal Cache

Evita chiamate LLM ridondanti entro 24h. `cache_guard.py` invalida la cache se:
- Il prezzo si è mosso oltre una soglia
- È arrivato un nuovo 8-K SEC (event materiale)

### `src/data/market_events.py` — Market Events

Rileva eventi che invalidano cache o richiedono analisi fresca:
- Earnings release (da calendar yfinance)
- Nuovi filing 8-K
- Stock split

---

## Database (`src/db/`)

### `src/db/init_db.py`

Inizializzazione idempotente SQLite. Abilita WAL mode e foreign keys. CRUD helpers per tutte le tabelle.

```bash
python -m src.db.init_db   # crea db/hedge_fund.db se non esiste
```

**Funzioni CRUD**:
- `init_db()` — crea tabelle se non esistono
- `get_connection()` — context manager SQLite
- `insert_prediction(run_id, agent_id, ticker, signal, confidence, reasoning)`
- `get_recent_predictions(n_days)` — predizioni ultime N giorni
- `insert_outcome(prediction_id, ticker, actual_return_1d/5d/20d)`
- `get_agent_accuracy(agent_id)` — accuracy media per agente
- `update_agent_weight(agent_id, ticker, weight)` — aggiorna peso EWA
- `insert_pipeline_run(run_id, tickers, status)` — log run
- `complete_pipeline_run(run_id, status, error_msg)` — chiude run

### `src/db/schema.sql`

DDL di riferimento (le tabelle sono create anche via Python in `init_db.py`).

---

## Schema database SQLite

### Tabella `predictions`

| Campo | Tipo | Note |
|-------|------|------|
| `id` | INTEGER PK | autoincrement |
| `run_id` | TEXT | UUID run pipeline |
| `agent_id` | TEXT | nome agente (es. `warren_buffett`) |
| `ticker` | TEXT | simbolo ticker |
| `signal` | TEXT | BUY / SELL / HOLD |
| `confidence` | REAL | 0.0 – 1.0 |
| `reasoning_hash` | TEXT | SHA256 del reasoning LLM |
| `timestamp` | DATETIME | UTC |

Indici: `run_id`, `(agent_id, ticker)`, `timestamp`.

### Tabella `outcomes`

| Campo | Tipo | Note |
|-------|------|------|
| `id` | INTEGER PK | autoincrement |
| `prediction_id` | INTEGER FK | → predictions.id |
| `ticker` | TEXT | |
| `actual_return_1d` | REAL | rendimento T+1 |
| `actual_return_5d` | REAL | rendimento T+5 |
| `actual_return_20d` | REAL | rendimento T+20 |
| `window` | TEXT | "1d" / "5d" / "20d" |
| `evaluated_at` | DATETIME | UTC |

### Tabella `agent_weights`

| Campo | Tipo | Note |
|-------|------|------|
| `agent_id` | TEXT | |
| `ticker` | TEXT | |
| `weight` | REAL | ≥ 0, aggiornato via EWA |
| `updated_at` | DATETIME | |

UNIQUE(`agent_id`, `ticker`). Default weight = 1.0.

### Tabella `agent_hyperparams`

| Campo | Tipo | Note |
|-------|------|------|
| `agent_id` | TEXT | |
| `param_name` | TEXT | |
| `value` | TEXT | serializzato JSON |
| `updated_at` | DATETIME | |

UNIQUE(`agent_id`, `param_name`).

### Tabella `pipeline_runs`

| Campo | Tipo | Note |
|-------|------|------|
| `run_id` | TEXT PK | UUID |
| `started_at` | DATETIME | |
| `finished_at` | DATETIME | NULL se in corso |
| `status` | TEXT | running / completed / failed |
| `tickers` | TEXT | JSON array ticker |
| `error_msg` | TEXT | NULL se ok |

---

## Feedback loop (`src/feedback/`)

### `src/feedback/outcome_tracker.py`

Scarica prezzi storici reali dopo ogni run e calcola rendimenti effettivi a T+1, T+5, T+20 giorni dalla data della predizione. Popola la tabella `outcomes`. Stampa ranking agenti per accuracy.

### `src/feedback/weight_adjuster.py`

Aggiorna i pesi degli agenti usando **Exponentially Weighted Average (EWA)**:

```
w_new = α × accuracy_recente + (1 − α) × w_old
```

Con `α = 0.5`. Aggiornamento scritto su tabella `agent_weights`. Gli agenti con accuracy superiore alla media ricevono peso maggiore nel portfolio_manager.

### `src/feedback/prompt_injector.py`

Inietta statistiche di performance passata nel prompt LLM degli agenti (opzionale). Permette all'agente di "sapere" quanto è stato accurato nelle ultime N settimane.

### `src/feedback/logger.py`

Nodo `prediction_log_node` nel grafo. Legge `analyst_signals` e `portfolio_recommendations` dallo state e li persiste su SQLite via `init_db.py`.

---

## Utilità (`src/utils/`)

### `src/utils/llm.py` — call_llm()

Wrapper centrale per tutte le chiamate LLM. Features:
- **Retry automatico**: fino a 6 tentativi con backoff esponenziale
- **JSON mode**: attivato se il modello lo supporta (`has_json_mode()`)
- **Fallback JSON**: parsing regex se il modello non rispetta il formato
- **Model config**: legge da `state["metadata"]["model_config"]` o default claude-sonnet

```python
response = call_llm(
    state=state,
    prompt="...",
    system_prompt="...",
    response_format="json",
    agent_id="warren_buffett"
)
```

### `src/utils/analysts.py` — ANALYST_CONFIG

Registry centralizzato. Ogni voce:
```python
ANALYST_CONFIG = {
    "warren_buffett": {
        "display_name": "Warren Buffett",
        "philosophy": "Value investing, moat...",
        "agent_fn": warren_buffett_agent,
        "order": 1,
        "dimension": "FUNDAMENTALS"
    },
    ...
}
```

Usato da: CLI (selezione analisti), graph.py (registrazione nodi), display.py (output), portfolio_manager.py (mapping dimensioni).

### `src/utils/trade_levels.py`

Calcola livelli di trading basati su ATR-14:
- **Entry**: close corrente
- **Stop-Loss**: `entry ± 1 × ATR14` (direction-aware)
- **Take-Profit**: `entry ∓ 2 × ATR14` (R/R = 1:2)

### `src/utils/ema_filter.py`

Filtro EMA opzionale (disabilitato per default). Audit interno ha mostrato −5% win rate quando attivo. Mantenuto per A/B testing futuro.

### `src/utils/email_report.py`

Costruisce email HTML con:
- Tabella voti agenti per ticker
- Raccomandazioni portfolio (entry/SL/TP)
- Metriche di rischio (VaR, correlazioni)
- Heatmap conviction

Invio via SMTP (configurato in `.env`).

---

## API & Tools (`src/tools/`)

### `src/tools/api.py` — Financial Datasets API

Wrapper per `api.financialdatasets.ai`. Richiede API key in `.env`.

Funzioni:
- `get_prices(ticker, start, end)` — prezzi storici
- `get_financial_metrics(ticker)` — ratio pre-calcolati
- `get_company_news(ticker, days)` — news con sentiment
- `search_line_items(ticker, items)` — voci bilancio custom
- `get_market_cap(ticker)` — market cap corrente
- `get_insider_trades(ticker)` — transazioni insider

Rate limiting: backoff 60s → 90s → 120s su 429 (max 3 retry).

### `src/tools/api_shim.py`

Layer di compatibilità che wrappa `api.py`. Ogni agente chiama `register_state(state)` prima di usare le funzioni API, permettendo al shim di iniettare context (ticker, date) automaticamente.

---

## LLM routing (`src/llm/`)

### `src/llm/models.py`

Registry dei provider LLM supportati:

| Provider | Modelli |
|----------|---------|
| **Anthropic** | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1` |
| **Groq** | modelli Llama/Mixtral via API Groq |
| **DeepSeek** | `deepseek-chat`, `deepseek-reasoner` |
| **Ollama** | modelli locali (zero cost, no API key) |

`ModelInfo` contiene: `has_json_mode`, `temperature`, `context_window`, `provider`.

### `src/llm/agent_router.py`

Permette override del modello LLM per singolo agente. Ad esempio: usare `claude-opus` per Damodaran (analisi complessa) e `claude-haiku` per sentiment (veloce e cheap).

---

## Indicatori tecnici (`src/indicators/`)

### `src/indicators/technical_indicators.py`

Libreria completa calcolata da DataFrame OHLCV:

| Indicatore | Parametri | Output |
|-----------|-----------|--------|
| RSI | periodo 14 | 0–100 |
| EMA | 12, 26 | prezzo |
| MACD | 12/26/9 | MACD line, signal, histogram |
| Bollinger Bands | 20, 2σ | upper, middle, lower |
| ATR | 14 | volatilità assoluta |
| Stochastic | %K, %D | 0–100 |
| ADX | 14 | forza trend 0–100 |
| VWAP | giornaliero | prezzo |
| OBV | — | volume cumulativo |
| Ichimoku | 9/26/52 | cloud, tenkan, kijun |
| Fibonacci | — | livelli 23.6/38.2/50/61.8% |

### `src/indicators/regime_detector.py`

Classifica il regime di mercato in base a:
- **ADX** > 25 → trending (up/down)
- **Volatilità** vs media storica → ranging vs volatile
- **Momentum** 20gg

Output: `TRENDING_UP`, `TRENDING_DOWN`, `RANGING`, `VOLATILE`.

### `src/indicators/multi_timeframe.py`

Allinea segnali su timeframe giornaliero e intraday (5 minuti). Calcola confluenza: segnale più forte quando entrambi i timeframe concordano.

---

## Alert & Monitoring (`src/alerts/`, `src/monitor/`)

### `src/alerts/email_sender.py`

Invio SMTP con:
- Rate limit: max 1 alert per ticker ogni 60 minuti
- Formato: multipart MIME (plain text + HTML)
- Supporto Gmail e SMTP custom

### `src/monitor/daemon.py`

Daemon che gira in background (loop continuo con `schedule`). Monitora:
- Movimenti di prezzo significativi (> soglia configurabile in %)
- Nuove news per i ticker monitorati
- Scadenze posizioni aperte

### `src/monitor/price_checker.py`

Rileva movimenti prezzo oltre soglia in finestra temporale configurabile. Trigger → `alert_builder.py` → `email_sender.py`.

---

## CLI (`src/cli/`)

### `src/cli/input.py`

Interfaccia `questionary` interattiva. Prompt:
1. Selezione analisti (checkbox multipli)
2. Scelta modello LLM (select)
3. Date range (input validato)
4. Ticker manuali o da YAML

Restituisce dizionario strutturato usato da `main.py`.

---

## Portfolio (`src/portfolio/`)

### `src/portfolio/manager.py`

Tracker posizioni aperte persiste su SQLite (tabella `positions` — creata separatamente o via `positions.json`).

Funzioni:
- `get_open_positions()` — lista posizioni aperte
- `get_position_by_ticker(ticker)` — singola posizione
- Tracking: entry price, data apertura, side (LONG/SHORT), size

---

## Backtesting (`src/backtesting/`)

### `src/backtesting/engine.py` — BacktestEngine

Loop principale: itera giorno per giorno nel range IS, per ogni giorno:
1. Recupera OHLCV storico fino a quel giorno
2. Invoca `AgentController.run_agent()` → ottiene segnali
3. Esegue trade via `TradeExecutor`
4. Calcola valore portfolio
5. Appende snapshot

Output: `PerformanceMetrics`.

### `src/backtesting/portfolio.py`

Stato portfolio durante backtest: cash, posizioni (ticker → size/cost), margine short. Supporta long + short. Funzioni: `add_position()`, `close_position()`, `calculate_value(prices)`.

### `src/backtesting/metrics.py`

Calcolo metriche con 252 giorni trading/anno, risk-free rate 4.34%:

| Metrica | Formula |
|---------|---------|
| **Sharpe** | `(mean_return − rf) / std_return × √252` |
| **Sortino** | `(mean_return − rf) / downside_std × √252` |
| **Max Drawdown** | `max(cummax − cumulative) / cummax` |
| **Calmar** | `annualized_return / max_drawdown` |

### `src/backtesting/walk_forward.py` — WalkForwardAnalyzer

Esegue due passaggi BacktestEngine indipendenti (IS e OOS). Report diagnostico overfitting:
- **Sharpe decay**: `sharpe_IS / sharpe_OOS` (ideale < 2)
- **Return retention**: `return_OOS / return_IS × 100%` (ideale > 50%)
- **Calmar stability**: confronto Calmar IS vs OOS

### `src/backtesting/benchmarks.py`

Scarica SPY e calcola:
- **Benchmark return** buy-and-hold
- **Alpha**: `strategy_return − beta × benchmark_return`
- **Beta**: regressione rendimenti giornalieri

---

## Test (`tests/`)

```bash
pytest tests/                              # tutti i test
pytest tests/backtesting/                  # solo unit test backtesting
pytest tests/backtesting/integration/      # integration test E2E
```

| File | Scope |
|------|-------|
| `test_portfolio.py` | Cash management, position tracking, P&L calc |
| `test_execution.py` | BUY/SELL/SHORT/COVER fill logic |
| `test_metrics.py` | Formula Sharpe/Sortino/drawdown |
| `test_controller.py` | Agent callback invocation |
| `test_valuation.py` | Portfolio value + exposure snapshot |
| `test_results.py` | Output formatting, report generation |
| `integration/test_integration_long_only.py` | E2E backtest long-only |
| `integration/test_integration_long_short.py` | E2E backtest long+short |
| `integration/test_integration_short_only.py` | E2E backtest short-only |
| `test_api_rate_limiting.py` | Retry backoff 60→90→120s su 429 |

---

## Config (`config/`)

| File | Contenuto |
|------|-----------|
| `tickers.yaml` | Lista ticker monitorati (equity + crypto + benchmark) |
| `risk_params.yaml` | Parametri risk management (vol target, Kelly, soglie VIX, cap settoriali) |

---

## Database & log runtime (`db/`, `logs/`, `cache/`)

| Path | Contenuto | Note |
|------|-----------|------|
| `db/hedge_fund.db` | SQLite database principale | Creato da `python -m src.db.init_db` |
| `db/schema.sql` | DDL reference | Non eseguito automaticamente |
| `logs/runs.jsonl` | Append-only log JSON lines di ogni run pipeline | Una riga per run |
| `cache/sec_edgar/` | Cache disco filing SEC EDGAR | TTL 7 giorni, auto-invalidato |

---

## Topologia del grafo

```
START
  │
  ▼
data_prefetch          ← batch fetch OHLCV + financials + SEC + macro (no LLM)
  │
  ├──────────────────────────────────────────────────────────────────┐
  │  Fan-out parallelo (nodi saltati se non in active_analyst_nodes) │
  │                                                                  │
  ├─ warren_buffett     ├─ ben_graham       ├─ charlie_munger        │
  ├─ michael_burry      ├─ bill_ackman      ├─ cathie_wood           │
  ├─ phil_fisher        ├─ peter_lynch      ├─ mohnish_pabrai        │
  ├─ damodaran          ├─ druckenmiller                             │
  ├─ fundamentals       ├─ valuation        ├─ growth_agent          │
  ├─ technicals         ├─ breakout_momentum                         │
  ├─ sentiment          ├─ macro_agent                               │
  └──────────────────────────────────────────────────────────────────┘
  │
  ▼
devils_advocate        ← veto rule-based, VIX regime sizing (skip flag disponibile)
  │
  ▼
risk_manager           ← VaR, correlazioni, trade levels ATR (skip flag disponibile)
  │
  ▼
portfolio_manager      ← aggregazione 4D + sizing + 1× LLM call reasoning
  │
  ▼
prediction_log         ← scrivi su SQLite (skip flag disponibile)
  │
  ▼
time_exit              ← alert email posizioni ≥4gg (solo mode=review)
  │
  ▼
END
```

**Mutazioni state** per nodo:

| Nodo | Scrive in |
|------|-----------|
| `data_prefetch` | `data.prefetched_data`, `data.macro_data` |
| Analyst nodes | `data.analyst_signals[ticker][agent_id]` (merge parallelo-safe) |
| `devils_advocate` | `data.devils_advocate_output` |
| `risk_manager` | `data.risk_output` |
| `portfolio_manager` | `data.portfolio_recommendations`, `logs/runs.jsonl`, SQLite |
| `prediction_log` | SQLite `predictions` |

---

## Dimensioni di analisi (FIX C2)

Il sistema aggrega segnali su 4 dimensioni ortogonali con peso uguale (25% ciascuna):

| Dimensione | Agenti inclusi | Peso |
|-----------|---------------|------|
| **FUNDAMENTALS** | warren_buffett, ben_graham, charlie_munger, michael_burry, bill_ackman, cathie_wood, phil_fisher, peter_lynch, mohnish_pabrai, damodaran, fundamentals, valuation, growth | 25% |
| **TECHNICAL** | technicals, breakout_momentum | 25% |
| **SENTIMENT** | sentiment | 25% |
| **MACRO** | macro_agent, druckenmiller | 25% |

Il consensus ratio deve essere ≥ 65% (almeno 3 dimensioni su 4 concordi) per generare una raccomandazione non-NEUTRAL.

---

## Sistema di risk management

### Costanti chiave (`portfolio_manager.py`)

| Costante | Valore | Descrizione |
|----------|--------|-------------|
| `NET_SCORE_THRESHOLD` | 0.25 | Min \|score\| per classificare bullish/bearish |
| `MIN_CONSENSUS_RATIO` | 0.65 | Min fraction di dimensioni concordi |
| `WC_THRESHOLD` | 0.008 | Min \|weighted conviction\| per aprire trade (0.8%) |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly fraction |
| `MAX_SINGLE_POSITION` | 20.0% | Hard cap posizione singola |

### Regime macro e position cap

| Regime | Condizione VIX | Max posizione |
|--------|---------------|--------------|
| `RISK_ON` | < soglia caution | 20% |
| `CAUTION` | tra soglie | 12% |
| `RISK_OFF` | > soglia risk-off | 5% |

### ATR Trade Levels

```
Entry  = close corrente
SL     = entry ± 1 × ATR(14)       # stop loss
TP     = entry ∓ 2 × ATR(14)       # take profit   → R/R = 1:2
```

---

## Portfolio manager — logica di aggregazione

```
Per ogni ticker:

1. Raggruppa analyst_signals per dimensione
   FUNDAMENTALS → [warren, ben, charlie, ...]
   TECHNICAL    → [technicals, breakout]
   SENTIMENT    → [sentiment]
   MACRO        → [macro, druckenmiller]

2. Score per dimensione:
   dim_score = Σ(signal_value × confidence × agent_weight) / Σ(agent_weight)
   dove signal_value: LONG=+1, SHORT=-1, NEUTRAL=0

3. Score finale:
   final_score = mean(dim_scores)   # 25% per dimensione

4. Filtri:
   if |final_score| < 0.25  → NEUTRAL
   if consensus_ratio < 0.65 → NEUTRAL
   if |weighted_conviction| < 0.008 → no trade

5. Position sizing:
   size = target_daily_vol / realized_vol_20d
   size = size × kelly_fraction (0.25)
   size = min(size, regime_cap)
   size = min(size, 20%)
```

---

## Backtesting e walk-forward

### Backtest storico

Il `BacktestEngine` simula il portfolio day-by-day nel periodo storico:
- Fill al prezzo di chiusura del giorno
- Supporto long + short
- Commissioni configurabili
- Tracking cash + posizioni

### Walk-forward IS/OOS

Diagnostica overfitting tramite due periodi separati:

```
In-Sample (IS):    training period  → parametri ottimizzati qui
Out-of-Sample (OOS): test period    → verifica generalizzazione

Metriche diagnostica:
  Sharpe decay     = sharpe_IS / sharpe_OOS     (< 2.0 = sano)
  Return retention = return_OOS / return_IS     (> 50% = sano)
  Calmar stability = calmar_IS vs calmar_OOS    (simili = sano)
```

---

## Feedback loop e weight adjustment

```
Run pipeline
    │
    ▼
prediction_log → salva signal + confidence su SQLite
    │
    ▼  (dopo T+1, T+5, T+20 giorni)
outcome_tracker → scarica prezzi reali, calcola rendimenti effettivi
    │
    ▼
weight_adjuster → EWA update
    w_new = 0.5 × accuracy_recente + 0.5 × w_old
    │
    ▼
agent_weights table → aggiornato
    │
    ▼  (run successiva)
portfolio_manager → usa pesi aggiornati per aggregazione dimensionale
```

---

## Variabili d'ambiente

Crea `.env` nella root del progetto:

```bash
# Provider LLM (almeno uno richiesto)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=...

# Financial Datasets API (opzionale — alcuni agenti lo usano)
FINANCIAL_DATASETS_API_KEY=...

# Email alerts (opzionale)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=app-password
ALERT_RECIPIENT=recipient@email.com

# Modello default (opzionale, default: claude-sonnet-4-6)
DEFAULT_LLM_MODEL=claude-sonnet-4-6

# Ollama (opzionale, per uso locale zero-cost)
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Limitazioni e stato dell'arte

### Stato attuale

- **~85 file Python**, ~20.000 LOC
- **15 agenti filosofici** + 4 agenti infrastrutturali
- **Pipeline produzione** funzionante (`run_pipeline.py`)
- **Backtesting engine** completo con walk-forward IS/OOS
- **Feedback loop** EWA operativo
- **Cache multi-livello**: filesystem TTL (prezzi/financials), signal cache 24h, cache SEC 7gg
- **11 ticker** monitorati default (8 equity + 3 crypto + SPY)
- **6 provider LLM** supportati (Anthropic, OpenAI, Groq, DeepSeek, Ollama, + qualsiasi OpenAI-compatible)

### Limitazioni note

1. **Nessuna esecuzione reale**: il sistema produce solo raccomandazioni, non connette a broker.
2. **Latenza**: con tutti gli analisti attivi e LLM cloud, una run completa su 11 ticker può richiedere 3–8 minuti.
3. **Costi LLM**: in modalità `full` con claude-opus, stimare ~$0.50–2.00 per run completa. Usare `light` o `haiku` per ridurre i costi.
4. **yfinance instabilità**: l'API yfinance non è ufficiale. Dati financials storici possono avere gap o inconsistenze.
5. **Crypto limitato**: nessun dato SEC per crypto, sentiment limitato a news yfinance.
6. **EMA filter disabilitato**: audit interno ha mostrato -5% win rate — mantenuto per ricerca futura.
7. **Walk-forward**: diagnostica di overfitting disponibile ma non automaticamente usata per ottimizzare i parametri.
8. **No paper trading**: manca integrazione con simulatore di mercato real-time per validazione live.

---

*Athanor Alpha — progetto educativo open-source. Non usare per decisioni finanziarie reali.*
