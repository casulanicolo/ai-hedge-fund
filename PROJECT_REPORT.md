# Athanor Alpha — Report Tecnico Completo

> Questo documento descrive in modo esaustivo l'intero sistema Athanor Alpha così com'è stato costruito. È pensato per essere fornito come contesto iniziale in una nuova sessione di chat con Claude.

---

## 1. Cos'è Athanor Alpha

Athanor Alpha è un **sistema algoritmico di hedge fund multi-agente**, costruito in Python usando LangGraph. Simula 15 filosofie di investimento distinte (Warren Buffett, Ben Graham, Michael Burry, ecc.), ognuna implementata come agente LLM separato. Gli agenti analizzano titoli azionari in parallelo, producono segnali BUY/SELL/HOLD con confidence score, che vengono aggregati da un Risk Manager e un Portfolio Manager per produrre raccomandazioni operative con entry, stop-loss e take-profit.

Il sistema include una pipeline di produzione giornaliera, un motore di backtesting completo, un framework di walk-forward IS/OOS testing, un database SQLite per il logging e un loop di feedback che aggiorna dinamicamente i pesi degli agenti in base alle performance storiche.

**Dimensione del codebase**: ~20.138 righe di Python, distribuite in 85 file.

---

## 2. Struttura delle directory

```
athanor-alpha/
├── src/
│   ├── agents/              # 21 file — tutti gli agenti (15 filosofi + infra)
│   ├── backtesting/         # 12 file — engine, portfolio, metriche, IS/OOS
│   ├── data/                # 7 file  — prefetch, cache, modelli dati, SEC EDGAR
│   ├── db/                  # 3 file  — schema SQLite, CRUD, modelli DB
│   ├── feedback/            # 4 file  — outcome tracker, weight adjuster, logger
│   ├── graph/               # 2 file  — AgentState LangGraph, graph builder
│   ├── indicators/          # 3 file  — indicatori tecnici, regime detector, multi-timeframe
│   ├── alerts/              # 2 file  — email sender, template HTML
│   ├── monitor/             # 4 file  — daemon di monitoraggio, price/news checker
│   ├── utils/               # 9 file  — display, LLM helper, email report, progress, ollama
│   ├── tools/               # 2 file  — api.py, api_shim.py
│   ├── cli/                 # 1 file  — input.py (questionary CLI)
│   ├── llm/                 # 1 file  — models.py (LLM provider registry)
│   ├── main.py              # Entry point: run_hedge_fund() + grafo LangGraph
│   ├── run_pipeline.py      # Pipeline di produzione giornaliera
│   ├── backtester.py        # CLI per backtest singolo periodo
│   └── walk_forward_backtest.py  # CLI per IS/OOS walk-forward test
├── config/
│   └── tickers.yaml         # Lista ticker monitorati
├── db/
│   ├── schema.sql           # Schema SQLite (5 tabelle)
│   └── hedge_fund.db        # Database runtime (creato automaticamente)
└── requirements.txt
```

---

## 3. Agenti (`src/agents/`)

### 3.1 Agenti filosofici (15 agenti)

Ogni agente:
1. Legge i dati prefetchati dallo stato LangGraph tramite `api_shim.py`
2. Calcola score multi-dimensionali (growth, valuation, risk, moat, ecc.)
3. Chiama il LLM con un system prompt nel "tono" del filosofo
4. Restituisce `{signal: "bullish"/"bearish"/"neutral", confidence: 0–100, reasoning: str}`
5. Scrive il risultato come messaggio JSON nello stato LangGraph

| File | Funzione | LOC | Filosofia |
|------|----------|-----|-----------|
| `warren_buffett.py` | `warren_buffett_agent()` | 827 | Moat, earnings consistency, intrinsic value, margin of safety |
| `charlie_munger.py` | `charlie_munger_agent()` | 857 | Business quality, management, competitive advantage, mental models |
| `rakesh_jhunjhunwala.py` | `rakesh_jhunjhunwala_agent()` | 708 | Growth acceleration, profitability, contrarian, conviction sizing |
| `phil_fisher.py` | `phil_fisher_agent()` | 599 | Margin stability, scuttlebutt research, quality at reasonable price |
| `stanley_druckenmiller.py` | `stanley_druckenmiller_agent()` | 597 | Growth/momentum, macro themes, insider activity, sentiment |
| `peter_lynch.py` | `peter_lynch_agent()` | 503 | PEG ratio, earnings growth, 10-bagger hunting, everyday businesses |
| `valuation.py` | `valuation_analyst_agent()` | 491 | Owner earnings DCF, intrinsic value, P/E normalization |
| `bill_ackman.py` | `bill_ackman_agent()` | 469 | Business quality, financial discipline, activism potential |
| `sentiment.py` | `sentiment_agent()` | 467 | News classification, urgency detection, sentiment scoring, 8-K events |
| `cathie_wood.py` | `cathie_wood_agent()` | 437 | Disruptive innovation, R&D intensity, TAM expansion, exponential growth |
| `aswath_damodaran.py` | `aswath_damodaran_agent()` | 417 | FCFF DCF, CAPM, relative valuation, story+numbers discipline |
| `michael_burry.py` | `michael_burry_agent()` | 371 | Deep value, tail risk, market dislocation, contrarian shorts |
| `mohnish_pabrai.py` | `mohnish_pabrai_agent()` | 360 | Dhandho investing, downside protection, asymmetric risk/reward |
| `ben_graham.py` | `ben_graham_agent()` | 349 | Net-net screening, Graham Number, margin of safety, financial strength |
| `growth_agent.py` | `growth_analyst_agent()` | 336 | Growth trend analysis, acceleration scoring, forward estimates |
| `technicals.py` | `technical_analyst_agent()` | 340 | Pattern tecnico, support/resistance, trend confirmation via LLM |
| `news_sentiment.py` | `news_sentiment_agent()` | 221 | News-based sentiment scoring per ticker |
| `fundamentals.py` | varie | 313 | Financial statement scoring (revenue, margins, ROE, debt) |

### 3.2 Agenti di infrastruttura

| File | Funzione | LOC | Ruolo |
|------|----------|-----|-------|
| `data_prefetch.py` | `data_prefetch_agent()` | ~100 | Singleton: fetcha tutti i dati market prima degli analisti |
| `risk_manager.py` | `risk_manager_agent()` | 474 | Correlazione, VaR parametrico, concentrazione settoriale, ATR levels |
| `portfolio_manager.py` | `portfolio_manager_agent()` | 521 | Aggregazione segnali con pesi EWA, position sizing, conviction scoring |

### 3.3 Risk Manager — dettaglio

Calcola per ogni run:
- **Matrice di correlazione** tra tutti i ticker
- **VaR parametrico 95%** su log-returns
- **Concentrazione settoriale** (warn se >60% segnali bullish in un settore)
- **Drawdown stimato** (alert se >15%)
- **Trade levels ATR-based** per ogni ticker:
  - Entry: close corrente
  - Stop-loss: entry ± 2×ATR(14)
  - Take-profit: entry ∓ 4×ATR(14)
  - Risk/Reward: 1:2

Soglie hard-coded:
```python
MAX_SECTOR_PCT        = 0.60   # 60% segnali in un settore
MAX_PORTFOLIO_VAR     = 0.04   # 4% VaR giornaliero
MAX_DRAWDOWN_LIMIT    = 0.15   # 15% max drawdown stimato
MAX_SINGLE_TICKER_PCT = 0.25   # 25% del portafoglio per ticker
```

### 3.4 Portfolio Manager — dettaglio

- **Conviction mapping**: HIGH → 15%, MEDIUM → 9%, LOW → 4% del portafoglio
- **Position limits**: min 2%, max 20% per ticker
- **Aggregazione**: media pesata dei segnali con `agent_weights` da SQLite (EWA)
- **Output per ticker**: action, sizing %, conviction, entry, SL, TP, position USD, R/R ratio, consensus score

---

## 4. Pipeline principale (`src/main.py`)

### Funzione core: `run_hedge_fund()`

```python
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,            # snapshot portafoglio corrente
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
) -> dict:
    # Compila e invoca il grafo LangGraph
    # Restituisce {decisions, analyst_signals}
```

### Topologia LangGraph

```
START
  ↓
data_prefetch_node          ← batch fetch yfinance + SEC EDGAR per tutti i ticker
  ↓
┌─────────────────────────────────────────────────────────────┐
│  [PARALLELO — fan-out]                                       │
│  warren_buffett_node    ben_graham_node    charlie_munger    │
│  michael_burry_node     bill_ackman_node   cathie_wood_node  │
│  peter_lynch_node       phil_fisher_node   damodaran_node    │
│  mohnish_pabrai_node    jhunjhunwala_node  druckenmiller_node│
│  growth_node            technicals_node    sentiment_node    │
│  fundamentals_node      news_sentiment_node                  │
└─────────────────────────────────────────────────────────────┘
  ↓
risk_manager_node           ← correlazione, VaR, concentrazione, ATR levels
  ↓
portfolio_manager_node      ← aggregazione segnali, sizing, conviction
  ↓
END
```

Il grafo è **compilato dinamicamente** in base agli analisti selezionati tramite `selected_analysts`. La riduzione degli stati usa:
- `data`: `_merge_dicts()` — i nodi paralleli scrivono senza conflitti
- `metadata`: `_keep_first()` — immutabile dopo la prima scrittura
- `messages`: `operator.add` — append concatenato

---

## 5. Stato LangGraph (`src/graph/state.py`)

```python
class AgentState(TypedDict):
    data: Annotated[dict, _merge_dicts]
    metadata: Annotated[dict, _keep_first]
    messages: Annotated[list, operator.add]
```

### Struttura di `data`

```python
{
  "tickers": ["AAPL", "MSFT", ...],
  "end_date": "2025-04-01",
  "start_date": "2025-03-01",

  "prefetched_data": {
    "AAPL": {
      "ohlcv_daily": pd.DataFrame,      # 1 anno OHLCV giornaliero
      "ohlcv_5m": pd.DataFrame,         # 5 giorni intraday 5 minuti
      "info": dict,                     # market cap, P/E, settore, ecc.
      "income_stmt": pd.DataFrame,      # annuale + trimestrale
      "balance_sheet": pd.DataFrame,
      "cash_flow": pd.DataFrame,
      "holders": pd.DataFrame,
      "fetched_at": "ISO-8601",
    },
    ...
  },

  "analyst_signals": {
    "warren_buffett": {
      "AAPL": {"signal": "bullish", "confidence": 82, "reasoning": "..."},
      ...
    },
    ...
  },

  "feedback_history": {
    ("warren_buffett", "AAPL"): [lista ultime N predizioni da SQLite],
    ...
  },

  "agent_weights": {
    ("warren_buffett", "AAPL"): 1.23,    # aggiornati da EWA
    ...
  },

  "risk_output": {
    "correlation_matrix": {...},
    "sector_concentration": {...},
    "parametric_var": float,
    "max_drawdown_estimate": float,
    "trade_levels": {"AAPL": {"entry": float, "sl": float, "tp": float, "rr": float}},
  },

  "portfolio_output": [
    {
      "ticker": "AAPL",
      "action": "BUY",
      "sizing_pct": 15.0,
      "conviction": "HIGH",
      "entry_price": 175.50,
      "stop_loss": 168.00,
      "take_profit": 190.00,
      "position_usd": 15000,
      "risk_reward": 2.0,
      "consensus_score": 0.74,
      "reasoning": "...",
    },
    ...
  ],
}
```

---

## 6. Layer dati (`src/data/`)

### 6.1 `prefetch.py` — DataPrefetcher (singleton)

Fetcha e cachea per ogni ticker:
- OHLCV giornaliero 1 anno (`yfinance`)
- OHLCV intraday 5 minuti — ultimi 5 trading days (`yfinance`)
- Company info: market cap, P/E, P/B, settore, dipendenti
- Financial statements annuali e trimestrali: income, balance sheet, cash flow
- Institutional holders

### 6.2 `state_reader.py` (360 LOC)

Utility functions per leggere dati dal `prefetched_data` dello stato in modo type-safe, con fallback gestiti.

### 6.3 `cache.py` + `ttl_cache.py`

Cache TTL su filesystem:
- Prezzi: 1 ora
- Financials: 7 giorni
- Directory: `/cache/`

### 6.4 `models.py` — Pydantic models

- `FinancialMetrics`: PE, PB, PS, ROE, ROA, ROIC, debt ratios, beta, ecc.
- `LineItem`: revenue, EBIT, FCF, earnings, dividends, ecc.
- `Price`: OHLCV
- `CompanyNews`: titolo, testo, data, fonte, autore (opzionale)
- `InsiderTrade`: nome, ruolo, tipo transazione, valore

### 6.5 `sec_edgar.py`

Scraping diretto di SEC EDGAR per 10-K / 10-Q. Usato come fallback quando la Financial Datasets API non ha dati.

---

## 7. Layer API (`src/tools/`)

### `api.py` (366 LOC)

Wrapper per la Financial Datasets API (`api.financialdatasets.ai`):
- `get_prices(ticker, start, end)` — OHLCV
- `get_price_data(ticker, start, end)` — alias con caching
- `get_financial_metrics(ticker, end_date, limit)` — metriche TTM/annuali
- `get_company_news(ticker, end_date, start_date, limit)` — news feed
- `search_line_items(ticker, line_items, period, limit)` — voci bilancio custom
- `get_market_cap(ticker, end_date)` — market cap puntuale
- `get_insider_trades(ticker, end_date, start_date, limit)` — transazioni insider

**Rate limiting**: backoff lineare (60s → 90s → 120s → ...) su risposta 429. Max 3 retry.

### `api_shim.py`

Shim di compatibilità che wrappa `api.py` registrando il contesto dello stato agente per i chiamanti negli agenti LLM.

---

## 8. Database (`src/db/` + `db/schema.sql`)

SQLite con WAL mode e foreign keys abilitati. 5 tabelle:

### `predictions`
```sql
id, run_id, agent_id, ticker,
signal CHECK('BUY','SELL','HOLD'),
confidence REAL CHECK(0.0-1.0),
reasoning_hash TEXT,      -- SHA-256 del reasoning (dedup)
timestamp TEXT            -- ISO-8601 UTC
```
Indici su: `run_id`, `(agent_id, ticker)`, `timestamp`.

### `outcomes`
```sql
id, prediction_id REFERENCES predictions(id),
ticker,
actual_return_1d, actual_return_5d, actual_return_20d,
window CHECK('1d','5d','20d'),
evaluated_at TEXT
```

### `agent_weights`
```sql
id, agent_id, ticker,
weight REAL DEFAULT 1.0 CHECK(>=0),
updated_at TEXT,
UNIQUE(agent_id, ticker)
```

### `agent_hyperparams`
```sql
id, agent_id, param_name, value TEXT, updated_at TEXT,
UNIQUE(agent_id, param_name)
```

### `pipeline_runs`
```sql
run_id TEXT PRIMARY KEY,   -- UUID
started_at, finished_at,
status CHECK('running','completed','failed'),
tickers TEXT,              -- JSON array
error_msg TEXT
```

---

## 9. Loop di feedback (`src/feedback/`)

### Flusso

1. Ogni run salva le predizioni in `predictions` con `run_id` (UUID)
2. Dopo 1/5/20 giorni, `outcome_tracker.py` misura i ritorni reali e popola `outcomes`
3. `weight_adjuster.py` calcola i nuovi pesi con EWA:
   ```
   w_new = alpha × accuracy + (1 - alpha) × w_old
   ```
4. I pesi aggiornati vengono scritti in `agent_weights`
5. Al run successivo, `portfolio_manager` legge i pesi e scala i segnali degli agenti

| File | LOC | Ruolo |
|------|-----|-------|
| `outcome_tracker.py` | 416 | Fetch outcomes, calcola P&L vs predizione, ranking agenti |
| `prompt_injector.py` | 324 | Inietta performance passata nei prompt degli agenti |
| `weight_adjuster.py` | ~150 | Aggiorna pesi EWA in SQLite |
| `logger.py` | ~80 | Logging strutturato predizioni/outcome |

---

## 10. Pipeline di produzione (`src/run_pipeline.py`, 329 LOC)

### Come si usa

```bash
# Tickers dal config/tickers.yaml
python -m src.run_pipeline

# Override ticker
python -m src.run_pipeline AAPL MSFT NVDA

# Senza email
python -m src.run_pipeline --no-email
```

### Flusso

1. `load_tickers()` — da YAML o CLI args
2. `init_db()` — inizializza SQLite se non esiste
3. `_build_initial_state()` — costruisce `AgentState` con tickers, date, config LLM
4. `compiled_graph.invoke(state)` — esegue il workflow LangGraph completo (2-5 min)
5. `insert_predictions()` — salva tutti i segnali in SQLite
6. `weight_adjuster.update_weights()` — aggiorna pesi EWA
7. `send_digest()` — email HTML con voti agenti, raccomandazioni portafoglio, note risk
8. `update_pipeline_run(status="completed")` — chiude il log di audit

Modello LLM default: **Claude Sonnet 4.5** (`claude-sonnet-4-5-20251001`, provider Anthropic).

---

## 11. Indicatori tecnici (`src/indicators/`)

### `technical_indicators.py` (340 LOC)

Calcola su DataFrame OHLCV:
- SMA(20), EMA(12/26)
- RSI(14)
- MACD(12, 26, 9) con signal line e histogram
- Bollinger Bands(20, ±2σ)
- ATR(14) — usato dal Risk Manager per SL/TP
- Stochastic Oscillator(%K, %D)
- ADX — forza del trend
- VWAP — Volume Weighted Average Price
- OBV — On Balance Volume
- Ichimoku Cloud
- Fibonacci retracements

### `regime_detector.py` (183 LOC)

Classifica il regime di mercato per ticker:
- `TRENDING_UP` / `TRENDING_DOWN`
- `RANGING`
- `VOLATILE`

Basato su ADX, volatilità rolling, e momentum.

### `multi_timeframe.py` (85 LOC)

Allineamento segnali tra timeframe daily (1D) e intraday (5M).

---

## 12. Sistema di alert email (`src/alerts/`)

### `email_sender.py`

- SMTP configurato tramite `.env` (`SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_RECIPIENT`)
- Multipart MIME (testo plain + HTML)
- Rate limiting: max 1 email per ticker ogni 60 minuti
- `send_alert(ticker, signal, reasoning)` — alert immediato per ticker
- `send_digest(portfolio_output, risk_output)` — digest giornaliero completo

### `src/utils/email_report.py` (497 LOC)

Costruisce email HTML avanzate con:
- Tabella voti agenti (BUY/SELL/HOLD con conviction)
- Entry/SL/TP per ogni posizione raccomandata
- Risk metrics (VaR, correlazioni, concentrazione)
- Portfolio heatmap testuale

---

## 13. Monitor daemon (`src/monitor/`)

| File | Ruolo |
|------|-------|
| `daemon.py` | Loop continuo che esegue price/news check |
| `price_checker.py` | Monitora prezzi in real-time, triggera alert su movimenti >X% |
| `news_checker.py` | Monitora news in ingresso per i ticker monitorati |
| `alert_builder.py` | Costruisce payload alert per email/notifiche |

---

## 14. Backtesting (`src/backtesting/`)

### 14.1 Engine (`engine.py`, 194 LOC)

Loop di backtest giornaliero:
1. Prefetch dati storici per il periodo
2. Per ogni giorno T in `[start_date, end_date]`:
   - Fetch prezzi T-1 → T
   - `AgentController.run_agent()` → invoca `run_hedge_fund()` con snapshot portafoglio
   - `TradeExecutor.execute_trade()` → applica decisioni al portafoglio
   - Calcola valore totale portafoglio
   - Aggiorna metriche di performance (rolling)
   - Stampa tabella daily (date, ticker, action, qty, price, positions, metrics)
3. Restituisce `PerformanceMetrics`

### 14.2 Portfolio (`portfolio.py`, 195 LOC)

Traccia:
- Cash disponibile
- Posizioni long e short per ticker (shares)
- Cost basis long/short
- Margin usato per short
- Realized gains per ticker (long + short separati)

### 14.3 Metrics (`metrics.py`, 77 LOC)

Calcola dalla curva di equity:
- **Sharpe Ratio**: `sqrt(252) × (mean_excess / std_excess)`, RF = 4.34% annuo
- **Sortino Ratio**: `sqrt(252) × (mean_excess / downside_dev)`
- **Max Drawdown**: `min((V - rolling_max) / rolling_max) × 100`

### 14.4 Tipi (`types.py`, 105 LOC)

TypedDict principali:
- `AgentDecision`: `{action: "buy"|"sell"|"short"|"cover"|"hold", quantity: float}`
- `AgentOutput`: `{decisions: AgentDecisions, analyst_signals: AgentSignals}`
- `PortfolioValuePoint`: `{Date, Portfolio Value, Long/Short Exposure, Gross/Net Exposure}`
- `PerformanceMetrics`: `{sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_date, ...}`

---

## 15. Walk-Forward IS/OOS Testing (NUOVA FUNZIONALITÀ)

### File creati

#### `src/backtesting/walk_forward.py` (539 LOC)

**`PeriodResult`** (dataclass) — metriche per un singolo periodo:
```python
@dataclass
class PeriodResult:
    label: str                     # "IN-SAMPLE" | "OUT-OF-SAMPLE"
    start_date, end_date: str
    initial_capital, final_value: float
    total_return_pct: float
    annualized_return_pct: float   # (1+r)^(252/n) - 1
    volatility_ann_pct: float      # std_daily × sqrt(252) × 100
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown_pct: float | None
    max_drawdown_date: str | None
    calmar_ratio: float | None     # ann_return / |max_drawdown|
    positive_days_pct: float
    benchmark_return_pct: float | None     # SPY buy-and-hold totale
    benchmark_ann_return_pct: float | None # SPY annualizzato
    alpha_ann_pct: float | None    # Jensen's alpha annualizzato
    beta: float | None             # beta vs SPY (regressione OLS)
    n_trading_days: int
```

**`WalkForwardResult`** (dataclass):
```python
@dataclass
class WalkForwardResult:
    is_result: PeriodResult
    oos_result: PeriodResult
    sharpe_decay: float | None        # IS Sharpe - OOS Sharpe (ideale < 0.3)
    return_retention: float | None    # OOS return / IS return (ideale > 0.7)
```

**`WalkForwardAnalyzer`** — classe principale:
- `_run_period(start, end, label)` → esegue `BacktestEngine` indipendente
- `_compute_extended_metrics(...)` → calcola tutte le metriche aggiuntive
- `_compute_alpha_beta(returns, start, end)` → regressione OLS vs SPY
- `run()` → esegue IS + OOS in sequenza, calcola diagnostics
- `print_report(result)` → tabella Rich con colori (verde/giallo/rosso)
- `save_results(result, output_dir)` → salva JSON in `results/`

**Diagnostica di overfitting**:

| Indicatore | Formula | Ottimo | Preoccupante |
|------------|---------|--------|--------------|
| Sharpe Decay | IS Sharpe − OOS Sharpe | < 0.3 | > 0.8 |
| Return Retention | OOS return / IS return | > 0.7 | < 0.3 o negativo |
| OOS Sharpe | — | > 0.5 | ≤ 0 |

**Verdict automatico** (4 check binari):
1. OOS profittevole (total return > 0)
2. OOS Sharpe > 0
3. OOS batte SPY buy-and-hold
4. Sharpe decay < 0.8 (bassa degradazione IS→OOS)

Score 3-4: PROMISING — Score 2: MIXED — Score 0-1: CAUTION

#### `src/walk_forward_backtest.py` (CLI entry point)

```bash
# Selezione modello/analisti interattiva
poetry run python src/walk_forward_backtest.py \
  --tickers AAPL,MSFT,NVDA \
  --is-start 2023-01-01 --is-end 2023-12-31 \
  --oos-start 2024-01-01 --oos-end 2024-12-31

# Fully automated (nessun prompt)
poetry run python src/walk_forward_backtest.py \
  --tickers AAPL,MSFT,NVDA \
  --is-start 2023-01-01 --is-end 2023-12-31 \
  --oos-start 2024-01-01 --oos-end 2024-12-31 \
  --model gpt-4o \
  --analysts-all

# Con Ollama (zero costi API)
poetry run python src/walk_forward_backtest.py \
  --tickers AAPL,MSFT \
  --is-start 2024-01-01 --is-end 2024-06-30 \
  --oos-start 2024-07-01 --oos-end 2024-12-31 \
  --ollama \
  --initial-capital 50000
```

Parametri disponibili:
- `--tickers` — CSV di ticker (default: da `config/tickers.yaml`)
- `--is-start`, `--is-end` — periodo in-sample
- `--oos-start`, `--oos-end` — periodo out-of-sample
- `--model` — nome modello LLM
- `--ollama` — usa Ollama locale
- `--analysts` / `--analysts-all` — selezione analisti
- `--initial-capital` — capitale iniziale (default: 100.000)
- `--margin-requirement` — margine per short (default: 0.0)
- `--no-save` — non salva JSON
- `--output-dir` — directory output (default: `results/`)

---

## 16. Bug preesistenti corretti durante lo sviluppo

| File | Bug | Fix applicato |
|------|-----|--------------|
| `src/utils/analysts.py:13` | `from src.agents.sentiment import sentiment_analyst_agent` — funzione inesistente | `from src.agents.sentiment import sentiment_agent as sentiment_analyst_agent` |
| `src/main.py:8` | `from src.agents.portfolio_manager import portfolio_management_agent` — funzione inesistente | `from src.agents.portfolio_manager import portfolio_manager_agent as portfolio_management_agent` |
| `src/main.py:9` | `from src.agents.risk_manager import risk_management_agent` — funzione inesistente | `from src.agents.risk_manager import risk_manager_agent as risk_management_agent` |

---

## 17. Configurazione

### `config/tickers.yaml`

```yaml
tickers:
  - AAPL   # Apple - Tech
  - MSFT   # Microsoft - Tech
  - GOOGL  # Alphabet - Tech
  - AMZN   # Amazon - Consumer/Cloud
  - NVDA   # Nvidia - Semiconductors
  - JPM    # JPMorgan - Finance
  - JNJ    # Johnson & Johnson - Healthcare
  - XOM    # ExxonMobil - Energy
  - BRK-B  # Berkshire Hathaway - Conglomerate
  - SPY    # S&P 500 ETF - Benchmark
```

### `.env` (variabili richieste)

```env
# LLM
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...

# Dati finanziari
FINANCIAL_DATASETS_API_KEY=...

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASSWORD=...
ALERT_RECIPIENT=...
```

---

## 18. LLM providers supportati

Configurati in `src/llm/models.py`:

| Provider | Modelli supportati |
|----------|-------------------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4.1, o3-mini, ecc. |
| Anthropic | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| Groq | llama-3.3-70b, mixtral-8x7b, ecc. |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Ollama | llama3, mistral, phi3, qualsiasi modello locale |

---

## 19. Dependencies (`requirements.txt`)

```
# Orchestrazione LLM
langgraph, langchain, langchain-core
langchain-anthropic, langchain-openai, langchain-groq
langchain-ollama, langchain-deepseek

# Dati finanziari
yfinance

# Data science
pandas, numpy, matplotlib

# Infrastructure
requests, schedule, python-dotenv

# CLI & output
rich, tabulate, colorama, questionary

# Testing
pytest
```

---

## 20. Come eseguire il sistema

### Pipeline giornaliera
```bash
python -m src.run_pipeline
python -m src.run_pipeline AAPL MSFT NVDA --no-email
```

### Backtest singolo periodo
```bash
poetry run python src/backtester.py \
  --tickers AAPL,MSFT,NVDA \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --initial-cash 100000
```

### Walk-forward IS/OOS
```bash
poetry run python src/walk_forward_backtest.py \
  --tickers AAPL,MSFT,NVDA \
  --is-start 2023-01-01 --is-end 2023-12-31 \
  --oos-start 2024-01-01 --oos-end 2024-12-31 \
  --model gpt-4o --analysts-all
```

### Segnale singolo (oggi)
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

### Inizializza database
```bash
python -m src.db.init_db
```

---

## 21. Git log (ultimi commit rilevanti)

```
52b63b1  feat: improved risk manager, portfolio manager, email report with SL/TP/entry
e58f142  fix: agent imports, graph names, rate limit retry, remove druckenmiller, clear test positions
96b9aed  feat: Fasi 2-5 complete - pipeline end-to-end
16587b4  feat: Phase 1.3 + 1.4 + 2.1 - SQLite schema, AgentState, DataPrefetcher
aea6d73  fase 1.2: tickers.yaml e .env.example
c3d69b9  aggiungi .gitignore
b9b1459  fase 1.1: rimozione web app, docker, migrazione a requirements.txt
```

---

## 22. Statistiche finali

| Metrica | Valore |
|---------|--------|
| Righe di Python totali | 20.138 |
| File Python | 85 |
| Agenti filosofici | 15 |
| Tabelle database | 5 |
| Provider LLM supportati | 5 |
| Data source integrati | 3 (yfinance, SEC EDGAR, Financial Datasets API) |
| Ticker configurati | 10 |
| Ticker benchmark | 1 (SPY) |

---

*Report generato il 2026-04-01 — Athanor Alpha, branch `main`*
