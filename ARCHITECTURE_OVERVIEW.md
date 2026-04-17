# Athanor Alpha — Architecture Overview

> **Purpose of this document**: Full-context reference for LLMs and developers.
> Describes the live system state as of 2026-04-17 (post FIX C2).
> Use it to understand file responsibilities, data contracts, and decision logic
> before reading or modifying any source file.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [Repository Layout](#2-repository-layout)
3. [Pipeline Topology (graph.py)](#3-pipeline-topology-graphpy)
4. [State Contract (state.py)](#4-state-contract-statepy)
5. [Data Layer — Prefetch](#5-data-layer--prefetch)
6. [The 4 Signal Dimensions](#6-the-4-signal-dimensions)
   - 6.1 [FUNDAMENTALS — Persona Agents](#61-fundamentals--persona-agents)
   - 6.2 [TECHNICAL — Price/Volume Agents](#62-technical--pricevolume-agents)
   - 6.3 [SENTIMENT — News + 10-Q MD&A](#63-sentiment--news--10-q-mda)
   - 6.4 [MACRO — VIX + Yield Curve](#64-macro--vix--yield-curve)
7. [Pre-Decision Layer](#7-pre-decision-layer)
   - 7.1 [Devil's Advocate](#71-devils-advocate)
   - 7.2 [Risk Manager](#72-risk-manager)
8. [Portfolio Manager — Decision Engine](#8-portfolio-manager--decision-engine)
   - 8.1 [Dimension-Balanced Aggregation](#81-dimension-balanced-aggregation)
   - 8.2 [Weighted Conviction (WC)](#82-weighted-conviction-wc)
   - 8.3 [Position Sizing](#83-position-sizing)
   - 8.4 [Macro Regime Position Caps](#84-macro-regime-position-caps)
   - 8.5 [AGENT_DIMENSION_MAP (authoritative)](#85-agent_dimension_map-authoritative)
9. [Infrastructure](#9-infrastructure)
   - 9.1 [Run Log Persistence](#91-run-log-persistence)
   - 9.2 [Critical Alert System](#92-critical-alert-system)
   - 9.3 [SQLite Portfolio Decisions](#93-sqlite-portfolio-decisions)
10. [Agent Output Contract (AgentOutput)](#10-agent-output-contract-agentoutput)
11. [Key Constants Reference](#11-key-constants-reference)
12. [Execution Modes (run_pipeline.py)](#12-execution-modes-run_pipelinepy)

---

## 1. Core Philosophy

Athanor Alpha is a **LangGraph-based multi-agent trading pipeline** for
3-4 day equity swing trades. It targets a small universe of ~10-20 liquid
tickers (large-cap US equities + crypto).

### The Orthogonality Problem (FIX C2)

Before FIX C2, the system had **12 analyst agents** but no structural
separation of concerns. Six agents (Buffett, Graham, Munger, Burry, Ackman,
Cathie Wood) all read fundamentals data and voted independently. Their votes
were summed equally, making FUNDAMENTALS contribute ~60% of the final signal
even when TECHNICAL/SENTIMENT/MACRO disagreed. This produced systematic
over-confidence and "crowding" of fundamentals signals.

### The Solution: 4 Orthogonal Dimensions

FIX C2 introduces **dimension-based aggregation** with equal weight per
dimension, not per agent. Agents are grouped into exactly one of four
canonical dimensions:

| Dimension      | What it measures                        | Signal independence              |
|----------------|-----------------------------------------|----------------------------------|
| FUNDAMENTALS   | Business quality, valuation, moat       | Financial statements + LLM views |
| TECHNICAL      | Price structure, momentum, volume       | OHLCV only, no fundamentals      |
| SENTIMENT      | News flow, management tone (10-Q MD&A)  | Text sources only                |
| MACRO          | VIX, yield curve, rate trends           | Global market, not per-ticker     |

Each active dimension contributes **exactly 25%** to the final net_score
(or 1/N if fewer than 4 dimensions are active in a given run). This
eliminates the crowding problem: 8 FUNDAMENTALS agents + 1 TECHNICAL agent
cannot outvote 2-2 disagreement across dimensions.

---

## 2. Repository Layout

```
athanor-alpha/
├── src/
│   ├── agents/                    # All LangGraph nodes
│   │   ├── data_prefetch.py       # START node — batch fetch only, no LLM
│   │   ├── macro_agent.py         # MACRO dimension (FIX C2 — new)
│   │   ├── sentiment.py           # SENTIMENT dimension
│   │   ├── technicals.py          # TECHNICAL dimension (indicators)
│   │   ├── breakout_momentum.py   # TECHNICAL dimension (breakout/volume)
│   │   ├── fundamentals.py        # FUNDAMENTALS dimension (generic)
│   │   ├── warren_buffett.py      # FUNDAMENTALS persona
│   │   ├── ben_graham.py          # FUNDAMENTALS persona
│   │   ├── charlie_munger.py      # FUNDAMENTALS persona
│   │   ├── michael_burry.py       # FUNDAMENTALS persona
│   │   ├── bill_ackman.py         # FUNDAMENTALS persona
│   │   ├── cathie_wood.py         # FUNDAMENTALS persona
│   │   ├── phil_fisher.py         # FUNDAMENTALS persona
│   │   ├── peter_lynch.py         # FUNDAMENTALS persona
│   │   ├── mohnish_pabrai.py      # FUNDAMENTALS persona
│   │   ├── rakesh_jhunjhunwala.py # FUNDAMENTALS persona
│   │   ├── aswath_damodaran.py    # FUNDAMENTALS persona
│   │   ├── stanley_druckenmiller.py # FUNDAMENTALS persona (macro-flavour)
│   │   ├── valuation.py           # FUNDAMENTALS (valuation models)
│   │   ├── growth_agent.py        # FUNDAMENTALS (growth analysis)
│   │   ├── devils_advocate.py     # Pre-decision: adversarial veto
│   │   ├── risk_manager.py        # Pre-decision: sizing constraints
│   │   ├── portfolio_manager.py   # Decision engine + run log writer
│   │   └── time_exit_agent.py     # Post-decision: time-based exit check
│   ├── data/
│   │   ├── prefetch.py            # DataPrefetcher singleton (yfinance)
│   │   ├── sec_edgar.py           # SEC EDGAR 8-K/10-Q fetcher + cache
│   │   ├── macro_fetcher.py       # MacroFetcher: ^VIX ^TNX ^IRX ^FVX
│   │   ├── state_reader.py        # Typed accessors for prefetched_data
│   │   ├── ttl_cache.py           # TTL cache layer
│   │   └── signal_cache.py        # Per-agent signal cache
│   ├── graph/
│   │   ├── graph.py               # LangGraph graph builder
│   │   └── state.py               # AgentState TypedDict + AgentOutput Pydantic model
│   ├── indicators/
│   │   ├── technical_indicators.py # RSI, MACD, Bollinger, ATR, VWAP, OBV, Ichimoku, Fibonacci
│   │   └── regime_detector.py      # TRENDING/RANGING/VOLATILE classifier
│   ├── utils/
│   │   ├── analysts.py            # ANALYST_CONFIG — single source of truth for all agents
│   │   ├── llm.py                 # call_llm() wrapper (Anthropic/OpenAI)
│   │   ├── trade_levels.py        # ATR-based entry/SL/TP calculator
│   │   └── progress.py            # Progress tracker
│   ├── db/
│   │   └── init_db.py             # SQLite schema + insert helpers
│   ├── feedback/
│   │   └── logger.py              # prediction_log_node
│   └── portfolio/
│       └── manager.py             # get_open_positions() (SQLite)
├── config/
│   └── risk_params.yaml           # Tunable risk parameters
├── logs/
│   └── runs.jsonl                 # Append-only run log (created at first run)
└── ARCHITECTURE_OVERVIEW.md      # This file
```

---

## 3. Pipeline Topology (graph.py)

File: `src/graph/graph.py`

The graph is **static** (compiled once at import time). Conditional behaviour
is implemented via wrapper functions that read `state["metadata"]` flags —
not via dynamic edge routing.

```
START
  │
  ▼
data_prefetch          ← fetches yfinance + SEC EDGAR + macro data. No LLM.
  │
  ├─► warren_buffett   ─┐
  ├─► ben_graham        │
  ├─► charlie_munger    │  Fan-out: all analyst nodes run in parallel.
  ├─► michael_burry     │  Nodes not in active_analyst_nodes are skipped
  ├─► bill_ackman       │  via _make_conditional_analyst() wrapper.
  ├─► cathie_wood       │
  ├─► technicals        │
  ├─► fundamentals      │
  ├─► sentiment         │
  ├─► breakout_momentum │
  └─► macro             ┘
                        │
                        ▼  Fan-in: all analyst nodes → devils_advocate
                 devils_advocate    ← skip if skip_devils_advocate=True
                        │
                        ▼
                  risk_manager      ← skip if skip_risk_manager=True
                        │
                        ▼
               portfolio_manager    ← always runs
                        │
                        ▼
               prediction_log       ← skip if skip_prediction_log=True
                        │
                        ▼
                 time_exit          ← skip if skip_time_exit=True (active: review mode)
                        │
                        ▼
                       END
```

### Analyst node registry (`_ALL_ANALYST_NODES`)

Defined in `graph.py` lines ~154–166. To add/remove an agent, edit **only
this list** (plus `AGENT_DIMENSION_MAP` in `portfolio_manager.py`):

```python
_ALL_ANALYST_NODES = [
    ("warren_buffett",    warren_buffett_agent),
    ("ben_graham",        ben_graham_agent),
    ("charlie_munger",    charlie_munger_agent),
    ("michael_burry",     michael_burry_agent),
    ("bill_ackman",       bill_ackman_agent),
    ("cathie_wood",       cathie_wood_agent),
    ("technicals",        technical_analyst_agent),
    ("fundamentals",      fundamentals_analyst_agent),
    ("sentiment",         sentiment_agent),
    ("breakout_momentum", breakout_momentum_agent),
    ("macro",             macro_agent),
]
```

Node names in the graph (used in `active_analyst_nodes`) are the first
element of each tuple (e.g. `"macro"`, `"technicals"`).

---

## 4. State Contract (state.py)

File: `src/graph/state.py`

```python
class AgentState(TypedDict, total=False):
    data:     Annotated[dict[str, Any], _merge_dicts]   # parallel-safe shallow merge
    metadata: Annotated[dict[str, Any], _keep_first]    # written once at pipeline start
    messages: Annotated[list[Any],      operator.add]   # append-only message log
```

### `state["data"]` keys

| Key                      | Written by          | Read by                      |
|--------------------------|---------------------|------------------------------|
| `prefetched_data`        | data_prefetch       | all analyst agents           |
| `macro_data`             | data_prefetch       | macro_agent                  |
| `analyst_signals`        | every analyst agent | devils_advocate, risk_manager, portfolio_manager |
| `risk_report`            | risk_manager        | portfolio_manager            |
| `devils_advocate_output` | devils_advocate     | portfolio_manager            |
| `portfolio_recommendations` | portfolio_manager | prediction_log, time_exit  |
| `tickers`                | make_initial_state  | all agents                   |
| `feedback_history`       | make_initial_state  | analyst agents (EWA weights) |
| `agent_weights`          | make_initial_state  | analyst agents               |

### `state["metadata"]` keys

| Key                      | Type          | Purpose                                   |
|--------------------------|---------------|-------------------------------------------|
| `run_id`                 | str           | UUID for this pipeline run                |
| `tickers`                | list[str]     | Tickers to analyse                        |
| `model_name`             | str           | LLM model ID (e.g. `claude-sonnet-4-6`)   |
| `model_provider`         | str           | `"Anthropic"` or `"OpenAI"`               |
| `active_analyst_nodes`   | list[str]     | Node names to activate (skip others)      |
| `skip_devils_advocate`   | bool          | Skip devils_advocate node if True         |
| `skip_risk_manager`      | bool          | Skip risk_manager node if True            |
| `skip_prediction_log`    | bool          | Skip prediction_log node if True          |
| `skip_time_exit`         | bool          | Skip time_exit node if True               |
| `run_mode`               | str           | `"full"` \| `"review"` \| `"quick"`       |

---

## 5. Data Layer — Prefetch

All market data is fetched **once** before any analyst runs, stored in
`state["data"]["prefetched_data"]`. No analyst agent calls yfinance directly.

### DataPrefetcher (`src/data/prefetch.py`)

Singleton. Fetches per ticker via `yf.Ticker()`:

| Key in TickerPayload    | Content                                           |
|-------------------------|---------------------------------------------------|
| `ohlcv_daily`           | DataFrame, 1-year daily OHLCV, auto-adjusted      |
| `ohlcv_weekly`          | DataFrame, 2-year weekly OHLCV                    |
| `ohlcv_4h`              | DataFrame, 60-day 4h candles                      |
| `ohlcv_5m`              | DataFrame, 5-day intraday 5m candles              |
| `info`                  | dict: market_cap, P/E, sector, beta, etc.         |
| `income_stmt`           | DataFrame, annual income statement                |
| `income_stmt_q`         | DataFrame, quarterly income statement             |
| `balance_sheet`         | DataFrame, annual balance sheet                   |
| `balance_sheet_q`       | DataFrame, quarterly balance sheet                |
| `cash_flow`             | DataFrame, annual cash flow                       |
| `cash_flow_q`           | DataFrame, quarterly cash flow                    |
| `holders`               | DataFrame, institutional holders                  |
| `fetched_at`            | str, ISO-8601 UTC timestamp                       |

### SECFetcher (`src/data/sec_edgar.py`)

Fetches 8-K (material events) and 10-Q filings from SEC EDGAR. Data is
merged into `prefetched_data[ticker]["sec_filings"]`:

```python
sec_filings = {
    "8-K":  [{"accession": ..., "filed": ..., "document": ..., "description": ...}, ...],
    "10-Q": [{"accession": ..., "filed": ..., "document": ..., "cik": ...}, ...],
}
```

The sentiment agent reads `sec_filings["10-Q"][0]` to fetch the MD&A section
from EDGAR archives via HTTP (URL pattern:
`https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}`).

### MacroFetcher (`src/data/macro_fetcher.py`)

Fetches **global** (not per-ticker) macro data. Called once in
`data_prefetch_agent`, written to `state["data"]["macro_data"]`.

```python
# yfinance symbols fetched:
"^VIX"   — CBOE Volatility Index          (period="15d")
"^TNX"   — 10-Year Treasury Yield (%)      (period="30d")
"^IRX"   — 13-Week T-Bill / 3M proxy (%)   (period="30d")
"^FVX"   — 5-Year Treasury Yield (%)       (period="30d")

# Derived fields:
"yield_spread_10y3m" = yield_10y - yield_3m   # negative = inverted curve
"rate_trend_20d"     = TNX[today] - TNX[20d ago]   # positive = rates rising
"vix_trend_10d"      = VIX[today] - VIX[10d ago]   # positive = fear rising
```

Output schema (`MacroData`):

```python
{
    "vix":                float | None,
    "yield_10y":          float | None,
    "yield_3m":           float | None,
    "yield_5y":           float | None,
    "yield_spread_10y3m": float | None,
    "rate_trend_20d":     float | None,
    "vix_trend_10d":      float | None,
    "fetched_at":         str,
}
```

---

## 6. The 4 Signal Dimensions

### 6.1 FUNDAMENTALS — Persona Agents

Each persona agent applies a specific investor philosophy to the same
financial data. They all read from `prefetched_data` (income statement,
balance sheet, cash flow, info) and produce one `AgentOutput` per ticker.

**Active persona agents** (as registered in `graph.py`):

| Node name         | Agent function              | Philosophy                                |
|-------------------|-----------------------------|-------------------------------------------|
| `warren_buffett`  | `warren_buffett_agent`      | Moat, ROE, free cash flow                 |
| `ben_graham`      | `ben_graham_agent`          | Net-nets, margin of safety                |
| `charlie_munger`  | `charlie_munger_agent`      | Quality at fair price, mental models      |
| `michael_burry`   | `michael_burry_agent`       | Deep value, contrarian, distressed assets |
| `bill_ackman`     | `bill_ackman_agent`         | Activist, concentrated bets               |
| `cathie_wood`     | `cathie_wood_agent`         | Disruptive innovation, 5-year arc         |
| `fundamentals`    | `fundamentals_analyst_agent`| Financial ratio screening                 |

Additional persona agents defined in `src/utils/analysts.py` (ANALYST_CONFIG)
but **not in `_ALL_ANALYST_NODES`** — inactive in the graph:
`phil_fisher`, `peter_lynch`, `mohnish_pabrai`, `rakesh_jhunjhunwala`,
`aswath_damodaran`, `stanley_druckenmiller`, `valuation_analyst`, `growth_analyst`.

> **Key insight**: All 7 active FUNDAMENTALS agents produce one vote each.
> In the old system they contributed 7 votes out of ~10 total.
> Post FIX C2 they collectively contribute **25% of the final net_score**
> (see §8.1).

### 6.2 TECHNICAL — Price/Volume Agents

Two agents, both read only from `ohlcv_daily` / `ohlcv_5m` / `ohlcv_weekly` /
`ohlcv_4h`. Zero fundamental data.

**`technicals` node** (`src/agents/technicals.py`):
- Computes full indicator snapshot via `src/indicators/technical_indicators.py`:
  RSI-14, MACD (12/26/9), Bollinger Bands (20,2), ATR-14, VWAP, OBV,
  Ichimoku (9/26/52), Fibonacci retracements.
- Detects market regime via `src/indicators/regime_detector.py`:
  `TRENDING` / `RANGING` / `VOLATILE`.
- Multi-timeframe: daily + intraday 5m when available.
- One LLM call per ticker → `AgentOutput`.

**`breakout_momentum` node** (`src/agents/breakout_momentum.py`):
- Specialised breakout/volume surge detection:
  - `VOLUME_SURGE_THRESHOLD = 2.0` (volume > 2× 20-day average = anomalous)
  - ATR expansion ratio (recent ATR vs 20d ago)
  - 52-week high proximity
  - Resistance breakout (close > 20d high with volume confirmation)
  - Weekly trend confirmation (EMA8 vs EMA21 on weekly candles)
  - Momentum: ROC-5d and ROC-10d
  - RSI-14 penalty: −0.20 to breakout score if RSI > 70 (UP) or RSI < 30 (SHORT)
- One LLM call per ticker → `AgentOutput`.

### 6.3 SENTIMENT — News + 10-Q MD&A

File: `src/agents/sentiment.py`  
`AGENT_ID = "sentiment_agent"`

**Sources (in order of priority)**:

1. **yfinance `.news`** — recent headlines + summaries (free, no API key).
   Filtered to last `NEWS_LOOKBACK_DAYS = 7` days.
   Each headline is classified by `_classify_event_type()` and `_classify_urgency()`.

2. **SEC 8-K filings** — material events from `sec_filings["8-K"]`.
   Read from prefetched SEC data; no additional HTTP call.

3. **10-Q MD&A section** (FIX C2 addition) — management tone from the latest
   10-Q filing. Fetched via `_fetch_10q_mda_snippet()`:
   - Reads `sec_filings["10-Q"][0]` from state
   - Constructs URL: `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}`
   - Searches for the second occurrence of the MD&A section header (skips TOC entry)
   - Strips HTML via `_strip_html()` (handles numeric `&#NNN;` and named `&amp;` entities)
   - Returns first 2,500 characters of clean text
   - Graceful fallback: HTTP error or missing data → agent continues without MD&A

**Output schema** (`SentimentSignal`):
```python
{
    "direction":       "LONG" | "SHORT" | "NEUTRAL",
    "expected_return": float,          # -0.10 to +0.10
    "confidence":      float,          # 0.1 to 1.0
    "sentiment_score": float,          # -1.0 to +1.0
    "event_type":      "earnings" | "M&A" | "regulatory" | "macro" | "other",
    "urgency":         "low" | "medium" | "high",
    "reasoning":       str,
}
```

**Urgency weights** (used in LLM prompt weighting, not in PM aggregation):
```python
URGENCY_WEIGHTS = {"high": 3.0, "medium": 1.5, "low": 1.0}
```

### 6.4 MACRO — VIX + Yield Curve

File: `src/agents/macro_agent.py`  
`AGENT_ID = "macro_agent"`

**Dimension: MACRO**. Architecture: one LLM call for the **global** macro
assessment, then per-ticker sensitivity scaling. No per-ticker LLM calls.

#### Deterministic scoring (`_compute_macro_scores`)

Three components, fixed weights:

| Component           | Weight | Source field          | Score range |
|---------------------|--------|-----------------------|-------------|
| VIX score           | 45%    | `vix`                 | 0–100       |
| Yield curve score   | 40%    | `yield_spread_10y3m`  | 0–100       |
| Rate trend score    | 15%    | `rate_trend_20d`      | 0–100       |

**VIX → score mapping**:
| VIX range  | Score | Regime   |
|------------|-------|----------|
| < 15       | 90    | CALM     |
| 15–18      | 80    | CALM     |
| 18–25      | 60    | NORMAL   |
| 25–35      | 30    | ELEVATED |
| ≥ 35       | 10    | CRISIS   |

**Yield spread (10Y−3M) → score mapping**:
| Spread (pp) | Score |
|-------------|-------|
| > +1.5      | 90    |
| +0.5 to +1.5| 72    |
| 0 to +0.5   | 55    |
| −0.5 to 0   | 38    |
| −1.0 to −0.5| 22    |
| < −1.0      | 10    |

**Rate trend (20d change in 10Y yield, pp) → score mapping**:
| Trend         | Score |
|---------------|-------|
| < −0.50       | 78    |
| −0.50 to −0.20| 65    |
| −0.20 to +0.20| 52    |
| +0.20 to +0.50| 38    |
| > +0.50       | 25    |

**Regime classification** (from composite score):
```
composite ≥ 62  →  RISK_ON
composite ≤ 38  →  RISK_OFF
else            →  CAUTION
```

#### Per-ticker sensitivity (`TICKER_SENSITIVITY`)

Applied **after** the global LLM signal. Multiplies `confidence` and
`expected_return` by a ticker-specific factor:

```python
TICKER_SENSITIVITY = {
    # Crypto (most macro-reactive)
    "BTC-USD": 2.0, "ETH-USD": 2.0, "SOL-USD": 2.0, "BNB-USD": 2.0,
    # Crypto-proxy equities
    "MSTR":    1.8, "COIN":    1.8,
    # High-beta growth
    "TSLA":    1.5, "SMCI":    1.5,
    "NVDA":    1.4,
    "MELI":    1.3, "META":    1.2, "AMZN":    1.1,
    # Broad market
    "AAPL":    0.9, "MSFT":    0.9, "GOOGL":   1.0,
    # Defensives / financials
    "JPM":     0.8, "V":       0.8, "UNH":     0.6,
    # Default for all other tickers: 1.0
}
# Caps: confidence ≤ 1.0, |expected_return| ≤ 0.05
```

Each ticker signal stored in `analyst_signals["macro_agent"]` includes:
```python
{
    "direction":       "LONG" | "SHORT" | "NEUTRAL",
    "expected_return": float,   # sensitivity-scaled, capped ±0.05
    "confidence":      float,   # sensitivity-scaled, capped 1.0
    "reasoning":       str,     # global reasoning + "[macro sensitivity Nx]"
    "macro_regime":    str,     # "RISK_ON" | "CAUTION" | "RISK_OFF"
    "vix":             float | None,
    "yield_spread":    float | None,
    "macro_composite": float,
    # + ATR-based trade levels from compute_trade_levels()
}
```

---

## 7. Pre-Decision Layer

### 7.1 Devil's Advocate

File: `src/agents/devils_advocate.py`  
`AGENT_ID = "devils_advocate"`

Adversarial agent that runs after all analysts (fan-in), before risk_manager.
Does **not** call an LLM for scoring — applies rule-based vetoes.

**Responsibilities**:
1. **VIX regime classification** → global `size_multiplier` (0.40–1.00)
2. **Signal coherence veto** → tickers where agent consensus < threshold are vetoed

**VIX regime → size_multiplier**:
| VIX range | Regime   | size_multiplier |
|-----------|----------|-----------------|
| < 18      | LOW      | 1.00            |
| 18–25     | NORMAL   | 0.90            |
| 25–35     | ELEVATED | 0.70            |
| > 35      | CRISIS   | 0.40            |

**Coherence threshold** (dynamic, based on VIX regime):
| VIX regime | Coherence threshold |
|------------|---------------------|
| LOW        | 0.45                |
| NORMAL     | 0.55                |
| ELEVATED   | 0.65                |
| CRISIS     | 0.75                |

Output in `state["data"]["devils_advocate_output"]`:
```python
{
    "vix_level":           float,
    "vix_regime":          "LOW" | "ELEVATED" | "HIGH" | "EXTREME",
    "macro_risk":          "LOW" | "MEDIUM" | "HIGH",
    "size_multiplier":     float,
    "coherence_threshold": float,
    "vetoed_tickers":      list[str],
    "veto_reasons":        dict[str, str],
    "reasoning":           str,
}
```

### 7.2 Risk Manager

File: `src/agents/risk_manager.py`  
`AGENT_ID = "risk_manager"`

Computes portfolio-level risk constraints. Does **not** block signals —
only annotates. The portfolio_manager applies the constraints.

**Responsibilities**:
- Correlation matrix (10-ticker pairwise Pearson on log returns)
- Sector concentration check (`SECTOR_MAP` hardcoded, `sector_cap = 0.30` from YAML)
- Parametric VaR at 95% (`VAR_CONFIDENCE = 0.95`, `MAX_PORTFOLIO_VAR = 0.04`)
- Max drawdown estimate (`MAX_DRAWDOWN_LIMIT = 0.15`)
- ATR-based trade levels (entry, SL, TP) for all bullish/bearish tickers
- Crypto risk model (B6): separate VaR + `MAX_CRYPTO_POSITION_PCT = 0.02`

**ATR trade level parameters**:
```python
ATR_PERIOD  = 14     # Wilder's smoothing
ATR_SL_MULT = 1.0    # Stop Loss = entry ± 1×ATR14
ATR_TP_MULT = 2.0    # Take Profit = entry ∓ 2×ATR14  →  R/R = 1:2
```

Output in `state["data"]["risk_report"]`:
```python
{
    "daily_var_95":           float,    # portfolio-level
    "max_drawdown_estimate":  float,
    "correlation_matrix":     dict[str, dict[str, float]],
    "ticker_flags":           dict[str, list[str]],   # per-ticker risk warnings
    "trade_levels":           dict[str, dict],        # {ticker: {entry, stop_loss, take_profit, rr_ratio}}
    "warnings":               list[str],
    "macro_regime":           dict,     # from detect_macro_regime()
}
```

---

## 8. Portfolio Manager — Decision Engine

File: `src/agents/portfolio_manager.py`  
`AGENT_ID = "portfolio_manager"`

Always runs (never skipped). Reads analyst_signals + risk_report + devils_advocate_output.
Writes `state["data"]["portfolio_recommendations"]`.

### 8.1 Dimension-Balanced Aggregation

Function: `_weighted_signals(state, tickers) → dict[str, dict]`

The aggregation is a **2-level process**:

**Level 1 — Intra-dimension averaging** (per-dimension):
```
For each dimension D in {FUNDAMENTALS, TECHNICAL, SENTIMENT, MACRO}:
    dim_score[D] = mean(score_i for all agents i in D)
    dim_conf[D]  = mean(confidence_i for all agents i in D)
    dim_wc[D]    = mean(ER_i × conf_i for non-neutral agents i in D)
```

Where `score_i = +confidence_i` (bullish), `−confidence_i` (bearish),
or `0.0` (neutral) for each agent.

**Level 2 — Cross-dimension equal weighting**:
```
n_dims       = number of active dimensions (those with ≥1 agent)
final_score  = sum(dim_score[D] for D in active_dims) / n_dims
final_conf   = sum(dim_conf[D]  for D in active_dims) / n_dims
wc_final     = sum(dim_wc[D]    for D in active_dims) / n_dims
```

With all 4 dimensions active: each contributes **exactly 25%**.
If only 3 active (e.g. macro agent skipped): each contributes **33.3%**.

**Net signal classification**:
```
final_score > +0.25  →  "bullish"  (NET_SCORE_THRESHOLD = 0.25)
final_score < −0.25  →  "bearish"
else                 →  "neutral"
```

**Consensus ratio** (dimension-based):
```
n_agreeing_dims = count of dims where dim_score agrees with net signal
consensus_ratio = n_agreeing_dims / n_dims
```

Trade only if `consensus_ratio ≥ MIN_CONSENSUS_RATIO = 0.65`.

### 8.2 Weighted Conviction (WC)

```
WC = sum(ER_i × conf_i for non-neutral agents) / n_non_neutral_agents
```

Used as a gate: `|WC| ≥ WC_THRESHOLD = 0.008` (0.8%) to enter a trade.

WC is also passed to the LLM in the prompt and included in every
recommendation's reasoning string as `"WC=+0.0XX"`.

### 8.3 Position Sizing

Function: `_compute_sizing(ticker, agg, risk_report, state)`.

Three checks before sizing:
1. `consensus_ratio ≥ 0.65` else HOLD
2. `conviction ≥ MIN_CONVICTION_TO_TRADE = 0.30` else HOLD
3. `|WC| ≥ WC_THRESHOLD = 0.008` else HOLD

If all pass, compute sizing as `min(vol_adjusted_size, half_kelly_size)`:

**Volatility Targeting**:
```
vol_size = target_daily_vol / ATR_pct
         (target_daily_vol from risk_params.yaml, default 0.01 = 1%)
```

**Half-Kelly**:
```
kelly_f  = (p_win × (rr_ratio + 1) − 1) / rr_ratio
half_f   = kelly_f × kelly_fraction    (default 0.5)
```

Where `p_win = avg_confidence` (from dimension aggregation),
`rr_ratio` from risk_manager's trade_levels (default 2.0).

Both values are clipped to `[min_position_pct, max_position_pct]`
from `config/risk_params.yaml` (defaults: min=2%, max=15%).

Post-sizing adjustments (in order):
1. Risk penalty from ticker_flags (0.80×) and portfolio VaR > 4% (0.75×)
2. VIX size_multiplier from devils_advocate (0.40–1.00)
3. Correlation penalty: secondary BUY in pair with |corr| > 0.70 → sizing × 0.70
4. `MAX_SINGLE_POSITION = 20.0` hard cap
5. **Macro regime cap (FIX C2)** — applied last, overrides everything

### 8.4 Macro Regime Position Caps

```python
MACRO_REGIME_POSITION_CAPS = {
    "RISK_ON":  20.0,   # normal max (same as MAX_SINGLE_POSITION)
    "CAUTION":  12.0,   # reduced: macro uncertainty
    "RISK_OFF":  5.0,   # drastic: VIX crisis OR deeply inverted curve
    "UNKNOWN":  12.0,   # treat as CAUTION
}
```

The regime is read from `analyst_signals["macro_agent"][first_ticker]["macro_regime"]`.
If macro_agent did not run, fallback = `"CAUTION"` (conservative).

### 8.5 AGENT_DIMENSION_MAP (authoritative)

Defined in `portfolio_manager.py`. This is the **single source of truth**
mapping every `agent_id` key (as written into `analyst_signals`) to a
canonical dimension.

```python
AGENT_DIMENSION_MAP = {
    # ── FUNDAMENTALS ──────────────────────────────────────────────────────
    "fundamentals_analyst_agent": "FUNDAMENTALS",
    "warren_buffett_agent":       "FUNDAMENTALS",
    "ben_graham_agent":           "FUNDAMENTALS",
    "charlie_munger_agent":       "FUNDAMENTALS",
    "michael_burry_agent":        "FUNDAMENTALS",
    "bill_ackman_agent":          "FUNDAMENTALS",
    "cathie_wood_agent":          "FUNDAMENTALS",
    "phil_fisher_agent":          "FUNDAMENTALS",
    "mohnish_pabrai_agent":       "FUNDAMENTALS",
    "peter_lynch_agent":          "FUNDAMENTALS",
    "rakesh_jhunjhunwala_agent":  "FUNDAMENTALS",
    "aswath_damodaran_agent":     "FUNDAMENTALS",
    # ── TECHNICAL ─────────────────────────────────────────────────────────
    "technical_analyst_agent":    "TECHNICAL",
    "breakout_momentum":          "TECHNICAL",
    # ── SENTIMENT ─────────────────────────────────────────────────────────
    "sentiment_agent":            "SENTIMENT",
    "news_sentiment_agent":       "SENTIMENT",   # legacy key, inactive
    # ── MACRO ─────────────────────────────────────────────────────────────
    "macro_agent":                "MACRO",
    # Agents NOT listed → bucketed under "OTHER" (still contribute, don't
    # inflate any canonical dimension)
}
```

> **Important**: The `agent_id` key used to write signals into
> `analyst_signals` may differ from the node name in the graph.
> Node names (graph registration): `"warren_buffett"`, `"macro"`, etc.
> Agent_id keys in signals: `"warren_buffett_agent"`, `"macro_agent"`, etc.
> Always check the `AGENT_ID` constant in each agent file.

---

## 9. Infrastructure

### 9.1 Run Log Persistence

Function: `_write_run_log()` in `portfolio_manager.py`.  
Called at end of every `portfolio_manager_agent()` call.

**Output file**: `logs/runs.jsonl` (created automatically, append-only).

**Schema per line** (one JSON object per run):

```json
{
  "run_id":    "uuid-string",
  "timestamp": "2026-04-17T14:23:00Z",
  "tickers":   ["AAPL", "BTC-USD", "NVDA"],
  "macro_regime": "CAUTION",
  "per_ticker": {
    "AAPL": {
      "net_score":           0.3120,
      "dim_scores":          {"FUNDAMENTALS": 0.45, "TECHNICAL": 0.20, "MACRO": 0.25},
      "weighted_conviction": 0.0148,
      "consensus_ratio":     0.750,
      "n_dims":              3,
      "n_agents":            9
    }
  },
  "recommendations": [
    {
      "ticker":     "AAPL",
      "action":     "BUY",
      "sizing_pct": 10.0,
      "conviction": 0.68,
      "wc":         0.0148,
      "reasoning":  "Strong FUNDAMENTALS + TECHNICAL alignment WC=+0.015..."
    }
  ],
  "portfolio_summary": "...",
  "risk_notes": "..."
}
```

Use `runs.jsonl` for ex-post analysis of decision quality, dimension drift,
and macro regime accuracy.

### 9.2 Critical Alert System

Function: `_check_macro_alert(scores, macro_data)` in `macro_agent.py`.  
Called immediately after `_compute_macro_scores()`, before the LLM call.

**Trigger**: `scores["regime"] == "RISK_OFF"` only.

**Effect**:
1. Prints ASCII banner to stdout (`print(..., flush=True)`)
2. Logs at `logger.critical(...)` level

**Banner example**:
```
╔══════════════════════════════════════════════════════════════╗
║          ⚠  MACRO CRITICAL ALERT — RISK_OFF REGIME  ⚠       ║
╠══════════════════════════════════════════════════════════════╣
║  Composite score :  14.2/100  (threshold ≤38)                ║
║  VIX             : 42.0        (regime: CRISIS    )          ║
║  Yield spread    : -1.200pp    (10Y - 3M)                    ║
║                                                              ║
║  ACTION: All position caps hard-limited to 5%.               ║
║          Review open positions for immediate risk reduction. ║
╚══════════════════════════════════════════════════════════════╝
```

### 9.3 SQLite Portfolio Decisions

`portfolio_manager_agent` also persists decisions to SQLite via
`src/db/init_db.py → insert_portfolio_decision()`. This is separate from
`runs.jsonl` and supports the feedback loop (EWA agent weights).

---

## 10. Agent Output Contract (AgentOutput)

File: `src/graph/state.py`

All analyst agents (FUNDAMENTALS, TECHNICAL, SENTIMENT, MACRO) must
produce output conforming to `AgentOutput`:

```python
class AgentOutput(BaseModel):
    direction:       Literal["LONG", "SHORT", "NEUTRAL"]
    expected_return: float = 0.0     # clamped [-0.10, +0.10]
    confidence:      float = 0.5     # clamped [0.10,  1.00]
    reasoning:       str   = ""
```

Written to `state["data"]["analyst_signals"][agent_id][ticker]`:

```python
{
    "direction":       "LONG",
    "expected_return": 0.025,
    "confidence":      0.72,
    "reasoning":       "...",
    # agents may add extra fields (e.g. macro_regime, sentiment_score)
    # extras are ignored by _weighted_signals()
}
```

---

## 11. Key Constants Reference

All in `src/agents/portfolio_manager.py` unless noted.

| Constant                    | Value  | Meaning                                             |
|-----------------------------|--------|-----------------------------------------------------|
| `NET_SCORE_THRESHOLD`       | 0.25   | Min |net_score| to classify as bullish/bearish      |
| `MIN_CONSENSUS_RATIO`       | 0.65   | Min fraction of dims agreeing to enter trade        |
| `MIN_CONVICTION_TO_TRADE`   | 0.30   | Min conviction score to enter trade                 |
| `WC_THRESHOLD`              | 0.008  | Min |WC| to enter trade (0.8%)                      |
| `KELLY_FRACTION`            | 0.25   | Fraction of full Kelly used for sizing              |
| `CORR_PENALTY_THRESHOLD`    | 0.70   | Pairwise correlation above which penalty applies    |
| `CORR_PENALTY_FACTOR`       | 0.70   | Multiply secondary position sizing by this factor   |
| `MAX_SINGLE_POSITION`       | 20.0   | Hard cap on any single position (%)                 |
| `MIN_SINGLE_POSITION`       | 2.0    | Minimum position size to include (%)                |
| `TOTAL_GROSS_BUDGET`        | 100.0  | Max sum of all BUY sizings before normalisation     |
| `MAX_ACTIVE_TRADES`         | 3      | Top N trades; remaining demoted to HOLD             |
| `PORTFOLIO_SIZE_USD`        | 10000  | Portfolio size (env: `PORTFOLIO_SIZE_USD`)          |
| `RISK_PER_TRADE_PCT`        | 0.005  | Risk per trade as fraction of portfolio (0.5%)      |
| `INFO_NET_SCORE_THRESHOLD`  | 0.10   | Min |net_score| to show informational levels on HOLD|
| `ATR_PERIOD` (risk_manager) | 14     | ATR lookback period (Wilder's smoothing)            |
| `ATR_SL_MULT` (risk_manager)| 1.0    | Stop loss = entry ± 1×ATR14                         |
| `ATR_TP_MULT` (risk_manager)| 2.0    | Take profit = entry ∓ 2×ATR14 (R/R = 1:2)          |
| `MAX_CRYPTO_POSITION_PCT`   | 0.02   | Max 2% per crypto ticker (risk_manager)             |
| `sector_cap` (risk_params)  | 0.30   | Max 30% of portfolio in one sector (YAML)           |

---

## 12. Execution Modes (run_pipeline.py)

Controlled via `--mode` flag. Each mode sets `metadata` flags that control
which nodes execute via the conditional wrappers.

| Mode      | Active analysts             | DA  | RM  | PredLog | TimeExit |
|-----------|-----------------------------|-----|-----|---------|----------|
| `full`    | all in `_ALL_ANALYST_NODES` | ✓   | ✓  | ✓       | ✗        |
| `quick`   | subset (fast analysts)      | ✓   | ✓  | ✗       | ✗        |
| `review`  | all                         | ✓   | ✓  | ✓       | ✓        |

In `review` mode, `time_exit_agent` reads `positions.json` / SQLite and
sends email alerts for positions open ≥ 4 sessions.
