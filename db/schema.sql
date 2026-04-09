-- Athanor Alpha — SQLite Schema
-- db/schema.sql
-- Run via: src/db/init_db.py

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─────────────────────────────────────────────
-- 1. PREDICTIONS
-- Every agent signal emitted in every pipeline run
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,           -- UUID per pipeline run
    agent_id        TEXT    NOT NULL,           -- e.g. "warren_buffett", "technicals"
    ticker          TEXT    NOT NULL,           -- e.g. "AAPL"
    signal          TEXT    NOT NULL            -- "BUY" | "SELL" | "HOLD"
                    CHECK(signal IN ('BUY', 'SELL', 'HOLD')),
    confidence      REAL    NOT NULL            -- 0.0 – 1.0
                    CHECK(confidence BETWEEN 0.0 AND 1.0),
    reasoning_hash  TEXT,                       -- SHA-256 of reasoning text (for dedup)
    timestamp       TEXT    NOT NULL,           -- ISO-8601 UTC, e.g. "2025-03-01T14:05:00Z"
    entry_price     REAL,                       -- prezzo di chiusura al momento della previsione
    stop_loss       REAL,                       -- livello SL calcolato dall'agente (ATR-based)
    take_profit     REAL                        -- livello TP calcolato dall'agente (ATR-based)
);

CREATE INDEX IF NOT EXISTS idx_predictions_run    ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_agent  ON predictions(agent_id, ticker);
CREATE INDEX IF NOT EXISTS idx_predictions_ts     ON predictions(timestamp);

-- ─────────────────────────────────────────────
-- 2. OUTCOMES
-- Realised price returns measured against predictions
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER NOT NULL REFERENCES predictions(id),
    ticker          TEXT    NOT NULL,
    actual_return_1d  REAL,                     -- return T+1 day
    actual_return_5d  REAL,                     -- return T+5 days
    actual_return_20d REAL,                     -- return T+20 days
    window          TEXT    NOT NULL            -- "1d" | "5d" | "20d"
                    CHECK(window IN ('1d', '5d', '20d')),
    evaluated_at    TEXT    NOT NULL            -- ISO-8601 UTC when this row was written
);

CREATE INDEX IF NOT EXISTS idx_outcomes_prediction ON outcomes(prediction_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_ticker     ON outcomes(ticker);

-- ─────────────────────────────────────────────
-- 3. AGENT WEIGHTS
-- Dynamic per-agent, per-ticker confidence weights
-- updated by the EWA weight adjuster
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_weights (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    weight      REAL    NOT NULL DEFAULT 1.0
                CHECK(weight >= 0.0),
    updated_at  TEXT    NOT NULL,               -- ISO-8601 UTC
    UNIQUE(agent_id, ticker)                    -- one row per (agent, ticker) pair
);

CREATE INDEX IF NOT EXISTS idx_weights_agent ON agent_weights(agent_id, ticker);

-- ─────────────────────────────────────────────
-- 4. AGENT HYPERPARAMS
-- Flexible key-value store for per-agent parameters
-- (learning rate, confidence floor, etc.)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_hyperparams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL,
    param_name  TEXT    NOT NULL,
    value       TEXT    NOT NULL,               -- stored as text; cast on read
    updated_at  TEXT    NOT NULL,               -- ISO-8601 UTC
    UNIQUE(agent_id, param_name)
);

-- ─────────────────────────────────────────────
-- 5. PIPELINE RUNS  (audit log)
-- One row per execution of the daily pipeline
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id      TEXT    PRIMARY KEY,            -- UUID
    started_at  TEXT    NOT NULL,               -- ISO-8601 UTC
    finished_at TEXT,                           -- NULL if still running / crashed
    status      TEXT    NOT NULL DEFAULT 'running'
                CHECK(status IN ('running', 'completed', 'failed')),
    tickers     TEXT    NOT NULL,               -- JSON array, e.g. '["AAPL","NVDA",...]'
    error_msg   TEXT                            -- populated on failure
);

-- ─────────────────────────────────────────────
-- 6. SIGNAL CACHE  (Fase 2)
-- Cache segnali agente/ticker per evitare chiamate LLM ridondanti.
-- TTL gestito a runtime da signal_cache.py (default 24h).
-- PRIMARY KEY (agent_id, ticker) — un solo record per coppia.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signal_cache (
    agent_id    TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    signal      TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.0,
    reasoning   TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL,               -- ISO-8601 UTC
    PRIMARY KEY (agent_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_signal_cache_created ON signal_cache(created_at);

-- ─────────────────────────────────────────────
-- 7. PORTFOLIO DECISIONS  (Fase 3)
-- Output completo del Portfolio Manager per ogni run.
-- Una riga per ticker per run. Include livelli operativi
-- (BUY/SELL) e informativi (HOLD/vetati DA).
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolio_decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL,        -- UUID del pipeline run
    timestamp           TEXT    NOT NULL,        -- ISO-8601 UTC
    ticker              TEXT    NOT NULL,
    action              TEXT    NOT NULL         -- "BUY" | "SELL" | "HOLD"
                        CHECK(action IN ('BUY', 'SELL', 'HOLD')),
    net_score           REAL,                    -- weighted net score degli agenti
    avg_confidence      REAL,                    -- confidenza media agenti
    weighted_conviction REAL,                    -- WC = sum(ER*conf)/n_non_neutral
    conviction          REAL,                    -- conviction score PM (0-1)
    sizing_pct          REAL,                    -- peso % nel portafoglio
    consensus           TEXT,                    -- es. "7/10 bullish"
    -- Livelli operativi (solo BUY/SELL attivi)
    entry_price         REAL,
    stop_loss           REAL,
    take_profit         REAL,
    size_usd            REAL,
    rr_ratio            REAL,
    -- Livelli informativi (HOLD con |net_score|>0.10 o vetati DA)
    info_entry          REAL,
    info_sl             REAL,
    info_tp             REAL,
    info_size_usd       REAL,
    info_rr_ratio       REAL,
    info_direction      TEXT,                    -- "long" | "short" | NULL
    -- Flag veto
    devil_vetoed        INTEGER NOT NULL DEFAULT 0,  -- 1 se vetato DA
    -- Reasoning LLM
    reasoning           TEXT
);

CREATE INDEX IF NOT EXISTS idx_portfolio_decisions_run    ON portfolio_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_decisions_ticker ON portfolio_decisions(ticker, timestamp);
