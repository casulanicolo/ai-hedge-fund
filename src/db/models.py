"""
src/db/models.py
────────────────
Typed dataclass representations of the Athanor Alpha SQLite tables.
These are plain data containers — no ORM, no magic.
Use init_db.py to create the tables, and these classes to pass data around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


SignalType   = Literal["BUY", "SELL", "HOLD"]
WindowType   = Literal["1d", "5d", "20d"]
RunStatus    = Literal["running", "completed", "failed"]


# ─────────────────────────────────────────────
# 1. Prediction
# ─────────────────────────────────────────────

@dataclass
class Prediction:
    """
    One agent signal for one ticker in one pipeline run.
    Maps to the `predictions` table.
    """
    run_id:         str
    agent_id:       str
    ticker:         str
    signal:         SignalType
    confidence:     float               # 0.0 – 1.0
    timestamp:      str = field(default_factory=_utcnow)
    reasoning_hash: Optional[str] = None
    id:             Optional[int] = None   # set after INSERT

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.signal not in ("BUY", "SELL", "HOLD"):
            raise ValueError(f"signal must be BUY/SELL/HOLD, got {self.signal!r}")


# ─────────────────────────────────────────────
# 2. Outcome
# ─────────────────────────────────────────────

@dataclass
class Outcome:
    """
    Realised return measured after a prediction.
    Maps to the `outcomes` table.
    """
    prediction_id:      int
    ticker:             str
    window:             WindowType
    evaluated_at:       str = field(default_factory=_utcnow)
    actual_return_1d:   Optional[float] = None
    actual_return_5d:   Optional[float] = None
    actual_return_20d:  Optional[float] = None
    id:                 Optional[int] = None


# ─────────────────────────────────────────────
# 3. AgentWeight
# ─────────────────────────────────────────────

@dataclass
class AgentWeight:
    """
    Dynamic EWA weight for one (agent_id, ticker) pair.
    Maps to the `agent_weights` table.
    """
    agent_id:   str
    ticker:     str
    weight:     float = 1.0             # default weight at startup
    updated_at: str = field(default_factory=_utcnow)
    id:         Optional[int] = None

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")


# ─────────────────────────────────────────────
# 4. AgentHyperparam
# ─────────────────────────────────────────────

@dataclass
class AgentHyperparam:
    """
    Flexible key-value store for per-agent parameters.
    Maps to the `agent_hyperparams` table.
    """
    agent_id:   str
    param_name: str
    value:      str             # always stored as text; caller casts on read
    updated_at: str = field(default_factory=_utcnow)
    id:         Optional[int] = None


# ─────────────────────────────────────────────
# 5. PipelineRun
# ─────────────────────────────────────────────

@dataclass
class PipelineRun:
    """
    Audit record for one execution of the daily pipeline.
    Maps to the `pipeline_runs` table.
    """
    run_id:      str
    tickers:     list[str]          # will be JSON-serialised for storage
    started_at:  str = field(default_factory=_utcnow)
    finished_at: Optional[str] = None
    status:      RunStatus = "running"
    error_msg:   Optional[str] = None
