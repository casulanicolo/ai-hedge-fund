"""
src/graph/state.py
──────────────────
Athanor Alpha – extended LangGraph AgentState.

All top-level keys use Annotated reducers so that parallel analyst nodes
can write concurrently without triggering InvalidUpdateError.
"""

from __future__ import annotations
from datetime import date

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, field_validator


# ─────────────────────────────────────────────
# Pydantic model for agent output (Step 1.1/1.2)
# ─────────────────────────────────────────────

class AgentOutput(BaseModel):
    """
    Structured output produced by every analyst agent.

    Fields
    ------
    direction       : LONG / SHORT / NEUTRAL
    expected_return : estimated % return over 3-4 day horizon.
                      Clamped to [-0.10, +0.10].
    confidence      : signal quality score. Clamped to [0.1, 1.0].
    reasoning       : free-text explanation (optional).
    """

    direction:       Literal["LONG", "SHORT", "NEUTRAL"]
    expected_return: float = 0.0
    confidence:      float = 0.5
    reasoning:       str   = ""

    @field_validator("expected_return")
    @classmethod
    def clamp_expected_return(cls, v: float) -> float:
        return max(-0.10, min(0.10, v))

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.1, min(1.0, v))


# ─────────────────────────────────────────────
# Reducer helpers
# ─────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    """Shallow-merge two dicts. b values overwrite a on conflict."""
    result = dict(a)
    result.update(b)
    return result


def _keep_first(a: Any, b: Any) -> Any:
    """Keep the first (non-empty) value – used for metadata which only start writes."""
    return a if a else b


# ─────────────────────────────────────────────
# Sub-type aliases
# ─────────────────────────────────────────────

PrefetchedData  = dict[str, dict[str, Any]]
FeedbackHistory = dict[tuple[str, str], list[dict[str, Any]]]
AgentWeights    = dict[tuple[str, str], float]
AnalystSignals  = dict[str, dict[str, Any]]


# ─────────────────────────────────────────────
# Core state
# ─────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    Shared state passed through the LangGraph pipeline.

    Every top-level key has an Annotated reducer so parallel analyst
    nodes can each write without raising InvalidUpdateError.

    Layout
    ------
    data:
        prefetched_data  – filled by DataPrefetchAgent before all analysts
        analyst_signals  – each analyst writes its own entry here
        feedback_history – injected at pipeline start from SQLite
        agent_weights    – injected at pipeline start from SQLite
        risk_output      – written by RiskManagerAgent
        portfolio_output – written by PortfolioManagerAgent

    metadata:
        run_id, tickers, start_ts
    """

    # merge on parallel writes
    data:     Annotated[dict[str, Any], _merge_dicts]
    metadata: Annotated[dict[str, Any], _keep_first]
    messages: Annotated[list[Any], operator.add]


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def make_initial_state(
    run_id: str,
    tickers: list[str],
    start_ts: str,
    feedback_history: Optional[FeedbackHistory] = None,
    agent_weights: Optional[AgentWeights] = None,
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> AgentState:
    """Construct a fresh AgentState for a pipeline run."""
    return AgentState(
        data={
            "prefetched_data":  {},
            "analyst_signals":  {},
            "feedback_history": feedback_history or {},
            "agent_weights":    agent_weights or {},
            "risk_output":      {},
            "portfolio_output": [],
            # Convenience keys read directly by legacy philosophy agents
            "tickers":   tickers,
            "end_date":  end_date or date.today().isoformat(),
            "start_date": start_date,
        },
        metadata={
            "run_id":         run_id,
            "tickers":        tickers,
            "start_ts":       start_ts,
            "show_reasoning": False,
            "model_name":     "claude-sonnet-4-5",
            "model_provider": "Anthropic",
        },
        messages=[],
    )


# ─────────────────────────────────────────────
# Typed accessors
# ─────────────────────────────────────────────

def get_prefetched(state: AgentState, ticker: str) -> dict[str, Any]:
    """Return the prefetched payload for *ticker*, or {} if not yet available."""
    return state.get("data", {}).get("prefetched_data", {}).get(ticker, {})


def get_weight(state: AgentState, agent_id: str, ticker: str, default: float = 1.0) -> float:
    """Return the EWA weight for (agent_id, ticker), falling back to *default*."""
    weights: AgentWeights = state.get("data", {}).get("agent_weights", {})
    return weights.get((agent_id, ticker), default)


def get_feedback(state: AgentState, agent_id: str, ticker: str) -> list[dict[str, Any]]:
    """Return recent prediction history for (agent_id, ticker)."""
    history: FeedbackHistory = state.get("data", {}).get("feedback_history", {})
    return history.get((agent_id, ticker), [])


def set_analyst_signal(
    state: AgentState,
    agent_id: str,
    ticker: str,
    signal: dict[str, Any],
) -> None:
    """Write an analyst signal into state (mutates in place)."""
    data = state.setdefault("data", {})
    signals = data.setdefault("analyst_signals", {})
    if agent_id not in signals:
        signals[agent_id] = {}
    signals[agent_id][ticker] = signal


# ─────────────────────────────────────────────────────────────────────────────
# Display helper (compatibility shim for all analyst agents)
# ─────────────────────────────────────────────────────────────────────────────

def show_agent_reasoning(output: Any, agent_name: str = "") -> None:
    """Print agent reasoning to stdout when show_reasoning is enabled."""
    import json
    header = f"{'='*60}\n{agent_name} Reasoning\n{'='*60}"
    print(header)
    if isinstance(output, dict):
        print(json.dumps(output, indent=2, default=str))
    else:
        print(output)
    print()
