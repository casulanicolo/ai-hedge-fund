"""
src/graph/graph.py
───────────────────
Athanor Alpha — LangGraph pipeline graph.

Topology
--------

    START
      │
      ▼
    data_prefetch_node          ← fetches yfinance + SEC EDGAR for ALL tickers
      │
      ├─► warren_buffett_node   ─┐
      ├─► ben_graham_node        │
      ├─► charlie_munger_node    │  all analyst nodes run in parallel
      ├─► michael_burry_node     │  (LangGraph fan-out)
      ├─► bill_ackman_node       │
      ├─► cathie_wood_node       │
      ├─► druckenmiller_node     │
      ├─► damodaran_node         │
      ├─► technicals_node        │
      ├─► fundamentals_node      │
      └─► sentiment_node        ─┘
                                 │
                                 ▼
                           risk_manager_node
                                 │
                                 ▼
                         portfolio_manager_node
                                 │
                                 ▼
                           prediction_log_node
                                 │
                                 ▼
                               END

Notes
-----
- Analyst nodes are listed in ANALYST_NODES and fanned out automatically.
- To add/remove an agent, edit ANALYST_NODES only.
- The stub node functions below will be replaced by real implementations
  in later phases; they are importable now so `build_graph()` can be called
  without crashing.
"""

from __future__ import annotations

import logging
from typing import Callable

from langgraph.graph import END, START, StateGraph

from src.graph.state import AgentState

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.WARNING)


# ─────────────────────────────────────────────
# Agent node registry
# ─────────────────────────────────────────────
# Each entry is (node_name, callable).
# Import the real function once it exists; fall back to a stub in the meantime.

def _stub(name: str) -> Callable[[AgentState], AgentState]:
    """Return a no-op stub that logs and passes state through unchanged."""
    def _node(state: AgentState) -> AgentState:
        logger.debug("Stub node %r — not yet implemented", name)
        return state
    _node.__name__ = name
    return _node


def _import_or_stub(module_path: str, func_name: str, node_name: str) -> Callable:
    try:
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name)
        logger.debug("Loaded node %r from %s.%s", node_name, module_path, func_name)
        return fn
    except (ImportError, AttributeError):
        logger.debug("Node %r not yet implemented — using stub", node_name)
        return _stub(node_name)


# Prefetch node (Fase 2.3)
DATA_PREFETCH_NODE = (
    "data_prefetch",
    _import_or_stub("src.agents.data_prefetch", "data_prefetch_agent", "data_prefetch"),
)

# Analyst nodes — order here only affects display; they run in parallel
ANALYST_NODES: list[tuple[str, Callable]] = [
    ("warren_buffett",  _import_or_stub("src.agents.warren_buffett",  "warren_buffett_agent",  "warren_buffett")),
    ("ben_graham",      _import_or_stub("src.agents.ben_graham",      "ben_graham_agent",      "ben_graham")),
    ("charlie_munger",  _import_or_stub("src.agents.charlie_munger",  "charlie_munger_agent",  "charlie_munger")),
    ("michael_burry",   _import_or_stub("src.agents.michael_burry",   "michael_burry_agent",   "michael_burry")),
    ("bill_ackman",     _import_or_stub("src.agents.bill_ackman",     "bill_ackman_agent",     "bill_ackman")),
    ("cathie_wood",     _import_or_stub("src.agents.cathie_wood",     "cathie_wood_agent",     "cathie_wood")),
    ("druckenmiller",   _import_or_stub("src.agents.druckenmiller",   "druckenmiller_agent",   "druckenmiller")),
    ("damodaran",       _import_or_stub("src.agents.damodaran",       "damodaran_agent",       "damodaran")),
    ("technicals",      _import_or_stub("src.agents.technicals",      "technicals_agent",      "technicals")),
    ("fundamentals",    _import_or_stub("src.agents.fundamentals",    "fundamentals_agent",    "fundamentals")),
    ("sentiment",       _import_or_stub("src.agents.sentiment",       "sentiment_agent",       "sentiment")),
]

# Post-analyst nodes
RISK_MANAGER_NODE = (
    "risk_manager",
    _import_or_stub("src.agents.risk_manager", "risk_manager_agent", "risk_manager"),
)

PORTFOLIO_MANAGER_NODE = (
    "portfolio_manager",
    _import_or_stub("src.agents.portfolio_manager", "portfolio_manager_agent", "portfolio_manager"),
)

PREDICTION_LOG_NODE = (
    "prediction_log",
    _import_or_stub("src.feedback.logger", "prediction_log_node", "prediction_log"),
)


# ─────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and compile the full LangGraph pipeline.

    Returns a compiled StateGraph ready for `.invoke()`.
    """
    graph = StateGraph(AgentState)

    # 1. Register all nodes
    prefetch_name, prefetch_fn = DATA_PREFETCH_NODE
    graph.add_node(prefetch_name, prefetch_fn)

    analyst_names = []
    for name, fn in ANALYST_NODES:
        graph.add_node(name, fn)
        analyst_names.append(name)

    risk_name, risk_fn = RISK_MANAGER_NODE
    graph.add_node(risk_name, risk_fn)

    pm_name, pm_fn = PORTFOLIO_MANAGER_NODE
    graph.add_node(pm_name, pm_fn)

    log_name, log_fn = PREDICTION_LOG_NODE
    graph.add_node(log_name, log_fn)

    # 2. Edges
    # START → prefetch
    graph.add_edge(START, prefetch_name)

    # prefetch → all analysts (fan-out / parallel)
    for analyst_name in analyst_names:
        graph.add_edge(prefetch_name, analyst_name)

    # all analysts → risk manager (fan-in)
    for analyst_name in analyst_names:
        graph.add_edge(analyst_name, risk_name)

    # sequential tail
    graph.add_edge(risk_name, pm_name)
    graph.add_edge(pm_name, log_name)
    graph.add_edge(log_name, END)

    return graph.compile()


# ─────────────────────────────────────────────
# Module-level compiled graph (import and use directly)
# ─────────────────────────────────────────────

compiled_graph = build_graph()


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uuid
    from datetime import datetime, timezone

    from src.graph.state import make_initial_state

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    state = make_initial_state(
        run_id=str(uuid.uuid4()),
        tickers=["AAPL", "NVDA"],
        start_ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    result = compiled_graph.invoke(state)
    print("Graph executed successfully.")
    print("Keys in result['data']:", list(result.get("data", {}).keys()))
