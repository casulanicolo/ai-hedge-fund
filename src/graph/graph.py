"""
src/graph/graph.py
──────────────────
Athanor Alpha — LangGraph pipeline graph.

Topology (grafo statico, comportamento condizionale via skip wrapper)
---------------------------------------------------------------------

    START
      │
      ▼
    data_prefetch_node
      │
      ├─► warren_buffett_node   ─┐
      ├─► ben_graham_node        │
      ├─► charlie_munger_node    │  tutti i nodi analyst sono registrati
      ├─► michael_burry_node     │  ma quelli non in active_analyst_nodes
      ├─► bill_ackman_node       │  vengono saltati via skip wrapper
      ├─► cathie_wood_node       │
      ├─► technicals_node        │
      ├─► fundamentals_node      │
      ├─► sentiment_node         │
      └─► breakout_momentum_node ┘
                                 │
                                 ▼
                        devils_advocate_node   ← saltato se skip_devils_advocate=True
                                 │
                                 ▼
                           risk_manager_node   ← saltato se skip_risk_manager=True
                                 │
                                 ▼
                         portfolio_manager_node
                                 │
                                 ▼
                           prediction_log_node ← saltato se skip_prediction_log=True
                                 │
                                 ▼
                               END

Fase 4: Routine Operativa Multi-Slot
-------------------------------------
Il grafo è compilato una sola volta (statico). Il comportamento condizionale
è ottenuto tramite _make_conditional_analyst() e _make_conditional_node():
ogni nodo legge state["metadata"] e decide se eseguire o passare lo stato
invariato. I flag sono scritti in metadata da run_pipeline.py in base al --mode.

Notes
-----
- Analyst nodes are listed in ANALYST_NODES and fanned out automatically.
- To add/remove an agent, edit ANALYST_NODES only.
- devils_advocate runs after all analysts (fan-in) and before risk_manager.
"""

from __future__ import annotations

import logging
from typing import Callable

from langgraph.graph import END, START, StateGraph

from src.graph.state import AgentState

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _make_conditional_analyst(node_name: str, fn: Callable) -> Callable:
    """
    Wrappa un nodo analyst con logica di skip condizionale.

    Il nodo viene eseguito solo se node_name è presente in
    state["metadata"]["active_analyst_nodes"].
    Se il campo non esiste (compatibilità con vecchi stati),
    il nodo viene sempre eseguito.
    """
    def _wrapped(state: AgentState) -> AgentState:
        active = state.get("metadata", {}).get("active_analyst_nodes", None)
        if active is not None and node_name not in active:
            logger.info("SKIP analyst node %r (mode=%s)", node_name,
                        state.get("metadata", {}).get("run_mode", "full"))
            return state
        return fn(state)
    _wrapped.__name__ = node_name
    return _wrapped


def _make_conditional_node(node_name: str, fn: Callable, skip_key: str) -> Callable:
    """
    Wrappa un nodo post-analyst (devils_advocate, risk_manager, prediction_log)
    con logica di skip condizionale basata su un flag booleano in metadata.

    skip_key: chiave in state["metadata"] che se True fa saltare il nodo.
    """
    def _wrapped(state: AgentState) -> AgentState:
        should_skip = state.get("metadata", {}).get(skip_key, False)
        if should_skip:
            logger.info("SKIP node %r (mode=%s, %s=True)", node_name,
                        state.get("metadata", {}).get("run_mode", "full"), skip_key)
            return state
        return fn(state)
    _wrapped.__name__ = node_name
    return _wrapped


# ─────────────────────────────────────────────────────────────────────────────
# Agent node registry
# ─────────────────────────────────────────────────────────────────────────────

# Prefetch node — gira sempre in tutti i mode
DATA_PREFETCH_NODE = (
    "data_prefetch",
    _import_or_stub("src.agents.data_prefetch", "data_prefetch_agent", "data_prefetch"),
)

# Analyst nodes — tutti registrati nel grafo, ma con skip wrapper
# Per aggiungere/rimuovere un agente: modifica solo questa lista
_ALL_ANALYST_NODES: list[tuple[str, Callable]] = [
    ("warren_buffett",    _import_or_stub("src.agents.warren_buffett",    "warren_buffett_agent",       "warren_buffett")),
    ("ben_graham",        _import_or_stub("src.agents.ben_graham",        "ben_graham_agent",           "ben_graham")),
    ("charlie_munger",    _import_or_stub("src.agents.charlie_munger",    "charlie_munger_agent",       "charlie_munger")),
    ("michael_burry",     _import_or_stub("src.agents.michael_burry",     "michael_burry_agent",        "michael_burry")),
    ("bill_ackman",       _import_or_stub("src.agents.bill_ackman",       "bill_ackman_agent",          "bill_ackman")),
    ("cathie_wood",       _import_or_stub("src.agents.cathie_wood",       "cathie_wood_agent",          "cathie_wood")),
    ("technicals",        _import_or_stub("src.agents.technicals",        "technical_analyst_agent",    "technicals")),
    ("fundamentals",      _import_or_stub("src.agents.fundamentals",      "fundamentals_analyst_agent", "fundamentals")),
    ("sentiment",         _import_or_stub("src.agents.sentiment",         "sentiment_agent",            "sentiment")),
    ("breakout_momentum", _import_or_stub("src.agents.breakout_momentum", "breakout_momentum_agent",    "breakout_momentum")),
]

# Applica il wrapper condizionale a ogni analyst node
ANALYST_NODES: list[tuple[str, Callable]] = [
    (name, _make_conditional_analyst(name, fn))
    for name, fn in _ALL_ANALYST_NODES
]

# Fase 3: Devil's Advocate — saltato se skip_devils_advocate=True in metadata
DEVILS_ADVOCATE_NODE = (
    "devils_advocate",
    _make_conditional_node(
        "devils_advocate",
        _import_or_stub("src.agents.devils_advocate", "devils_advocate_agent", "devils_advocate"),
        skip_key="skip_devils_advocate",
    ),
)

# Risk Manager — saltato se skip_risk_manager=True in metadata
RISK_MANAGER_NODE = (
    "risk_manager",
    _make_conditional_node(
        "risk_manager",
        _import_or_stub("src.agents.risk_manager", "risk_manager_agent", "risk_manager"),
        skip_key="skip_risk_manager",
    ),
)

# Portfolio Manager — gira sempre (è il cuore del review mode)
PORTFOLIO_MANAGER_NODE = (
    "portfolio_manager",
    _import_or_stub("src.agents.portfolio_manager", "portfolio_manager_agent", "portfolio_manager"),
)

# Prediction Log — saltato se skip_prediction_log=True in metadata
PREDICTION_LOG_NODE = (
    "prediction_log",
    _make_conditional_node(
        "prediction_log",
        _import_or_stub("src.feedback.logger", "prediction_log_node", "prediction_log"),
        skip_key="skip_prediction_log",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assembla e compila il grafo LangGraph.

    Il grafo è sempre lo stesso (statico): tutti i nodi sono registrati.
    Il comportamento condizionale per i diversi --mode è gestito
    internamente dai wrapper _make_conditional_analyst e _make_conditional_node.
    """
    graph = StateGraph(AgentState)

    # 1. Registra tutti i nodi
    prefetch_name, prefetch_fn = DATA_PREFETCH_NODE
    graph.add_node(prefetch_name, prefetch_fn)

    analyst_names = []
    for name, fn in ANALYST_NODES:
        graph.add_node(name, fn)
        analyst_names.append(name)

    da_name, da_fn = DEVILS_ADVOCATE_NODE
    graph.add_node(da_name, da_fn)

    risk_name, risk_fn = RISK_MANAGER_NODE
    graph.add_node(risk_name, risk_fn)

    pm_name, pm_fn = PORTFOLIO_MANAGER_NODE
    graph.add_node(pm_name, pm_fn)

    log_name, log_fn = PREDICTION_LOG_NODE
    graph.add_node(log_name, log_fn)

    # 2. Edges (struttura fissa — il comportamento condizionale è nei wrapper)
    # START → prefetch
    graph.add_edge(START, prefetch_name)

    # prefetch → tutti gli analyst (fan-out)
    for analyst_name in analyst_names:
        graph.add_edge(prefetch_name, analyst_name)

    # tutti gli analyst → devils_advocate (fan-in)
    for analyst_name in analyst_names:
        graph.add_edge(analyst_name, da_name)

    # devils_advocate → risk_manager → portfolio_manager → prediction_log → END
    graph.add_edge(da_name, risk_name)
    graph.add_edge(risk_name, pm_name)
    graph.add_edge(pm_name, log_name)
    graph.add_edge(log_name, END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level compiled graph (import and use directly)
# ─────────────────────────────────────────────────────────────────────────────

compiled_graph = build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

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
    state["metadata"]["model_name"] = "claude-sonnet-4-5"
    state["metadata"]["model_provider"] = "Anthropic"

    result = compiled_graph.invoke(state)
    print("Graph executed successfully.")
    print("Keys in result['data']:", list(result.get("data", {}).keys()))
