"""
src/graph/execution_node.py
────────────────────────────
LangGraph node — wraps TradeExecutor.

Skipped when state["metadata"]["skip_execution"] is True (default).
Only runs when pipeline is invoked with --live flag.
"""

from __future__ import annotations

import logging
from typing import Any

from src.execution.orders import TradeOrder
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def execution_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node — execute TradeOrders via Alpaca."""

    # ── Skip check (default: skip=True, only runs with --live) ───────────────
    if state.get("metadata", {}).get("skip_execution", True):
        logger.info("[execution_node] skip_execution=True — no orders submitted")
        return {}

    data    = state.get("data", {})
    raw     = data.get("trade_orders", [])
    run_id  = state.get("metadata", {}).get("run_id", "unknown")

    if not raw:
        logger.info("[execution_node] No trade_orders in state — nothing to execute")
        return {"data": {"execution_report": {"run_id": run_id, "skipped": 0, "results": []}}}

    # ── Deserialise TradeOrder objects (may be Pydantic or dicts) ────────────
    orders: list[TradeOrder] = []
    for item in raw:
        if isinstance(item, TradeOrder):
            orders.append(item)
        elif isinstance(item, dict):
            try:
                orders.append(TradeOrder(**item))
            except Exception as exc:
                logger.warning("[execution_node] Skipping malformed order: %s", exc)
        else:
            logger.warning("[execution_node] Unknown order type: %s", type(item))

    if not orders:
        logger.warning("[execution_node] All orders failed deserialisation — nothing to execute")
        return {}

    # ── Build adapter + executor (imported here to avoid loading alpaca-py when skipped) ──
    try:
        from src.execution.alpaca_adapter import AlpacaBrokerAdapter
        from src.execution.executor import TradeExecutor

        adapter  = AlpacaBrokerAdapter()
        executor = TradeExecutor(adapter=adapter, run_id=run_id)
        report   = executor.execute(orders)

        logger.info(
            "[execution_node] ExecutionReport — submitted=%d rejected=%d skipped=%d",
            report.submitted, report.rejected, report.skipped,
        )
        return {"data": {"execution_report": report.model_dump()}}

    except Exception as exc:
        logger.error("[execution_node] Execution failed: %s", exc, exc_info=True)
        return {
            "data": {
                "execution_report": {
                    "run_id":  run_id,
                    "errors":  [str(exc)],
                    "skipped": len(orders),
                }
            }
        }
