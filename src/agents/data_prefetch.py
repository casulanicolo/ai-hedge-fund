"""
src/agents/data_prefetch.py
────────────────────────────
Athanor Alpha — DataPrefetch LangGraph node.

This is the first node in the pipeline (START → data_prefetch → analysts).
It calls DataPrefetcher.fetch_all() and SECFetcher.fetch_all(), then writes
the merged result into state["data"]["prefetched_data"].

No LLM call is made here — this node is pure I/O.
"""

from __future__ import annotations

import logging
from typing import Any

from src.data.prefetch import DataPrefetcher
from src.data.sec_edgar import SECFetcher
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def data_prefetch_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: batch-fetch all market data + SEC filings before analyst agents run.

    Reads tickers from state["metadata"]["tickers"].
    Writes payload to state["data"]["prefetched_data"].

    Returns a partial state dict (LangGraph merges it via the reducer).
    """
    tickers: list[str] = state.get("metadata", {}).get("tickers", [])

    if not tickers:
        logger.warning("data_prefetch_agent: no tickers found in state metadata — skipping fetch")
        return {}

    logger.info("data_prefetch_agent: fetching data for %d tickers: %s", len(tickers), tickers)

    # ── 1. Market data (yfinance) ─────────────────────────────────────────
    prefetcher = DataPrefetcher.get_instance()
    prefetched = prefetcher.fetch_all(tickers)

    logger.info(
        "data_prefetch_agent: yfinance complete — %d/%d tickers successful",
        len(prefetched), len(tickers),
    )

    # ── 2. SEC EDGAR filings ──────────────────────────────────────────────
    sec_fetcher = SECFetcher()
    sec_data = sec_fetcher.fetch_all(tickers)

    logger.info(
        "data_prefetch_agent: SEC EDGAR complete — %d/%d tickers fetched",
        len(sec_data), len(tickers),
    )

    # ── 3. Merge SEC data into each ticker's payload ──────────────────────
    for ticker in tickers:
        if ticker in prefetched and ticker in sec_data:
            prefetched[ticker]["sec_filings"] = sec_data[ticker]

    return {
        "data": {
            "prefetched_data": prefetched,
        }
    }