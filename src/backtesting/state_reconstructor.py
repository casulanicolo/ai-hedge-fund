"""
src/backtesting/state_reconstructor.py
──────────────────────────────────────
Builds an AgentState as it would have been visible at `as_of`.

Replaces DataPrefetchAgent / SECFetcher / MacroFetcher in backtest mode.
All data filtered through PointInTimeDataProvider — no future leakage.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from src.backtesting.point_in_time import PointInTimeDataProvider, _to_date
from src.graph.state import AgentState, make_initial_state

logger = logging.getLogger(__name__)


def _empty_payload(fetched_at: str) -> dict[str, Any]:
    """Skeleton matching DataPrefetcher.TickerPayload — empty frames."""
    return {
        "ohlcv_daily":     pd.DataFrame(),
        "ohlcv_weekly":    pd.DataFrame(),
        "ohlcv_4h":        pd.DataFrame(),
        "ohlcv_5m":        pd.DataFrame(),
        "info":            {},
        "income_stmt":     pd.DataFrame(),
        "income_stmt_q":   pd.DataFrame(),
        "balance_sheet":   pd.DataFrame(),
        "balance_sheet_q": pd.DataFrame(),
        "cash_flow":       pd.DataFrame(),
        "cash_flow_q":     pd.DataFrame(),
        "holders":         pd.DataFrame(),
        "fetched_at":      fetched_at,
    }


def _resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    agg = {"Open": "first", "High": "max", "Low": "min",
           "Close": "last", "Volume": "sum"}
    return daily.resample("W-FRI").agg({k: v for k, v in agg.items() if k in daily.columns}).dropna()


def reconstruct_state_at(
    as_of: date | str,
    tickers: list[str],
    *,
    lookback_days: int = 400,
    provider: Optional[PointInTimeDataProvider] = None,
    feedback_history: Optional[dict] = None,
    agent_weights: Optional[dict] = None,
) -> AgentState:
    """
    Build an AgentState mirroring what the live pipeline would have seen
    at end-of-day `as_of`.

    Notes
    -----
    Intraday frames (4h, 5m) left empty — yfinance does not preserve historical
    intraday beyond ~60 days. Backtest agents must operate on daily frames only.
    """
    as_of_d = _to_date(as_of)
    pit = provider or PointInTimeDataProvider()
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    prefetched: dict[str, dict[str, Any]] = {}
    start_d = as_of_d - timedelta(days=lookback_days)

    for t in tickers:
        payload = _empty_payload(fetched_at)
        try:
            daily = pit.get_ohlcv(t, start=start_d, end=as_of_d, as_of=as_of_d)
            payload["ohlcv_daily"]  = daily
            payload["ohlcv_weekly"] = _resample_weekly(daily)
        except Exception as exc:
            logger.warning("OHLCV fetch failed %s @ %s: %s", t, as_of_d, exc)

        try:
            payload["sec_filings"] = {
                **pit.get_filings(t, as_of=as_of_d),
                "xbrl_metrics": _xbrl_from(pit.get_fundamentals(t, as_of=as_of_d)),
            }
        except Exception as exc:
            logger.warning("SEC fetch failed %s @ %s: %s", t, as_of_d, exc)
            payload["sec_filings"] = {}

        prefetched[t] = payload

    try:
        macro = pit.get_macro(as_of_d)
        macro["fetched_at"] = fetched_at
    except Exception as exc:
        logger.warning("macro fetch failed @ %s: %s", as_of_d, exc)
        macro = {}

    state = make_initial_state(
        run_id=f"backtest-{as_of_d.isoformat()}-{uuid.uuid4().hex[:6]}",
        tickers=tickers,
        start_ts=fetched_at,
        feedback_history=feedback_history,
        agent_weights=agent_weights,
        end_date=as_of_d.isoformat(),
        start_date=start_d.isoformat(),
    )
    state["data"]["prefetched_data"] = prefetched
    state["data"]["macro_data"]      = macro
    state["metadata"]["backtest"]    = True
    state["metadata"]["as_of"]       = as_of_d.isoformat()
    return state


def _xbrl_from(fund: dict[str, Any]) -> dict[str, Any]:
    """Project PIT fundamentals dict into the xbrl_metrics shape."""
    if not fund or fund.get("error"):
        return {}
    return {k: v for k, v in fund.items()
            if k in ("revenue", "net_income", "eps_basic",
                     "total_assets", "total_debt", "operating_cf")}
