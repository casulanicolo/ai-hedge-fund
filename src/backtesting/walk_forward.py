"""
src/backtesting/walk_forward.py
───────────────────────────────
IS / OOS walk-forward harness on top of ForwardBacktestEngine.

Default split (Phase 5 spec)
----------------------------
  IS  : 2019-01-02 → 2022-12-30
  OOS : 2023-01-02 → today

Two independent runs, both starting from `capital` cash. Compares Sharpe
decay (IS - OOS) and return retention (OOS / IS). Both metrics align with
Bailey & López de Prado's overfitting heuristics:
  - Sharpe decay > 0.8  → likely overfit
  - Return retention < 0 → sign reversal, definitely overfit
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from src.backtesting.forward_engine import ForwardBacktestEngine

logger = logging.getLogger(__name__)


def run_walk_forward(
    *,
    tickers: list[str],
    capital: float = 100_000.0,
    is_start:  str = "2019-01-02",
    is_end:    str = "2022-12-30",
    oos_start: str = "2023-01-02",
    oos_end:   str = "2025-12-31",
    use_cache: bool = True,
) -> dict[str, Any]:
    logger.info("Walk-forward IS=[%s..%s] OOS=[%s..%s]", is_start, is_end, oos_start, oos_end)

    is_engine = ForwardBacktestEngine(
        tickers=tickers, start=is_start, end=is_end,
        initial_capital=capital, use_cache=use_cache,
    )
    is_report = is_engine.run()

    oos_engine = ForwardBacktestEngine(
        tickers=tickers, start=oos_start, end=oos_end,
        initial_capital=capital, use_cache=use_cache,
    )
    oos_report = oos_engine.run()

    is_sharpe  = is_report["metrics_base"].get("sharpe_ratio")
    oos_sharpe = oos_report["metrics_base"].get("sharpe_ratio")
    sharpe_decay = (is_sharpe - oos_sharpe) if (is_sharpe is not None and oos_sharpe is not None) else None

    is_ret  = is_report.get("total_return_pct")
    oos_ret = oos_report.get("total_return_pct")
    retention: Optional[float] = None
    if is_ret is not None and oos_ret is not None and abs(is_ret) > 1e-9:
        retention = oos_ret / is_ret

    return {
        "is":               is_report,
        "oos":              oos_report,
        "sharpe_decay":     sharpe_decay,
        "return_retention": retention,
    }
