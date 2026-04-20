"""
src/backtesting/forward_engine.py
─────────────────────────────────
Forward-only backtest engine — replaces engine.py / trader.py / controller.py.

Loop
----
For each trading day τ in [start, end]:
    1. state ← reconstruct_state_at(τ, tickers)        # PIT data only
    2. state ← compiled_graph.invoke(state)            # signals (no execution)
    3. orders ← portfolio_output.recommendations
    4. portfolio.execute(orders, fill_price=next_open) # T+1 fill
    5. mark to next-day close → equity curve point

Guarantees
----------
- No future leakage (all reads via PointInTimeDataProvider)
- T+1 execution (signal generated at close, filled at next open)
- Slippage applied per asset tier
- Sentiment routed to backtest proxy (no live news)
- Emails + execution_node skipped (metadata flags)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.backtesting.cache_layer import load_state, save_state
from src.backtesting.point_in_time import PointInTimeDataProvider, _to_date
from src.backtesting.slippage import apply_slippage
from src.backtesting.state_reconstructor import reconstruct_state_at

logger = logging.getLogger(__name__)


# ── portfolio ─────────────────────────────────────────────────────────────
@dataclass
class _Position:
    qty: float
    entry_price: float
    side: str        # "LONG" | "SHORT"
    stop_loss: float
    opened_at: str


@dataclass
class _Portfolio:
    cash: float
    positions: dict[str, _Position] = field(default_factory=dict)
    closed_trades: list[dict] = field(default_factory=list)

    def equity(self, marks: dict[str, float]) -> float:
        eq = self.cash
        for t, p in self.positions.items():
            mark = marks.get(t, p.entry_price)
            if p.side == "LONG":
                eq += p.qty * mark
            else:
                # short: profit when mark < entry
                eq += p.qty * (2 * p.entry_price - mark)
        return eq


# ── orchestrator ──────────────────────────────────────────────────────────
class ForwardBacktestEngine:
    """
    Day-by-day forward backtest. Compiled graph passed in to allow tests to
    inject a stub graph; default uses src.graph.graph.compiled_graph.
    """

    def __init__(
        self,
        *,
        tickers: list[str],
        start: date | str,
        end: date | str,
        initial_capital: float = 100_000.0,
        graph: Optional[Any] = None,
        provider: Optional[PointInTimeDataProvider] = None,
        use_cache: bool = True,
        max_position_pct: float = 0.10,    # cap any single position at 10% equity
    ):
        self.tickers          = [t.upper() for t in tickers]
        self.start            = _to_date(start)
        self.end              = _to_date(end)
        self.initial_capital  = float(initial_capital)
        self.use_cache        = use_cache
        self.max_position_pct = max_position_pct
        self.provider         = provider or PointInTimeDataProvider()
        self.portfolio        = _Portfolio(cash=self.initial_capital)
        self.equity_curve: list[dict] = []
        self._graph = graph

    # ── public ─────────────────────────────────────────────────────────
    def run(self) -> dict[str, Any]:
        days = self._trading_days()
        logger.info("ForwardBacktestEngine: %d trading days, %d tickers",
                    len(days), len(self.tickers))

        # Pre-fetch full OHLCV window once (used for next-open fills + marks)
        ohlcv = self._prefetch_ohlcv()

        graph = self._get_graph()

        for i, today in enumerate(days):
            try:
                self._step(today, days[i+1] if i + 1 < len(days) else None,
                           ohlcv, graph)
            except Exception as exc:
                logger.error("step %s failed: %s", today, exc, exc_info=True)

        return self._final_report()

    # ── core loop ──────────────────────────────────────────────────────
    def _step(
        self,
        today: date,
        next_day: Optional[date],
        ohlcv: dict[str, pd.DataFrame],
        graph: Any,
    ) -> None:
        # 1. reconstruct state @ end-of-day today
        state = self._load_or_build_state(today)

        # 2. signals (graph runs but skips execution + emails via metadata)
        try:
            state = graph.invoke(state)
        except Exception as exc:
            logger.error("graph.invoke failed @ %s: %s", today, exc, exc_info=True)
            return

        # 3. extract orders from portfolio_output
        orders = self._extract_orders(state)

        # 4. fill at next_day open (T+1). If no next day → skip fill.
        if next_day is not None:
            self._execute_orders(orders, next_day, ohlcv, today)

        # 5. mark to today's close → record equity
        marks = {t: self._close_on(ohlcv.get(t), today) for t in self.tickers}
        marks = {k: v for k, v in marks.items() if v is not None}
        eq = self.portfolio.equity(marks)
        self.equity_curve.append({"Date": pd.Timestamp(today), "Portfolio Value": eq})

    # ── data plumbing ──────────────────────────────────────────────────
    def _trading_days(self) -> list[date]:
        # Use SPY as the trading-day calendar
        df = self.provider.get_ohlcv("SPY", self.start, self.end, self.end)
        if df.empty:
            return []
        return [d.date() for d in df.index]

    def _prefetch_ohlcv(self) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for t in self.tickers:
            df = self.provider.get_ohlcv(t, self.start, self.end + timedelta(days=5), self.end)
            if not df.empty:
                out[t] = df
        return out

    def _load_or_build_state(self, today: date) -> Any:
        if self.use_cache:
            cached = load_state(today, self.tickers)
            if cached is not None:
                # Update in-flight metadata (signals, etc. are recomputed by graph)
                cached["metadata"]["backtest"] = True
                cached["metadata"]["skip_execution"] = True
                return cached

        state = reconstruct_state_at(today, self.tickers, provider=self.provider)
        state["metadata"]["skip_execution"]      = True
        state["metadata"]["skip_prediction_log"] = True   # no SQLite writes during backtest
        state["metadata"]["skip_time_exit"]      = True
        if self.use_cache:
            save_state(today, self.tickers, state)
        return state

    def _get_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        from src.graph.graph import compiled_graph
        return compiled_graph

    # ── orders + fills ─────────────────────────────────────────────────
    def _extract_orders(self, state: Any) -> list[dict]:
        po = state.get("data", {}).get("portfolio_output", {})
        recs = po.get("recommendations", []) if isinstance(po, dict) else []
        return list(recs)

    def _execute_orders(
        self,
        orders: list[dict],
        next_day: date,
        ohlcv: dict[str, pd.DataFrame],
        signal_day: date,
    ) -> None:
        marks_today = {t: self._close_on(ohlcv.get(t), signal_day) for t in self.tickers}
        equity_now = self.portfolio.equity({k: v for k, v in marks_today.items() if v is not None})

        for order in orders:
            ticker = (order.get("ticker") or "").upper()
            action = (order.get("action") or "HOLD").upper()
            if not ticker or action == "HOLD":
                continue

            df = ohlcv.get(ticker)
            open_px = self._open_on(df, next_day)
            if open_px is None:
                logger.warning("no open price %s on %s — skip", ticker, next_day)
                continue

            if action in ("BUY", "OPEN_LONG", "LONG"):
                self._open_position(ticker, "LONG", order, open_px, equity_now, next_day)
            elif action in ("SHORT", "OPEN_SHORT"):
                self._open_position(ticker, "SHORT", order, open_px, equity_now, next_day)
            elif action in ("CLOSE", "EXIT", "SELL"):
                self._close_position(ticker, open_px, next_day)

    def _open_position(
        self,
        ticker: str,
        side: str,
        order: dict,
        quoted_open: float,
        equity_now: float,
        fill_day: date,
    ) -> None:
        if ticker in self.portfolio.positions:
            return  # already open — no pyramiding in this engine
        size_usd = float(order.get("size_usd") or 0.0)
        if size_usd <= 0:
            sizing_pct = float(order.get("sizing_pct") or 0.0) / 100.0
            size_usd = equity_now * min(sizing_pct, self.max_position_pct)
        if size_usd <= 0:
            return

        # Cap by cash + global position cap
        size_usd = min(size_usd, equity_now * self.max_position_pct, self.portfolio.cash)
        if size_usd <= 0:
            return

        side_buy = "BUY" if side == "LONG" else "SELL"
        slip = apply_slippage(ticker, side_buy, quoted_open, max(1.0, size_usd / quoted_open))
        fill = slip.fill_price
        qty = size_usd / fill
        if qty <= 0:
            return

        self.portfolio.cash -= size_usd
        self.portfolio.positions[ticker] = _Position(
            qty=qty, entry_price=fill, side=side,
            stop_loss=float(order.get("stop_loss") or fill * (0.95 if side == "LONG" else 1.05)),
            opened_at=fill_day.isoformat(),
        )

    def _close_position(self, ticker: str, quoted_open: float, fill_day: date) -> None:
        pos = self.portfolio.positions.pop(ticker, None)
        if pos is None:
            return
        side_close = "SELL" if pos.side == "LONG" else "BUY"
        slip = apply_slippage(ticker, side_close, quoted_open, pos.qty)
        fill = slip.fill_price

        if pos.side == "LONG":
            proceeds = pos.qty * fill
            self.portfolio.cash += proceeds
        else:
            # short close: pay fill, gain (entry - fill) per share above the original cash
            self.portfolio.cash += pos.qty * (2 * pos.entry_price - fill)

        self.portfolio.closed_trades.append({
            "ticker": ticker, "side": pos.side, "qty": pos.qty,
            "entry": pos.entry_price, "exit": fill,
            "stop_loss": pos.stop_loss,
            "opened_at": pos.opened_at, "closed_at": fill_day.isoformat(),
        })

    # ── price helpers ──────────────────────────────────────────────────
    @staticmethod
    def _close_on(df: Optional[pd.DataFrame], d: date) -> Optional[float]:
        if df is None or df.empty:
            return None
        rows = df[df.index.date == d]
        if rows.empty:
            return None
        return float(rows["Close"].iloc[-1])

    @staticmethod
    def _open_on(df: Optional[pd.DataFrame], d: date) -> Optional[float]:
        if df is None or df.empty:
            return None
        rows = df[df.index.date == d]
        if rows.empty:
            return None
        return float(rows["Open"].iloc[0])

    # ── reporting ──────────────────────────────────────────────────────
    def _final_report(self) -> dict[str, Any]:
        from src.backtesting.metrics import PerformanceMetricsCalculator
        from src.backtesting.metrics_extended import compute_extended

        base = PerformanceMetricsCalculator().compute_metrics(self.equity_curve)
        ext  = compute_extended(self.equity_curve, self.portfolio.closed_trades)
        final_equity = self.equity_curve[-1]["Portfolio Value"] if self.equity_curve else self.initial_capital
        total_return_pct = (final_equity / self.initial_capital - 1.0) * 100.0

        return {
            "initial_capital":   self.initial_capital,
            "final_value":       final_equity,
            "total_return_pct":  total_return_pct,
            "n_trading_days":    len(self.equity_curve),
            "n_closed_trades":   len(self.portfolio.closed_trades),
            "n_open_positions":  len(self.portfolio.positions),
            "metrics_base":      base,
            "metrics_extended":  ext,
            "equity_curve":      self.equity_curve,
            "closed_trades":     self.portfolio.closed_trades,
        }
