"""
dashboard/data_sources.py
─────────────────────────
Single I/O layer for the Streamlit dashboard.

Every reader the tabs need lives here behind an @st.cache_data wrapper
(or @st.cache_resource for the broker connection). Tabs stay "stupid":
they call ds.<getter>() and render — no SQL, no HTTP, no path math.

Cache policy
------------
- live broker reads (Alpaca account/positions/orders) ........ ttl =  60s
- on-disk artefacts (SQLite, runs.jsonl, alpaca.jsonl) ....... ttl = 300s
- macro snapshot (VIX/yields) ................................ ttl =  60s
- backtest progress files (polled while running) ............. ttl =   2s
- backtest run index (filesystem listing) .................... ttl =  30s
- kill switch (cheap stat, must be live) ..................... no cache

Empty-state contract
--------------------
Every getter MUST return a safe shape (empty DataFrame / empty list /
None / zeroed dict) when the underlying data is missing. Tabs are
allowed to assume the *shape* but not the *content*.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

DB_PATH               = ROOT / "db"        / "hedge_fund.db"
RUNS_LOG_PATH         = ROOT / "logs"      / "runs.jsonl"
ALPACA_LOG_PATH       = ROOT / "logs"      / "alpaca.jsonl"
MONITOR_LOG_PATH      = ROOT / "logs"      / "monitor.log"
BACKTEST_RESULTS_DIR  = ROOT / "results"
BACKTEST_RUNS_DIR     = ROOT / "backtests" / "runs"        # saved runs (Tab 4 "Save Run")
BACKTEST_PROGRESS_DIR = ROOT / "backtests" / "progress"    # <label>.progress.json
KILL_SWITCH_PATH      = ROOT / ".athanor_kill"


# ─────────────────────────────────────────────────────────────────────────────
# TTL constants (single source of truth — tweak in one place)
# ─────────────────────────────────────────────────────────────────────────────

TTL_LIVE       = 60     # broker, macro, audit
TTL_DISK       = 300    # SQLite, accuracy, attribution
TTL_BACKTEST   = 30     # backtest run listing
TTL_PROGRESS   = 2      # backtest progress polling


# ─────────────────────────────────────────────────────────────────────────────
# Broker connection (singleton)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_broker():
    """
    Return the live AlpacaBrokerAdapter. Cached for the dashboard process.
    Returns None when credentials are missing or the SDK fails to construct
    — tabs render an "Alpaca offline" badge instead of crashing.
    """
    try:
        from src.execution.alpaca_adapter import AlpacaBrokerAdapter
        return AlpacaBrokerAdapter()
    except Exception as exc:
        log.warning("broker init failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SQLite helper
# ─────────────────────────────────────────────────────────────────────────────

def _sql(query: str, params: tuple = ()) -> pd.DataFrame:
    """Run a parameterised SELECT and return a DataFrame.
    Returns empty df on missing DB or SQL error."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as con:
            return pd.read_sql_query(query, con, params=params)
    except Exception as exc:
        log.warning("SQL failed [%s]: %s", query[:60], exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW   (filled in next file)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_account_snapshot() -> dict:
    """Broker account: AUM, cash, buying power, status. {} when offline."""
    broker = get_broker()
    if broker is None:
        return {}
    try:
        a = broker.get_account()
        return {
            "account_id":     a.account_id,
            "aum":            a.portfolio_value,
            "cash":           a.cash,
            "buying_power":   a.buying_power,
            "status":         a.status,
        }
    except Exception as exc:
        log.warning("get_account_snapshot failed: %s", exc)
        return {}


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_equity_curve(days: int = 365) -> pd.DataFrame:
    """Daily portfolio equity for last `days`. Columns: ['date','equity'].
    Source: Alpaca /v2/account/portfolio/history. Empty df → no history yet."""
    broker = get_broker()
    if broker is None:
        return pd.DataFrame(columns=["date", "equity"])
    try:
        # alpaca-py exposes get_portfolio_history on TradingClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(period=f"{max(1, days)}D", timeframe="1D")
        hist = broker._client.get_portfolio_history(history_filter=req)
        ts  = list(getattr(hist, "timestamp", []) or [])
        eq  = list(getattr(hist, "equity",    []) or [])
        if not ts or not eq:
            return pd.DataFrame(columns=["date", "equity"])
        df = pd.DataFrame({
            "date":   pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
            "equity": pd.to_numeric(eq, errors="coerce"),
        }).dropna(subset=["equity"])
        df = df.reset_index(drop=True)
        # Append live today point if not already present
        today = pd.Timestamp(datetime.now().date())
        if df.empty or df["date"].iloc[-1] < today:
            live_aum = float(acct.get("aum", 0.0)) if (acct := get_account_snapshot()) else 0.0
            if live_aum > 0:
                live_row = pd.DataFrame({"date": [today], "equity": [live_aum]})
                df = pd.concat([df, live_row], ignore_index=True)
        return df
    except Exception as exc:
        log.warning("get_equity_curve failed: %s", exc)
        return pd.DataFrame(columns=["date", "equity"])


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_benchmark(symbol: str = "SPY", days: int = 365) -> pd.DataFrame:
    """Benchmark close series via yfinance. Columns: ['date','close']."""
    try:
        import yfinance as yf
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days + 7)
        df = yf.download(symbol, start=start.isoformat(), end=end.isoformat(),
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty or "Close" not in df.columns:
            return pd.DataFrame(columns=["date", "close"])
        out = df[["Close"]].copy().reset_index()
        out.columns = ["date", "close"]
        out["date"]  = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        return out.dropna(subset=["close"]).reset_index(drop=True)
    except Exception as exc:
        log.warning("get_benchmark(%s) failed: %s", symbol, exc)
        return pd.DataFrame(columns=["date", "close"])


def _filter_range(df: pd.DataFrame, range_pill: str) -> pd.DataFrame:
    """Slice an equity/benchmark df by Tab-1 range pill (1Y/6M/3M/1M/WTD)."""
    if df.empty:
        return df
    end  = pd.Timestamp(df["date"].max())
    if range_pill == "WTD":
        start = end - pd.Timedelta(days=end.weekday())
    else:
        days = {"1Y": 365, "6M": 182, "3M": 92, "1M": 31}.get(range_pill, 365)
        start = end - pd.Timedelta(days=days)
    return df[df["date"] >= start].reset_index(drop=True)


def _sharpe(returns: pd.Series, rf_daily: float = 0.0) -> Optional[float]:
    r = returns.dropna()
    if len(r) < 5 or r.std(ddof=0) == 0:
        return None
    return float(((r.mean() - rf_daily) / r.std(ddof=0)) * (252 ** 0.5))


def _max_drawdown(equity: pd.Series) -> Optional[float]:
    e = equity.dropna()
    if e.empty:
        return None
    return float(((e / e.cummax()) - 1.0).min() * 100.0)


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_overview_kpis() -> dict:
    """
    Composite Tab-1 cards. Keys:
      aum, pl_ytd, pl_ytd_pct, sharpe_1y, max_dd_1y,
      win_rate_20, alpha_vs_spy_ytd
    All missing → None / 0.0.
    """
    out = {
        "aum":              0.0,
        "pl_ytd":           0.0,
        "pl_ytd_pct":       0.0,
        "pl_since_date":    None,
        "sharpe_1y":        None,
        "max_dd_1y":        None,
        "win_rate_20":      None,
        "alpha_vs_spy_ytd": None,
    }

    acct = get_account_snapshot()
    out["aum"] = float(acct.get("aum", 0.0))

    eq = get_equity_curve(days=365)
    if not eq.empty:
        eq_year = eq.copy()
        eq_nonzero = eq_year[eq_year["equity"] > 0]
        if not eq_nonzero.empty:
            last = float(eq_nonzero["equity"].iloc[-1])
            inception_date = eq_nonzero["date"].iloc[0]
            days_of_history = (eq_nonzero["date"].iloc[-1] - inception_date).days
            if days_of_history < 365:
                base = float(eq_nonzero["equity"].iloc[0])
            else:
                year_ago = eq_nonzero["date"].iloc[-1] - pd.Timedelta(days=365)
                base_row = eq_nonzero[eq_nonzero["date"] >= year_ago]
                base = float(base_row["equity"].iloc[0]) if not base_row.empty else float(eq_nonzero["equity"].iloc[0])
            out["pl_ytd"]        = last - base
            out["pl_ytd_pct"]    = ((last / base) - 1.0) * 100.0 if base > 0 else 0.0
            out["pl_since_date"] = str(inception_date.date()) if days_of_history < 365 else str((eq_nonzero["date"].iloc[-1] - pd.Timedelta(days=365)).date())

        rets = eq_year["equity"].pct_change()
        out["sharpe_1y"] = _sharpe(rets)
        out["max_dd_1y"] = _max_drawdown(eq_year["equity"])

        # Alpha vs SPY YTD
        spy = get_benchmark("SPY", days=365)
        if not spy.empty and not eq_nonzero.empty:
            inception_date = eq_nonzero["date"].iloc[0]
            spy_since = spy[(spy["date"] >= inception_date) & (spy["close"] > 0)]
            if not spy_since.empty:
                spy_ret = (spy_since["close"].iloc[-1] / spy_since["close"].iloc[0] - 1.0) * 100.0
                out["alpha_vs_spy_ytd"] = out["pl_ytd_pct"] - float(spy_ret)

    # Win rate (realized PnL) — last 20 closed trades from executed_orders
    df = _sql(
        """
        SELECT ticker, action, fill_price, quantity, filled_at
          FROM executed_orders
         WHERE status IN ('FILLED','PARTIAL_FILLED') AND filled_at IS NOT NULL
         ORDER BY filled_at DESC
         LIMIT 200
        """
    )
    if not df.empty:
        # Pair OPEN_LONG/OPEN_SHORT with CLOSE on the same ticker (FIFO).
        wins = losses = 0
        opens: dict[str, list[tuple[float, str]]] = {}
        for _, r in df[::-1].iterrows():     # chronological
            act = (r["action"] or "").upper()
            tk  = r["ticker"]
            px  = float(r["fill_price"] or 0.0)
            if act in ("OPEN_LONG", "OPEN_SHORT"):
                opens.setdefault(tk, []).append((px, act))
            elif act in ("CLOSE", "SELL", "EXIT"):
                stack = opens.get(tk) or []
                if not stack:
                    continue
                entry_px, side = stack.pop(0)
                pnl = (px - entry_px) if side == "OPEN_LONG" else (entry_px - px)
                if pnl > 0: wins += 1
                else:       losses += 1
        n = wins + losses
        if n > 0:
            recent = min(20, n)
            out["win_rate_20"] = (wins / n) * 100.0  # share of all closed trades
            out["_n_closed"]   = recent              # for tooltip

    return out


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_exposure_breakdown() -> dict:
    """{long_pct, short_pct, cash_pct, net_pct, gross_pct, beta_proxy}."""
    out = {"long_pct": 0.0, "short_pct": 0.0, "cash_pct": 100.0,
           "net_pct": 0.0, "gross_pct": 0.0, "beta_proxy": None}

    acct = get_account_snapshot()
    aum = float(acct.get("aum", 0.0))
    cash = float(acct.get("cash", 0.0))
    if aum <= 0:
        return out

    broker = get_broker()
    if broker is None:
        return out
    try:
        positions = broker.get_positions()
    except Exception as exc:
        log.warning("exposure positions fetch: %s", exc)
        return out

    long_mv  = sum(p.market_value for p in positions if p.market_value > 0)
    short_mv = sum(-p.market_value for p in positions if p.market_value < 0)

    out["long_pct"]  = (long_mv  / aum) * 100.0
    out["short_pct"] = (short_mv / aum) * 100.0
    out["cash_pct"]  = max(0.0, (cash / aum) * 100.0)
    out["net_pct"]   = out["long_pct"] - out["short_pct"]
    out["gross_pct"] = out["long_pct"] + out["short_pct"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — POSITIONS & ORDERS
# ─────────────────────────────────────────────────────────────────────────────

def _count_trading_days(start: datetime, end: datetime) -> int:
    """Count Mon-Fri days between start and end (exclusive of end date). Stdlib only."""
    from datetime import timedelta
    if end <= start:
        return 0
    days = 0
    current = start.date()
    end_date = end.date()
    while current < end_date:
        if current.weekday() < 5:
            days += 1
        current += timedelta(days=1)
    return days


def _latest_open_order_row(ticker: str) -> dict:
    """Most recent OPEN_LONG/OPEN_SHORT FILLED row for a ticker (for SL/TP/days)."""
    df = _sql(
        """
        SELECT action, fill_price, stop_loss, take_profit, filled_at, run_id
          FROM executed_orders
         WHERE ticker = ?
           AND action IN ('OPEN_LONG','OPEN_SHORT')
           AND status IN ('FILLED','PARTIAL_FILLED')
         ORDER BY filled_at DESC
         LIMIT 1
        """,
        (ticker,),
    )
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return row


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_positions() -> pd.DataFrame:
    """Live positions enriched with SL/TP/tag/days_held from executed_orders."""
    cols = ["ticker","side","qty","entry","mark","unrealized_pl",
            "unrealized_pct","days_held","stop_loss","take_profit","dist_sl","dist_tp","tag"]
    broker = get_broker()
    if broker is None:
        return pd.DataFrame(columns=cols)
    try:
        positions = broker.get_positions()
    except Exception as exc:
        log.warning("get_positions failed: %s", exc)
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    now_utc = datetime.now(timezone.utc)
    for p in positions:
        qty       = float(p.qty)
        entry     = float(p.avg_entry_price)
        mark      = float(p.market_value) / qty if qty else entry
        unr_pl    = float(p.unrealized_pl)
        unr_pct   = (unr_pl / (entry * abs(qty))) * 100.0 if entry and qty else 0.0
        meta      = _latest_open_order_row(p.ticker)
        filled_at = meta.get("filled_at")

        # Parse SL/TP — guard against strings or None
        sl: Optional[float] = None
        tp: Optional[float] = None
        try:
            raw_sl = meta.get("stop_loss")
            if raw_sl is not None:
                sl = float(raw_sl)
        except (TypeError, ValueError):
            pass
        try:
            raw_tp = meta.get("take_profit")
            if raw_tp is not None:
                tp = float(raw_tp)
        except (TypeError, ValueError):
            pass

        days_held: Optional[int] = None
        if filled_at:
            try:
                opened_dt = pd.to_datetime(filled_at, utc=True).to_pydatetime()
                days_held = _count_trading_days(opened_dt, now_utc)
            except Exception:
                days_held = None

        dist_sl: Optional[float] = None
        dist_tp: Optional[float] = None
        if sl is not None and mark:
            try:
                dist_sl = (mark - sl) / mark * 100.0
            except (ZeroDivisionError, TypeError):
                pass
        if tp is not None and mark:
            try:
                dist_tp = (tp - mark) / mark * 100.0
            except (ZeroDivisionError, TypeError):
                pass

        rows.append({
            "ticker":         p.ticker,
            "side":           p.side.upper(),
            "qty":            qty,
            "entry":          entry,
            "mark":           mark,
            "unrealized_pl":  unr_pl,
            "unrealized_pct": unr_pct,
            "days_held":      days_held,
            "stop_loss":      sl,
            "take_profit":    tp,
            "dist_sl":        dist_sl,
            "dist_tp":        dist_tp,
            "tag":            meta.get("run_id", "")[:8] if meta else "",
        })
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_positions_summary() -> dict:
    """Compact header dict: {aum, cash, buying_power, n_positions}.

    Always returns a populated dict — zeroes when broker is offline or flat —
    so the summary strip in Tab 2 can render unconditionally.
    """
    acct      = get_account_snapshot()
    positions = get_positions()
    n_pos     = 0 if positions is None or positions.empty else int(len(positions))
    return {
        "aum":           float(acct.get("aum")          or 0.0),
        "cash":          float(acct.get("cash")         or 0.0),
        "buying_power":  float(acct.get("buying_power") or 0.0),
        "n_positions":   n_pos,
    }


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_open_orders() -> pd.DataFrame:
    """Pending orders from broker."""
    cols = ["ticker","side","type","qty","limit","submitted_at","status"]
    broker = get_broker()
    if broker is None:
        return pd.DataFrame(columns=cols)
    try:
        orders = broker.get_open_orders()
    except Exception as exc:
        log.warning("get_open_orders failed: %s", exc)
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for o in orders:
        # Best-effort: alpaca-py exposes order_type + limit_price on the raw
        # order; OrderSnapshot stays minimal so look at action for side hint.
        side = "BUY" if (o.action or "").upper().startswith(("BUY","OPEN_LONG")) else "SELL"
        rows.append({
            "ticker":       o.ticker,
            "side":         side,
            "type":         "—",                     # not exposed in OrderSnapshot
            "qty":          o.qty,
            "limit":        o.filled_avg_price,      # placeholder until field added
            "submitted_at": o.submitted_at,
            "status":       o.status,
        })
    return pd.DataFrame(rows, columns=cols)


_RULE_LABELS: dict[str, str] = {
    "time_decay":         "⏱ Time Decay",
    "trailing_breakeven": "📈 Trailing Stop",
    "trailing_1x":        "📈 Trailing Stop",
    "trailing_1_5x":      "📈 Trailing Stop",
    "stop_loss":          "🛑 Stop Loss",
    "setup_broken_rsi":   "🛑 Setup Broken",
    "take_profit":        "🎯 Take Profit",
    "hold":               "— Hold",
}


def _reconcile_pending_orders() -> None:
    """For all non-terminal orders with a broker_order_id, fetch real status from Alpaca and update DB."""
    if not DB_PATH.exists():
        return
    try:
        with sqlite3.connect(DB_PATH) as con:
            pending = pd.read_sql_query(
                """SELECT id, broker_order_id, ticker, action FROM executed_orders
                   WHERE broker_order_id IS NOT NULL
                     AND status NOT IN ('FILLED','PARTIAL_FILLED','REJECTED','CANCELLED','EXPIRED')""",
                con,
            )
        if pending.empty:
            return
        broker = get_broker()
        if broker is None:
            return
        for _, row in pending.iterrows():
            try:
                order = broker._client.get_order_by_id(row["broker_order_id"])
                status_raw = str(order.status).upper()
                if "FILLED" in status_raw and "PARTIAL" not in status_raw:
                    our_status = "FILLED"
                elif "PARTIAL" in status_raw:
                    our_status = "PARTIAL_FILLED"
                elif "CANCEL" in status_raw:
                    our_status = "CANCELLED"
                elif "EXPIRED" in status_raw:
                    our_status = "EXPIRED"
                else:
                    continue
                fill_price = float(order.filled_avg_price or 0) or None
                filled_at  = str(order.filled_at) if order.filled_at else None
                filled_qty = float(order.filled_qty or 0) or None
                with sqlite3.connect(DB_PATH) as con:
                    con.execute(
                        """UPDATE executed_orders
                              SET status    = ?,
                                  fill_price = ?,
                                  filled_at  = ?,
                                  quantity   = COALESCE(?, quantity)
                            WHERE id = ?""",
                        (our_status, fill_price, filled_at, filled_qty, row["id"]),
                    )
            except Exception as exc:
                log.warning("reconcile order %s failed: %s", row.get("broker_order_id"), exc)
    except Exception as exc:
        log.warning("_reconcile_pending_orders failed: %s", exc)

    # ── Step 2: import Alpaca-side conditional fills (SL/TP) not yet in DB ──
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        broker = get_broker()
        if broker is None or not hasattr(broker, "_client"):
            return

        after = datetime.now(timezone.utc) - timedelta(days=7)
        alpaca_orders = broker._client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500, after=after)
        )

        def _norm_status(s) -> str:
            if hasattr(s, "name"):
                return s.name.upper()
            return str(s).split(".")[-1].upper()

        with sqlite3.connect(DB_PATH) as con:
            known_ids = {
                r[0]
                for r in con.execute(
                    "SELECT broker_order_id FROM executed_orders WHERE broker_order_id IS NOT NULL"
                ).fetchall()
            }
            # ensure alpaca-sync run_id exists
            con.execute(
                "INSERT OR IGNORE INTO pipeline_runs (run_id, started_at, status, tickers)"
                " VALUES ('alpaca-sync', ?, 'completed', '[]')",
                (datetime.now(timezone.utc).isoformat(),),
            )
            con.commit()

            for o in alpaca_orders:
                oid = str(o.id)
                if oid in known_ids:
                    continue
                status = _norm_status(o.status)
                if status != "FILLED":
                    continue
                fill_price = float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else None
                if not fill_price:
                    continue
                side_str = str(getattr(o, "side", "")).split(".")[-1].lower()
                # sell closes a long; buy closes a short
                if side_str == "sell":
                    open_action = "OPEN_LONG"
                elif side_str == "buy":
                    open_action = "OPEN_SHORT"
                else:
                    continue
                ticker = o.symbol
                filled_at_str = o.filled_at.isoformat() if getattr(o, "filled_at", None) else None
                filled_qty = float(o.filled_qty) if getattr(o, "filled_qty", None) else None
                submitted_at_str = (
                    o.submitted_at.isoformat() if getattr(o, "submitted_at", None)
                    else datetime.now(timezone.utc).isoformat()
                )

                open_row = con.execute(
                    """SELECT run_id, stop_loss, take_profit FROM executed_orders
                       WHERE ticker = ? AND action = ? AND status = 'FILLED'
                       ORDER BY filled_at DESC LIMIT 1""",
                    (ticker, open_action),
                ).fetchone()
                if open_row is None:
                    continue
                run_id   = open_row[0] or "alpaca-sync"
                stop_loss   = open_row[1]
                take_profit = open_row[2]

                con.execute(
                    """INSERT OR IGNORE INTO executed_orders
                           (run_id, broker_order_id, ticker, action,
                            quantity, notional_usd, fill_price,
                            stop_loss, take_profit, status,
                            submitted_at, filled_at,
                            rejection_reason, raw_alpaca_response)
                       VALUES (?, ?, ?, 'CLOSE', ?, NULL, ?, ?, ?, 'FILLED', ?, ?, NULL, '{}')""",
                    (run_id, oid, ticker,
                     int(filled_qty) if filled_qty else None,
                     fill_price, stop_loss, take_profit,
                     submitted_at_str, filled_at_str),
                )
                con.commit()
                log.info("[reconcile] imported conditional close %s %s broker=%s fill=%.4f",
                         open_action.replace("OPEN_", ""), ticker, oid[:8], fill_price)
    except Exception as exc:
        log.warning("_reconcile_pending_orders step2 failed: %s", exc)


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_trade_history_alpaca(days: int = 30) -> pd.DataFrame:
    """Complete trade history via SQLite LEFT JOIN pairing. Returns rich columns."""
    _COLS = ["ticker","entry_date","exit_date","pl","pl_pct","quantity",
             "entry_price","exit_price","side","close_reason","stop_loss","take_profit","top_agents"]
    try:
        _reconcile_pending_orders()

        df = _sql(
            """
            SELECT
                o1.ticker,
                o1.action          AS open_action,
                o1.fill_price      AS entry_price,
                o1.quantity,
                o1.filled_at       AS entry_date,
                o1.stop_loss,
                o1.take_profit,
                o2.fill_price      AS exit_price,
                o2.filled_at       AS exit_date,
                o2.action          AS close_action,
                o2.broker_order_id AS close_order_id
            FROM executed_orders o1
            LEFT JOIN executed_orders o2
                ON o2.broker_order_id = (
                    SELECT broker_order_id
                    FROM executed_orders
                    WHERE ticker = o1.ticker
                      AND action IN ('CLOSE','SELL','EXIT','FULL_EXIT','SCALE_OUT')
                      AND status IN ('FILLED','PARTIAL_FILLED')
                      AND datetime(filled_at) > datetime(o1.filled_at)
                    ORDER BY filled_at ASC
                    LIMIT 1
                )
            WHERE o1.action IN ('OPEN_LONG','OPEN_SHORT')
              AND o1.status IN ('FILLED','PARTIAL_FILLED')
              AND datetime(o1.filled_at) >= datetime('now', '-{days} day')
            ORDER BY o1.filled_at DESC
            """.format(days=int(days)),
        )
        if df.empty:
            return pd.DataFrame(columns=_COLS)

        # Load all actionable monitor_ticks for close_reason lookup
        close_reasons_df = _sql(
            """
            SELECT ticker, timestamp, rule, reason
              FROM monitor_ticks
             WHERE action_taken = 1
             ORDER BY timestamp DESC
            """,
        )
        # Build per-ticker list of (ts, rule) sorted descending for fast lookup
        ticks_by_ticker: dict[str, list[tuple]] = {}
        if not close_reasons_df.empty:
            close_reasons_df["_ts"] = pd.to_datetime(
                close_reasons_df["timestamp"], errors="coerce", utc=True
            )
            for _, t in close_reasons_df.dropna(subset=["_ts"]).iterrows():
                tkr = str(t["ticker"])
                ticks_by_ticker.setdefault(tkr, []).append(
                    (t["_ts"], str(t.get("rule") or ""))
                )

        rows: list[dict] = []
        seen: set[tuple] = set()
        for _, r in df.iterrows():
            exit_px_raw = r.get("exit_price")
            if exit_px_raw is None or pd.isna(exit_px_raw):
                continue
            entry_px   = float(r["entry_price"] or 0.0)
            exit_px    = float(exit_px_raw)
            qty        = float(r["quantity"] or 0.0)
            sl_f       = float(r["stop_loss"])   if r.get("stop_loss")   is not None and not pd.isna(r["stop_loss"])   else None
            tp_f       = float(r["take_profit"]) if r.get("take_profit") is not None and not pd.isna(r["take_profit"]) else None
            side       = "long" if str(r.get("open_action","")).upper() == "OPEN_LONG" else "short"
            ticker     = str(r["ticker"])
            entry_date = r.get("entry_date")
            exit_date  = r.get("exit_date")

            key = (ticker, str(entry_date), str(exit_date))
            if key in seen:
                continue
            seen.add(key)

            pl     = (exit_px - entry_px) * qty if side == "long" else (entry_px - exit_px) * qty
            pl_pct = (pl / (entry_px * qty) * 100.0) if entry_px and qty else 0.0

            # Determine close_reason from monitor_ticks.rule
            close_reason = "—"
            exit_ts = pd.to_datetime(exit_date, errors="coerce", utc=True) if exit_date else None
            if exit_ts is not None and not pd.isna(exit_ts) and ticker in ticks_by_ticker:
                best_dt: Optional[float] = None
                best_rule = ""
                for ts, rule in ticks_by_ticker[ticker]:
                    diff = (exit_ts - ts).total_seconds()
                    if 0 <= diff <= 600 and (best_dt is None or diff < best_dt):
                        best_dt, best_rule = diff, rule
                if best_rule:
                    close_reason = _RULE_LABELS.get(best_rule, best_rule.replace("_", " ").title())

            # Fallback heuristics when no matching tick found
            if close_reason == "—":
                if sl_f is not None:
                    if (side == "long"  and exit_px <= sl_f) or \
                       (side == "short" and exit_px >= sl_f):
                        close_reason = "🛑 Stop Loss"
                if close_reason == "—" and tp_f is not None:
                    if (side == "long"  and exit_px >= tp_f) or \
                       (side == "short" and exit_px <= tp_f):
                        close_reason = "🎯 Take Profit"

            rows.append({
                "ticker":       ticker,
                "entry_date":   entry_date,
                "exit_date":    exit_date,
                "pl":           pl,
                "pl_pct":       pl_pct,
                "quantity":     qty,
                "entry_price":  entry_px,
                "exit_price":   exit_px,
                "side":         side,
                "close_reason": close_reason,
                "stop_loss":    sl_f,
                "take_profit":  tp_f,
                "top_agents":   [],
            })

        return pd.DataFrame(rows, columns=_COLS) if rows else pd.DataFrame(columns=_COLS)
    except Exception as exc:
        log.warning("get_trade_history_alpaca failed: %s", exc)
        return pd.DataFrame(columns=_COLS)


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_trade_history(days: int = 30) -> pd.DataFrame:
    """Realised trades. Primary: LEFT JOIN pairing. Fallback: FIFO matching."""
    result = get_trade_history_alpaca(days=days)
    if not result.empty:
        return result

    # Fallback — minimal FIFO matching (legacy)
    cols = ["ticker","entry_date","exit_date","pl","pl_pct","quantity",
            "entry_price","exit_price","side","close_reason","stop_loss","take_profit","top_agents"]
    df = _sql(
        """
        SELECT ticker, action, fill_price, quantity, filled_at
          FROM executed_orders
         WHERE status IN ('FILLED','PARTIAL_FILLED')
           AND filled_at IS NOT NULL
           AND datetime(filled_at) >= datetime('now', ? )
         ORDER BY filled_at ASC
        """,
        (f"-{int(days)} day",),
    )
    if df.empty:
        return pd.DataFrame(columns=cols)

    open_stack: dict[str, list[dict]] = {}
    closed: list[dict] = []
    for _, r in df.iterrows():
        act = (r["action"] or "").upper()
        tk  = r["ticker"]
        px  = float(r["fill_price"] or 0.0)
        qty = float(r["quantity"]   or 0.0)
        ts  = r["filled_at"]
        if act in ("OPEN_LONG", "OPEN_SHORT"):
            open_stack.setdefault(tk, []).append(
                {"side": act, "px": px, "qty": qty, "ts": ts}
            )
        elif act in ("CLOSE", "SELL", "EXIT"):
            stack = open_stack.get(tk) or []
            if not stack:
                continue
            entry = stack.pop(0)
            pnl    = ((px - entry["px"]) if entry["side"] == "OPEN_LONG"
                      else (entry["px"] - px)) * min(qty, entry["qty"])
            pl_pct = (pnl / (entry["px"] * entry["qty"])) * 100.0 if entry["px"] and entry["qty"] else 0.0
            side   = "long" if entry["side"] == "OPEN_LONG" else "short"
            closed.append({
                "ticker":       tk,
                "entry_date":   entry["ts"],
                "exit_date":    ts,
                "pl":           pnl,
                "pl_pct":       pl_pct,
                "quantity":     entry["qty"],
                "entry_price":  entry["px"],
                "exit_price":   px,
                "side":         side,
                "close_reason": "MANUAL",
                "stop_loss":    None,
                "take_profit":  None,
                "top_agents":   [],
            })

    return pd.DataFrame(closed, columns=cols)


def get_trade_stats(df: pd.DataFrame) -> dict:
    """Compute advanced statistics from a trade history DataFrame."""
    empty: dict = {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "win_rate": None, "total_pl": 0.0, "avg_pl_per_trade": None,
        "avg_win": None, "avg_loss": None, "profit_factor": None,
        "max_win_streak": 0, "max_loss_streak": 0,
        "best_trade": None, "worst_trade": None,
        "stats_by_ticker": {},
    }
    if df is None or df.empty or "pl" not in df.columns:
        return empty

    pl      = pd.to_numeric(df["pl"], errors="coerce").fillna(0.0)
    tickers = df.get("ticker", pd.Series(dtype=str))

    wins   = pl[pl > 0]
    losses = pl[pl < 0]
    total  = len(pl)
    n_win  = int(len(wins))
    n_loss = int(len(losses))

    max_win = max_loss = cur_win = cur_loss = 0
    for v in pl:
        if v > 0:
            cur_win += 1; cur_loss = 0
            max_win = max(max_win, cur_win)
        elif v < 0:
            cur_loss += 1; cur_win = 0
            max_loss = max(max_loss, cur_loss)
        else:
            cur_win = cur_loss = 0

    best_trade: Optional[dict]  = None
    worst_trade: Optional[dict] = None
    if total > 0:
        best_idx  = int(pl.reset_index(drop=True).idxmax())
        worst_idx = int(pl.reset_index(drop=True).idxmin())
        tks = tickers.reset_index(drop=True)
        best_trade  = {"ticker": str(tks.iloc[best_idx])  if len(tks) > best_idx  else "?",
                       "pl":     float(pl.iloc[best_idx])}
        worst_trade = {"ticker": str(tks.iloc[worst_idx]) if len(tks) > worst_idx else "?",
                       "pl":     float(pl.iloc[worst_idx])}

    stats_by_ticker: dict[str, dict] = {}
    if not tickers.empty:
        tmp = pd.DataFrame({"ticker": tickers.values, "pl": pl.values})
        for tkr, grp in tmp.groupby("ticker"):
            n = len(grp)
            w = int((grp["pl"] > 0).sum())
            stats_by_ticker[str(tkr)] = {
                "n":        n,
                "win_rate": float(w / n * 100.0) if n > 0 else None,
                "total_pl": float(grp["pl"].sum()),
            }
        stats_by_ticker = dict(
            sorted(stats_by_ticker.items(), key=lambda x: x[1]["total_pl"], reverse=True)
        )

    return {
        "total_trades":    total,
        "winning_trades":  n_win,
        "losing_trades":   n_loss,
        "win_rate":        float(n_win / total * 100.0) if total > 0 else None,
        "total_pl":        float(pl.sum()),
        "avg_pl_per_trade": float(pl.mean()) if total > 0 else None,
        "avg_win":         float(wins.mean()) if n_win  > 0 else None,
        "avg_loss":        float(losses.mean()) if n_loss > 0 else None,
        "profit_factor":   (float(wins.sum()) / float(abs(losses.sum())))
                           if n_loss > 0 and losses.sum() != 0 else None,
        "max_win_streak":  max_win,
        "max_loss_streak": max_loss,
        "best_trade":      best_trade,
        "worst_trade":     worst_trade,
        "stats_by_ticker": stats_by_ticker,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — AGENT ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_DIR = {"BUY": 1.0, "LONG": 1.0, "SELL": -1.0, "SHORT": -1.0,
               "HOLD": 0.0, "NEUTRAL": 0.0}


def _signal_to_dir(s) -> float:
    if s is None:
        return 0.0
    return _SIGNAL_DIR.get(str(s).upper().strip(), 0.0)


def _pretty_agent(a) -> str:
    """Strip trailing _agent and convert snake → Title Case for display."""
    if a is None:
        return ""
    name = str(a)
    if name.endswith("_agent"):
        name = name[:-6]
    return name.replace("_", " ").title()


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_agent_weights() -> pd.DataFrame:
    """One row per agent: latest weight averaged across tickers + 30d trend.

    trend: latest_avg_weight − avg_weight 30 days ago (positive = trending up).
    Columns: agent_id, agent_name, weight_avg, weight_trend, n_tickers
    """
    cols = ["agent_id","agent_name","weight_avg","weight_trend","n_tickers"]
    df = _sql("SELECT agent_id, ticker, weight, updated_at FROM agent_weights")
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["updated_at"])

    # Latest weight per (agent, ticker)
    latest = (df.sort_values("updated_at")
                .groupby(["agent_id","ticker"], as_index=False)
                .tail(1))
    cur = (latest.groupby("agent_id")
                  .agg(weight_avg=("weight","mean"), n_tickers=("ticker","nunique"))
                  .reset_index())

    # 30d-ago snapshot for trend
    cutoff = df["updated_at"].max() - pd.Timedelta(days=30)
    past = df[df["updated_at"] <= cutoff]
    if past.empty:
        cur["weight_trend"] = 0.0
    else:
        past_latest = (past.sort_values("updated_at")
                            .groupby(["agent_id","ticker"], as_index=False)
                            .tail(1))
        past_avg = past_latest.groupby("agent_id")["weight"].mean().rename("past_w")
        cur = cur.merge(past_avg, on="agent_id", how="left")
        cur["weight_trend"] = (cur["weight_avg"] - cur["past_w"]).fillna(0.0)
        cur = cur.drop(columns=["past_w"])

    cur["agent_name"] = cur["agent_id"].map(_pretty_agent)
    return cur[cols].sort_values("weight_avg", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_agent_accuracy(window: str = "5d") -> pd.DataFrame:
    """Accuracy by agent for window in {1d, 5d, 20d}.

    Hit = sign(signal_dir) == sign(actual_return). HOLD predictions skipped.
    Columns: agent_id, agent_name, accuracy, n_predictions, avg_confidence
    """
    cols = ["agent_id","agent_name","accuracy","n_predictions","avg_confidence"]
    if window not in ("1d","5d","20d"):
        window = "5d"
    return_col = f"actual_return_{window}"

    df = _sql(f"""
        SELECT p.agent_id, p.signal, p.confidence,
               o.{return_col} AS actual_return
          FROM predictions p
          JOIN outcomes  o ON o.prediction_id = p.id
         WHERE o.{return_col} IS NOT NULL
    """)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["dir"] = df["signal"].map(_signal_to_dir)
    df = df[df["dir"] != 0.0]                 # skip HOLD
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["hit"] = ((df["dir"] * df["actual_return"]) > 0).astype(int)
    g = (df.groupby("agent_id")
           .agg(n_predictions=("hit","count"),
                accuracy=("hit","mean"),
                avg_confidence=("confidence","mean"))
           .reset_index())
    g["accuracy"]       = g["accuracy"]       * 100.0   # percent
    g["avg_confidence"] = g["avg_confidence"] * 100.0   # percent
    g["agent_name"]     = g["agent_id"].map(_pretty_agent)
    return g[cols].sort_values("accuracy", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_attribution_pl_ytd(window: str = "5d") -> pd.DataFrame:
    """Signed weighted attribution proxy.

    contribution_bps = signal_dir × weight × actual_return × 10_000

    The result is a basis-point proxy of P&L impact per agent×ticker, NOT a
    dollar realized P&L (which lives in executed_orders). Useful for ranking
    which agent×ticker decisions were directionally accretive YTD.

    Columns: agent_id, agent_name, ticker, contribution_bps, n_predictions
    """
    cols = ["agent_id","agent_name","ticker","contribution_bps","n_predictions"]
    if window not in ("1d","5d","20d"):
        window = "5d"
    return_col = f"actual_return_{window}"

    df = _sql(f"""
        SELECT p.agent_id, p.ticker, p.signal,
               o.{return_col} AS actual_return,
               p.timestamp
          FROM predictions p
          JOIN outcomes  o ON o.prediction_id = p.id
         WHERE o.{return_col} IS NOT NULL
    """)
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if not df.empty:
        ytd_start = pd.Timestamp(year=df["timestamp"].max().year, month=1, day=1, tz="UTC")
        df = df[df["timestamp"] >= ytd_start]
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Latest weight per (agent, ticker) — defaults to 1.0 when missing
    w = _sql("SELECT agent_id, ticker, weight, updated_at FROM agent_weights")
    if not w.empty:
        w["updated_at"] = pd.to_datetime(w["updated_at"], errors="coerce", utc=True)
        w = (w.sort_values("updated_at")
              .groupby(["agent_id","ticker"], as_index=False)
              .tail(1))[["agent_id","ticker","weight"]]
        df = df.merge(w, on=["agent_id","ticker"], how="left")
    else:
        df["weight"] = 1.0
    df["weight"] = df["weight"].fillna(1.0)

    df["dir"]    = df["signal"].map(_signal_to_dir)
    df["contrib"] = df["dir"] * df["weight"] * df["actual_return"] * 10_000.0

    g = (df.groupby(["agent_id","ticker"])
           .agg(contribution_bps=("contrib","sum"),
                n_predictions=("contrib","count"))
           .reset_index())
    g["agent_name"] = g["agent_id"].map(_pretty_agent)
    return g[cols].sort_values("contribution_bps", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_weight_trend(days: int = 90) -> pd.DataFrame:
    """Daily mean weight per agent over the last `days` days.

    Columns: date, agent_id, agent_name, weight
    """
    cols = ["date","agent_id","agent_name","weight"]
    df = _sql("SELECT agent_id, ticker, weight, updated_at FROM agent_weights")
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["updated_at"])
    cutoff = df["updated_at"].max() - pd.Timedelta(days=int(days))
    df = df[df["updated_at"] >= cutoff]
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Drop tz + time component → tz-naive datetime64[ns] aligned to midnight,
    # so Plotly renders "%b %d" ticks cleanly (no sub-second junk on the axis).
    df["date"] = (df["updated_at"]
                    .dt.tz_convert("UTC")
                    .dt.normalize()
                    .dt.tz_localize(None))
    g = (df.groupby(["date","agent_id"])["weight"]
           .mean()
           .reset_index())
    g["date"]       = pd.to_datetime(g["date"])           # explicit datetime64[ns]
    g["agent_name"] = g["agent_id"].map(_pretty_agent)
    return g[cols].sort_values(["agent_id","date"]).reset_index(drop=True)


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_struggling_agents(window: str = "5d", min_predictions: int = 5) -> pd.DataFrame:
    """Agents with accuracy < 50% AND at least `min_predictions` predictions.

    Columns: agent_id, agent_name, accuracy, n_predictions, weight_avg
    """
    cols = ["agent_id","agent_name","accuracy","n_predictions","weight_avg"]
    acc = get_agent_accuracy(window=window)
    if acc.empty:
        return pd.DataFrame(columns=cols)
    weights = get_agent_weights()
    w_lookup = (weights.set_index("agent_id")["weight_avg"]
                if not weights.empty else pd.Series(dtype=float))
    out = acc[(acc["accuracy"] < 50.0) & (acc["n_predictions"] >= min_predictions)].copy()
    out["weight_avg"] = out["agent_id"].map(w_lookup).fillna(0.0)
    return out[cols].sort_values("accuracy").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────

def _slug(label: str) -> str:
    """Filesystem-safe label."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label.strip())
    return safe or "run"


def _backtest_log_path(label: str) -> Path:
    return ROOT / "logs" / f"backtest_{_slug(label)}.log"


def _backtest_progress_path(label: str) -> Path:
    BACKTEST_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    return BACKTEST_PROGRESS_DIR / f"{_slug(label)}.progress.json"


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        import os
        if os.name == "nt":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
                return bool(ok) and code.value == STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(h)
        else:
            os.kill(int(pid), 0)
            return True
    except Exception:
        return False


def _scan_results_dir(directory: Path, source: str) -> list[dict]:
    """Read every *.json under `directory`, attach metadata for the runs list."""
    if not directory.exists():
        return []
    out: list[dict] = []
    for p in sorted(directory.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("bad backtest report %s: %s", p, exc)
            continue
        out.append({
            "path":       str(p),
            "name":       p.stem,
            "source":     source,                 # "live" | "saved"
            "modified":   datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                                  .strftime("%Y-%m-%d %H:%M"),
            "size_kb":    round(p.stat().st_size / 1024.0, 1),
            "metrics":    payload.get("metrics") or payload.get("performance") or {},
            "label":      payload.get("label") or p.stem,
            "start":      payload.get("start"),
            "end":        payload.get("end"),
            "tickers":    payload.get("tickers"),
        })
    return out


@st.cache_data(ttl=TTL_BACKTEST, show_spinner=False)
def list_saved_backtests() -> list[dict]:
    """Index of every backtest report. Merges:
       - results/      → CLI auto-saved (source="live")
       - backtests/runs/ → user-saved via the dashboard (source="saved")"""
    return (_scan_results_dir(BACKTEST_RUNS_DIR,    "saved")
            + _scan_results_dir(BACKTEST_RESULTS_DIR, "live"))


@st.cache_data(ttl=TTL_PROGRESS, show_spinner=False)
def get_backtest_progress(label: str) -> dict:
    """Read backtests/progress/<label>.progress.json. Empty when not running.

    Schema written by launch_backtest():
      pid, status (running|done|error|cancelled), started_at,
      finished_at, exit_code, log_path, args (list[str])
    """
    p = _backtest_progress_path(label)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("bad progress %s: %s", p, exc)
        return {}

    # Reconcile: if status==running but pid is dead, mark stale.
    if data.get("status") == "running" and not _pid_alive(int(data.get("pid") or 0)):
        data["status"] = "stale"
    return data


@st.cache_data(ttl=TTL_PROGRESS, show_spinner=False)
def tail_backtest_log(label: str, n: int = 80) -> list[str]:
    """Last N lines of logs/backtest_<label>.log."""
    p = _backtest_log_path(label)
    if not p.exists():
        return []
    try:
        with p.open(encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip() for ln in lines[-n:]]
    except Exception as exc:
        log.warning("log tail failed %s: %s", p, exc)
        return []


def is_backtest_running(label: str) -> bool:
    """True iff the progress file says 'running' AND the pid is alive."""
    prog = get_backtest_progress(label)
    return prog.get("status") == "running" and _pid_alive(int(prog.get("pid") or 0))


def launch_backtest(params: dict) -> dict:
    """Spawn `python -m src.backtesting.cli` as a detached subprocess.

    Required keys in `params`:
      label, start, end, tickers (str|list), capital, no_cache (bool),
      walk_forward (bool), is_start, is_end, oos_start, oos_end

    Returns the progress dict the dashboard polls afterwards.
    Raises RuntimeError if a run with the same label is already in flight.
    """
    import subprocess, sys, os

    label = _slug(params.get("label") or "run")

    if is_backtest_running(label):
        raise RuntimeError(
            f"Backtest '{label}' is already running. Cancel it first.")

    # Build CLI args
    tickers = params.get("tickers")
    if isinstance(tickers, (list, tuple)):
        tickers = ",".join(tickers)

    cmd: list[str] = [sys.executable, "-m", "src.backtesting.cli",
                      "--label", label]
    if params.get("walk_forward"):
        cmd.append("--walk-forward")
        for k_arg, k_param in (("--is-start","is_start"), ("--is-end","is_end"),
                                ("--oos-start","oos_start"), ("--oos-end","oos_end")):
            if params.get(k_param):
                cmd += [k_arg, str(params[k_param])]
    else:
        if params.get("start"): cmd += ["--start", str(params["start"])]
        if params.get("end"):   cmd += ["--end",   str(params["end"])]

    if tickers:                  cmd += ["--tickers", str(tickers)]
    if params.get("capital"):    cmd += ["--capital", str(params["capital"])]
    if params.get("no_cache"):   cmd.append("--no-cache")

    log_path = _backtest_log_path(label)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")

    creationflags = 0
    if os.name == "nt":
        # Detach so the subprocess survives a Streamlit rerun.
        creationflags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                         | getattr(subprocess, "DETACHED_PROCESS",         0))

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT,
        creationflags=creationflags,
        close_fds=(os.name != "nt"),
    )

    progress = {
        "label":       label,
        "pid":         proc.pid,
        "status":      "running",
        "started_at":  datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code":   None,
        "log_path":    str(log_path),
        "args":        cmd,
    }
    _backtest_progress_path(label).write_text(
        json.dumps(progress, indent=2), encoding="utf-8")

    # Bust caches so the next render sees the new run.
    get_backtest_progress.clear()
    list_saved_backtests.clear()
    return progress


def cancel_backtest(label: str) -> bool:
    """Best-effort terminate of a running backtest. Returns True on signal sent."""
    import os, signal as _signal
    prog = get_backtest_progress(label)
    pid  = int(prog.get("pid") or 0)
    if not pid or not _pid_alive(pid):
        return False
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.TerminateProcess(
                ctypes.windll.kernel32.OpenProcess(0x0001, False, pid), 1)
        else:
            os.kill(pid, _signal.SIGTERM)
    except Exception as exc:
        log.warning("cancel_backtest %s: %s", label, exc)
        return False

    prog.update({
        "status":      "cancelled",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    _backtest_progress_path(label).write_text(
        json.dumps(prog, indent=2), encoding="utf-8")
    get_backtest_progress.clear()
    return True


def reconcile_finished_backtest(label: str) -> dict:
    """If pid is dead but status is still 'running', stamp status from the log."""
    prog = get_backtest_progress(label)
    if not prog or prog.get("status") != "stale":
        return prog
    lines = tail_backtest_log(label, n=200)
    err = any("Traceback" in ln or "Error" in ln for ln in lines)
    prog.update({
        "status":      "error" if err else "done",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    _backtest_progress_path(label).write_text(
        json.dumps(prog, indent=2), encoding="utf-8")
    get_backtest_progress.clear()
    list_saved_backtests.clear()
    return prog


def load_backtest_report(path: str) -> dict:
    """Read full report JSON from disk. {} on failure."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("load_backtest_report %s: %s", path, exc)
        return {}


def save_backtest_run(source_path: str, *, new_label: Optional[str] = None) -> str:
    """Copy a `results/` report into `backtests/runs/` so it survives a clean.
       Returns the destination path. Idempotent on the same source."""
    import shutil
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(source_path)
    BACKTEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    dst_name = (_slug(new_label) + ".json") if new_label else src.name
    dst = BACKTEST_RUNS_DIR / dst_name
    shutil.copy2(src, dst)

    # Companion equity CSV (CLI writes alongside the JSON)
    eq_src = src.with_name(src.stem + "_equity.csv")
    if eq_src.exists():
        shutil.copy2(eq_src, dst.with_name(dst.stem + "_equity.csv"))

    list_saved_backtests.clear()
    return str(dst)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — REGIME & RISK
# ─────────────────────────────────────────────────────────────────────────────

def _yf_recent_close(symbol: str, days: int = 30) -> pd.Series:
    """Best-effort yfinance close series. Empty Series on any failure."""
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=f"{max(days,5)}d", auto_adjust=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df["Close"].astype(float).dropna()
    except Exception as exc:
        log.warning("yfinance %s failed: %s", symbol, exc)
        return pd.Series(dtype=float)


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_macro_snapshot() -> dict:
    """VIX + UST 10Y/3M curve + qualitative regime label.

    Regime mapping:
      VIX < 15  → CALM         (green)
      15..25    → NORMAL       (cyan)
      25..35    → ELEVATED     (orange)
      ≥ 35      → CRISIS       (red)

    Curve label uses the 10Y − 3M spread (in basis points).
    """
    out = {
        "vix":          None, "vix_trend":   [],
        "regime":      "UNKNOWN", "regime_color": "#999",
        "ust_10y":      None, "ust_3m":      None,
        "curve_spread": None, "curve_label": "—",
    }
    vix = _yf_recent_close("^VIX", days=30)
    if not vix.empty:
        v = float(vix.iloc[-1])
        out["vix"] = v
        out["vix_trend"] = [float(x) for x in vix.tail(20).tolist()]
        if   v < 15: out["regime"], out["regime_color"] = "CALM",     "#27ae60"
        elif v < 25: out["regime"], out["regime_color"] = "NORMAL",   "#3ec1d3"
        elif v < 35: out["regime"], out["regime_color"] = "ELEVATED", "#e8a83a"
        else:        out["regime"], out["regime_color"] = "CRISIS",   "#ff3b3b"

    tnx = _yf_recent_close("^TNX", days=10)   # 10Y yield ×10
    irx = _yf_recent_close("^IRX", days=10)   # 13-week T-bill ×10
    if not tnx.empty:
        out["ust_10y"] = float(tnx.iloc[-1]) / 10.0
    if not irx.empty:
        out["ust_3m"]  = float(irx.iloc[-1]) / 10.0
    if out["ust_10y"] is not None and out["ust_3m"] is not None:
        spread_bps = (out["ust_10y"] - out["ust_3m"]) * 100.0
        out["curve_spread"] = spread_bps
        out["curve_label"]  = "INVERTED" if spread_bps < 0 else "STEEP" if spread_bps > 100 else "FLAT"
    return out


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_correlation_matrix(tickers: tuple[str, ...], days: int = 20) -> pd.DataFrame:
    """Pairwise close-to-close return correlation over the last `days` sessions."""
    if not tickers or len(tickers) < 2:
        return pd.DataFrame()
    closes = {}
    for sym in tickers:
        s = _yf_recent_close(sym, days=days + 5)
        if not s.empty:
            closes[sym] = s
    if len(closes) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(closes).dropna(how="any").tail(days + 1)
    rets = df.pct_change().dropna(how="all")
    if rets.shape[0] < 3:
        return pd.DataFrame()
    return rets.corr()


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_var_1d_95(lookback_days: int = 250) -> dict:
    """Historical 1-day 95% VaR computed off the portfolio equity curve.

    Returns {var_usd, var_pct, method, lookback_days, n_returns}.
    `var_pct` and `var_usd` are reported as positive magnitudes (loss).
    """
    out = {"var_usd": 0.0, "var_pct": 0.0,
           "method": "historical", "lookback_days": lookback_days, "n_returns": 0}
    eq = get_equity_curve(days=lookback_days)
    if eq is None or eq.empty or "equity" not in eq.columns:
        return out
    series = pd.to_numeric(eq["equity"], errors="coerce").dropna()
    if len(series) < 5:
        return out
    import numpy as np
    rets = (series.pct_change()
                  .replace([np.inf, -np.inf], np.nan)
                  .dropna())
    if len(rets) < 5:
        out["n_returns"] = int(len(rets))
        return out
    q = float(rets.quantile(0.05))      # negative number
    out["var_pct"]   = abs(q) * 100.0
    out["var_usd"]   = abs(q) * float(series.iloc[-1])
    out["n_returns"] = int(len(rets))
    return out


# Static linear sensitivity scenarios — v1 (no factor regression yet).
# delta_equity = sum_over_positions( notional × beta_to_factor × shock )
# Per-position beta defaults below; refine when we wire a real factor model.
_STRESS_SCENARIOS: tuple[dict, ...] = (
    {"scenario": "SPY −5%",         "factor": "spy",  "shock": -0.05},
    {"scenario": "SPY +5%",         "factor": "spy",  "shock":  0.05},
    {"scenario": "VIX +50%",        "factor": "vix",  "shock":  0.50},
    {"scenario": "10Y yield +50bp", "factor": "rate", "shock":  0.005},
    {"scenario": "Tech sector −8%", "factor": "tech", "shock": -0.08},
)


def _factor_beta(ticker: str, factor: str) -> float:
    """Crude default betas. v1 — replaces a real factor regression later."""
    t = (ticker or "").upper()
    if factor == "spy":
        return 1.0
    if factor == "vix":
        return -0.20
    if factor == "rate":
        return -3.0 if t in {"TLT","IEF","IEI","TLH"} else -0.5
    if factor == "tech":
        return 1.0 if t in {"AAPL","NVDA","MSFT","GOOGL","META","AMZN","TSLA","AMD",
                            "QQQ","XLK","SOXX"} else 0.0
    return 0.0


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_stress_tests() -> pd.DataFrame:
    """Linear factor approximation (v1).

    Columns: scenario, delta_equity_usd, delta_equity_pct, worst_positions
    `worst_positions` is the list of the 3 most-impacted tickers per scenario.
    """
    cols = ["scenario","delta_equity_usd","delta_equity_pct","worst_positions"]
    positions = get_positions()
    acct      = get_account_snapshot()
    aum       = float((acct or {}).get("aum") or 0.0)
    if positions is None or positions.empty:
        return pd.DataFrame(columns=cols)

    qty   = pd.to_numeric(positions.get("qty",  pd.Series()), errors="coerce").fillna(0.0)
    mark  = pd.to_numeric(positions.get("mark", pd.Series()), errors="coerce").fillna(0.0)
    side  = positions.get("side", pd.Series(dtype=str)).astype(str).str.upper()
    sign  = side.map(lambda s: -1.0 if s == "SHORT" else 1.0)
    notional = qty.abs() * mark * sign     # signed exposure

    rows: list[dict] = []
    for sc in _STRESS_SCENARIOS:
        impacts = []
        for tkr, n in zip(positions["ticker"].astype(str), notional):
            beta = _factor_beta(tkr, sc["factor"])
            d_usd = float(n) * beta * sc["shock"]
            impacts.append((tkr, d_usd))

        impacts.sort(key=lambda x: x[1])           # most-negative first
        delta_usd = sum(d for _, d in impacts)
        rows.append({
            "scenario":         sc["scenario"],
            "delta_equity_usd": delta_usd,
            "delta_equity_pct": (delta_usd / aum * 100.0) if aum > 0 else 0.0,
            "worst_positions":  [{"ticker": t, "delta_usd": d} for t, d in impacts[:3]],
        })
    return pd.DataFrame(rows, columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — HEALTH & AUDIT
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_pipeline_runs_24h() -> pd.DataFrame:
    """Pipeline runs from the last 24h. Adds a derived duration_s column."""
    df = _sql(
        """
        SELECT run_id, started_at, finished_at, status, error_msg
          FROM pipeline_runs
         WHERE datetime(started_at) >= datetime('now','-1 day')
         ORDER BY started_at DESC
        """
    )
    if df.empty:
        return pd.DataFrame(columns=[
            "run_id","started_at","finished_at","duration_s","status","error_msg"])
    s = pd.to_datetime(df["started_at"],  errors="coerce", utc=True)
    e = pd.to_datetime(df["finished_at"], errors="coerce", utc=True)
    df["duration_s"] = (e - s).dt.total_seconds().round(1)
    return df[["run_id","started_at","finished_at","duration_s","status","error_msg"]]


def _today_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _scan_runs_jsonl_today(predicate) -> int:
    """Count lines in logs/runs.jsonl from today (UTC) where `predicate(obj)` is True."""
    if not RUNS_LOG_PATH.exists():
        return 0
    today = _today_utc_iso()
    n = 0
    try:
        with RUNS_LOG_PATH.open(encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or today not in ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if predicate(obj):
                    n += 1
    except Exception as exc:
        log.warning("scan runs.jsonl failed: %s", exc)
    return n


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_email_status_today() -> dict:
    """Email send counts for today (UTC).

    Sources:
      - logs/runs.jsonl  → events with action ∈ {email_sent, premarket_sent,
                            postmarket_sent, critical_alert_sent}
    """
    today = _today_utc_iso()

    def is_kind(obj, *needles):
        text = (str(obj.get("action") or "") + " " +
                str(obj.get("event")  or "") + " " +
                str(obj.get("kind")   or "")).lower()
        return any(n in text for n in needles)

    pre  = _scan_runs_jsonl_today(lambda o: is_kind(o, "premarket"))
    post = _scan_runs_jsonl_today(lambda o: is_kind(o, "postmarket"))
    crit = _scan_runs_jsonl_today(lambda o: is_kind(o, "critical", "kill", "alert"))
    return {
        "date":            today,
        "premarket_sent":  int(pre),
        "postmarket_sent": int(post),
        "critical_sent":   int(crit),
    }


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_alpaca_metrics_24h() -> dict:
    """Latency + error stats from logs/alpaca.jsonl (last 24h)."""
    out = {"p50_ms": None, "p95_ms": None, "errors": 0, "requests": 0,
           "last_call_at": None}
    if not ALPACA_LOG_PATH.exists():
        return out

    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    lats: list[float] = []
    errs = 0
    last_ts = None
    try:
        with ALPACA_LOG_PATH.open(encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                ts_raw = obj.get("ts") or obj.get("timestamp")
                try:
                    ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
                except Exception:
                    ts = None
                if ts is None or pd.isna(ts) or ts.to_pydatetime() < cutoff:
                    continue
                last_ts = max(last_ts, ts) if last_ts is not None else ts
                lat = obj.get("latency_ms")
                if isinstance(lat, (int, float)):
                    lats.append(float(lat))
                if obj.get("error") or str(obj.get("status","")).lower() in {"error","fail","failed"}:
                    errs += 1
    except Exception as exc:
        log.warning("alpaca.jsonl scan failed: %s", exc)

    if lats:
        s = pd.Series(lats)
        out["p50_ms"] = float(s.quantile(0.50))
        out["p95_ms"] = float(s.quantile(0.95))
    out["requests"]     = len(lats) + errs
    out["errors"]       = errs
    out["last_call_at"] = last_ts.strftime("%Y-%m-%d %H:%M:%S") if last_ts is not None else None
    return out


@st.cache_data(ttl=TTL_DISK, show_spinner=False)
def get_llm_cost_month() -> dict:
    """Best-effort LLM spend for the current month, parsed from runs.jsonl
       fields {tokens_in, tokens_out, model}. Returns None for fields without
       enough signal so the UI shows '—' instead of misleading zeros."""
    out = {"anthropic_usd": None, "openai_usd": None, "projected_usd": None,
           "tokens_in": 0, "tokens_out": 0, "calls": 0}
    if not RUNS_LOG_PATH.exists():
        return out

    month = datetime.now(timezone.utc).strftime("%Y-%m")
    anthro = openai = 0.0
    tin = tout = calls = 0
    seen_cost = False

    try:
        with RUNS_LOG_PATH.open(encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or month not in ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                model = str(obj.get("model") or "").lower()
                cost  = obj.get("cost_usd")
                if isinstance(cost, (int, float)):
                    seen_cost = True
                    if "claude" in model or "anthropic" in model: anthro += float(cost)
                    elif "gpt"  in model or "openai"    in model: openai += float(cost)
                ti, to = obj.get("tokens_in"), obj.get("tokens_out")
                if isinstance(ti, (int, float)): tin  += int(ti)
                if isinstance(to, (int, float)): tout += int(to)
                if "tokens_in" in obj or "cost_usd" in obj:
                    calls += 1
    except Exception as exc:
        log.warning("runs.jsonl LLM scan failed: %s", exc)

    if seen_cost:
        out["anthropic_usd"] = round(anthro, 4)
        out["openai_usd"]    = round(openai, 4)
        # Linear projection to month-end based on day-of-month elapsed.
        now  = datetime.now(timezone.utc)
        days_in_month = (datetime(now.year + (now.month==12), (now.month%12)+1, 1, tzinfo=timezone.utc)
                         - timedelta(days=1)).day
        elapsed = now.day
        if elapsed > 0:
            out["projected_usd"] = round((anthro + openai) * days_in_month / elapsed, 2)
    out["tokens_in"]  = tin
    out["tokens_out"] = tout
    out["calls"]      = calls
    return out


@st.cache_data(ttl=TTL_LIVE, show_spinner=False)
def get_audit_trail(limit: int = 50) -> pd.DataFrame:
    """Unified event stream: pipeline_runs ∪ executed_orders ∪ monitor_ticks.

    Columns: ts (UTC), kind, summary
    """
    cols = ["ts","kind","summary"]
    parts: list[pd.DataFrame] = []

    runs = _sql("""
        SELECT started_at AS ts, status, run_id, error_msg
          FROM pipeline_runs
         ORDER BY started_at DESC LIMIT ?
    """, (limit,))
    if not runs.empty:
        runs["kind"]    = "pipeline"
        runs["summary"] = runs.apply(
            lambda r: f"run {str(r['run_id'])[:8]} · {r['status']}"
                      + (f" · {r['error_msg']}" if r.get("error_msg") else ""),
            axis=1,
        )
        parts.append(runs[["ts","kind","summary"]])

    orders = _sql("""
        SELECT submitted_at AS ts, ticker, action, quantity, fill_price, status
          FROM executed_orders
         ORDER BY submitted_at DESC LIMIT ?
    """, (limit,))
    if not orders.empty:
        orders["kind"]    = "order"
        orders["summary"] = orders.apply(
            lambda r: (f"{r['action']} {r['ticker']} qty={r['quantity']}"
                       + (f" @ {r['fill_price']}" if r.get("fill_price") else "")
                       + f" · {r['status']}"),
            axis=1,
        )
        parts.append(orders[["ts","kind","summary"]])

    ticks = _sql("""
        SELECT timestamp AS ts, ticker, decision, reason
          FROM monitor_ticks
         WHERE action_taken = 1
         ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    if not ticks.empty:
        ticks["kind"]    = "monitor"
        ticks["summary"] = ticks.apply(
            lambda r: f"{r['ticker']} · {r['decision']} · {r.get('reason') or ''}",
            axis=1,
        )
        parts.append(ticks[["ts","kind","summary"]])

    if not parts:
        return pd.DataFrame(columns=cols)

    out = pd.concat(parts, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)
    out = out.dropna(subset=["ts"]).sort_values("ts", ascending=False).head(limit)
    return out[cols].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Kill switch (Tab 6)
# ─────────────────────────────────────────────────────────────────────────────

def is_kill_switch_active() -> bool:
    """Live (no cache) — Tab 6 polls each render."""
    return KILL_SWITCH_PATH.exists()


def arm_kill_switch(reason: str = "armed via dashboard") -> bool:
    """Touch .athanor_kill at project root. Returns True if file now exists.
       Phase 8 will wire the consumers; Tab 6 button creates the file today."""
    try:
        KILL_SWITCH_PATH.write_text(
            json.dumps({"armed_at": datetime.now(timezone.utc).isoformat(),
                        "reason":   reason}),
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        log.error("kill switch arm failed: %s", exc)
        return False


def disarm_kill_switch() -> bool:
    try:
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()
        return True
    except Exception as exc:
        log.error("kill switch disarm failed: %s", exc)
        return False


@st.cache_data(ttl=60)
def get_circuit_breaker_status() -> list:
    """Return list of CBStatus from a no-adapter check (flag files + DB only)."""
    try:
        from src.risk.circuit_breakers import check_all
        return check_all(adapter=None)
    except Exception as exc:
        log.warning("get_circuit_breaker_status failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Cross-tab utilities
# ─────────────────────────────────────────────────────────────────────────────

def clear_all_caches() -> None:
    """Sidebar 'Refresh' hook — wipes every @st.cache_data entry in this module."""
    st.cache_data.clear()


@dataclass(frozen=True)
class HeaderInfo:
    """Top strip of the page (mode, last sync, kill state)."""
    mode:        str   = "paper"
    last_sync:   str   = ""
    kill_armed:  bool  = False


def get_header_info() -> HeaderInfo:
    return HeaderInfo(
        mode="paper",
        last_sync=datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        kill_armed=is_kill_switch_active(),
    )
