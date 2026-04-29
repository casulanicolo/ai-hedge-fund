"""
src/alerts/postmarket_builder.py
──────────────────────────────────
Builds the post-market EOD digest email (Section 2 of 2 daily emails).

Data sources:
  - SQLite: today's executed_orders, monitor_ticks, agent_weights
  - Alpaca: current overnight positions
  - yfinance: SPY daily return for alpha comparison

CLI:
    python -m src.alerts.postmarket_builder --preview
    python -m src.alerts.postmarket_builder --preview --out postmarket.html
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = pathlib.Path(__file__).parent / "email_templates"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pl_color(v: float) -> str:
    return "#27ae60" if v >= 0 else "#c0392b"


def _action_style(action: str) -> tuple[str, str]:
    styles = {
        "OPEN_LONG":  ("#d4f4e2", "#1a7a4a"),
        "OPEN_SHORT": ("#fde0e0", "#b03030"),
        "CLOSE":      ("#e8f4fd", "#1a5276"),
        "SCALE_OUT":  ("#fef9ec", "#7a5c10"),
        "SCALE_IN":   ("#f0e8fd", "#5b2a8a"),
        "ADJUST_SL":  ("#f5f5f5", "#555555"),
    }
    return styles.get(action, ("#f9f9f9", "#333333"))


def _today_range() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.isoformat(), now.isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_executed_orders() -> list[dict]:
    orders = []
    try:
        from src.db.init_db import get_connection
        conn = get_connection()
        today_start, _ = _today_range()
        rows = conn.execute(
            """
            SELECT * FROM executed_orders
            WHERE submitted_at >= ?
            ORDER BY submitted_at ASC
            """,
            (today_start,),
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("[postmarket] executed_orders fetch failed: %s", exc)
        return []

    for r in rows:
        bg, fg = _action_style(r["action"])
        # Infer fill source
        if "monitor" in (r["run_id"] or ""):
            source = "Monitor intraday"
        elif r["action"] in ("OPEN_LONG", "OPEN_SHORT"):
            source = "Pre-market plan"
        elif r["action"] == "CLOSE" and r["status"] == "FILLED":
            source = "Exit rule"
        else:
            source = "Pipeline"

        submitted = r["submitted_at"] or ""
        time_str  = submitted[11:16] if len(submitted) >= 16 else "—"
        fill_str  = f"${r['fill_price']:.2f}" if r["fill_price"] else "—"

        orders.append({
            "time":             time_str,
            "ticker":           r["ticker"],
            "action":           r["action"],
            "action_bg":        bg,
            "action_fg":        fg,
            "qty":              r["quantity"] or "—",
            "fill_price":       fill_str,
            "source":           source,
            "status":           r["status"] or "?",
            "rejection_reason": r["rejection_reason"],
        })
    return orders


def _fetch_monitor_summary() -> dict:
    try:
        from src.db.init_db import get_connection
        conn = get_connection()
        today_start, _ = _today_range()
        rows = conn.execute(
            "SELECT decision, rule, action_taken FROM monitor_ticks WHERE timestamp >= ?",
            (today_start,),
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("[postmarket] monitor_ticks fetch failed: %s", exc)
        return {"total_ticks": 0, "actions_taken": 0, "rules_summary": []}

    total  = len(rows)
    taken  = sum(1 for r in rows if r["action_taken"])
    counts: Counter = Counter(r["rule"] or r["decision"] for r in rows)
    summary = sorted(counts.items(), key=lambda x: -x[1])

    return {
        "total_ticks":   total,
        "actions_taken": taken,
        "rules_summary": summary,
    }


def _fetch_overnight_positions(adapter=None) -> list[dict]:
    positions = []
    try:
        if adapter is None:
            from src.execution.alpaca_adapter import AlpacaBrokerAdapter
            adapter = AlpacaBrokerAdapter()
        raw = adapter.get_positions()
    except Exception as exc:
        logger.warning("[postmarket] Alpaca positions failed: %s", exc)
        return []

    for p in raw:
        try:
            unreal = float(p.unrealized_pl)
            qty    = float(p.qty)
            positions.append({
                "ticker":  p.ticker,
                "side":    str(p.side),
                "qty":     int(abs(qty)),
                "pl_str":  f"${unreal:+,.2f}",
                "pl_color": _pl_color(unreal),
                "rationale": "No exit signal — held per rules",
            })
        except Exception as exc:
            logger.warning("[postmarket] Position format failed: %s", exc)

    return positions


def _fetch_accuracy() -> dict:
    try:
        from src.db.init_db import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT agent_id, ticker, weight FROM agent_weights ORDER BY weight DESC"
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("[postmarket] agent_weights fetch failed: %s", exc)
        return {"top": [], "bottom": []}

    if not rows:
        return {"top": [], "bottom": []}

    # Aggregate average weight per agent
    from collections import defaultdict
    agent_weights: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        agent_weights[r["agent_id"]].append(float(r["weight"]))

    avg_weights = {a: sum(ws) / len(ws) for a, ws in agent_weights.items()}
    sorted_agents = sorted(avg_weights.items(), key=lambda x: -x[1])

    top    = [(a, f"{w:.3f}") for a, w in sorted_agents[:3]]
    bottom = [(a, f"{w:.3f}") for a, w in sorted_agents[-3:] if w < 1.0]

    return {"top": top, "bottom": bottom}


def _fetch_spy_return() -> Optional[float]:
    try:
        import yfinance as yf
        hist = yf.Ticker("SPY").history(period="2d", interval="1d")
        if len(hist) >= 2:
            return float((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100)
    except Exception:
        pass
    return None


def _fetch_pnl_summary(adapter=None) -> dict:
    """Compute today's P&L summary from Alpaca + SQLite."""
    defaults = {
        "pl_str": "N/A", "pl_color": "#888",
        "realized_str": "N/A", "realized_color": "#888",
        "unrealized_str": "N/A", "unrealized_color": "#888",
        "equity_eod": "N/A",
        "max_dd_str": "N/A",
        "peak_exposure": "N/A",
        "alpha_str": "N/A", "alpha_color": "#888",
        "fills": 0, "exits": 0, "rejects": 0,
    }
    realized = 0.0
    try:
        from src.db.init_db import get_connection
        conn = get_connection()
        today_start, _ = _today_range()
        rows = conn.execute(
            "SELECT action, status, fill_price, quantity, notional_usd, broker_order_id"
            " FROM executed_orders WHERE submitted_at >= ?",
            (today_start,),
        ).fetchall()

        fills   = sum(1 for r in rows if r["status"] == "FILLED" and r["action"] in ("OPEN_LONG", "OPEN_SHORT"))
        exits   = sum(1 for r in rows if r["status"] == "FILLED" and r["action"] in ("CLOSE", "SCALE_OUT"))
        rejects = sum(1 for r in rows if r["status"] == "REJECTED")

        # Fallback: if reconcile hasn't run yet, count submitted (has broker_order_id)
        if fills == 0:
            fills = sum(1 for r in rows if r["broker_order_id"] and r["action"] in ("OPEN_LONG", "OPEN_SHORT"))
        if exits == 0:
            exits = sum(1 for r in rows if r["broker_order_id"] and r["action"] in ("CLOSE", "SCALE_OUT"))

        defaults["fills"]   = fills
        defaults["exits"]   = exits
        defaults["rejects"] = rejects

        # Realized P&L: for each filled CLOSE/SCALE_OUT today, match to most recent OPEN
        close_rows = conn.execute(
            """
            SELECT ticker, action, fill_price, quantity
            FROM executed_orders
            WHERE submitted_at >= ? AND status = 'FILLED'
              AND action IN ('CLOSE', 'SCALE_OUT')
              AND fill_price IS NOT NULL AND quantity IS NOT NULL
            """,
            (today_start,),
        ).fetchall()
        for row in close_rows:
            open_row = conn.execute(
                """
                SELECT fill_price, action FROM executed_orders
                WHERE ticker = ? AND action IN ('OPEN_LONG', 'OPEN_SHORT')
                  AND status = 'FILLED' AND fill_price IS NOT NULL
                ORDER BY submitted_at DESC LIMIT 1
                """,
                (row["ticker"],),
            ).fetchone()
            if open_row:
                close_price = float(row["fill_price"])
                open_price  = float(open_row["fill_price"])
                qty         = float(row["quantity"])
                if open_row["action"] == "OPEN_LONG":
                    realized += (close_price - open_price) * qty
                else:
                    realized += (open_price - close_price) * qty
        conn.close()

    except Exception as exc:
        logger.warning("[postmarket] P&L summary DB failed: %s", exc)

    # Unrealized from Alpaca
    try:
        if adapter is None:
            from src.execution.alpaca_adapter import AlpacaBrokerAdapter
            adapter = AlpacaBrokerAdapter()
        acct    = adapter.get_account()
        equity  = getattr(acct, "equity", None) or acct.portfolio_value
        raw_pos = adapter.get_positions()
        unreal  = sum(float(p.unrealized_pl) for p in raw_pos)
        spy_ret = _fetch_spy_return()

        unreal_pct = unreal / equity * 100 if equity else 0.0
        total_pl   = realized + unreal

        defaults["unrealized_str"]   = f"${unreal:+,.2f} ({unreal_pct:+.2f}%)"
        defaults["unrealized_color"] = _pl_color(unreal)
        defaults["realized_str"]     = f"${realized:+,.2f}"
        defaults["realized_color"]   = _pl_color(realized)
        defaults["pl_str"]           = f"${total_pl:+,.2f}"
        defaults["pl_color"]         = _pl_color(total_pl)
        defaults["equity_eod"]       = f"${equity:,.2f}"

        if spy_ret is not None:
            port_ret_pct = total_pl / equity * 100 if equity else 0.0
            alpha        = port_ret_pct - spy_ret
            defaults["alpha_str"]   = f"{alpha:+.2f}% vs SPY ({spy_ret:+.2f}%)"
            defaults["alpha_color"] = _pl_color(alpha)

        defaults["max_dd_str"]      = "N/A (no intraday equity stream)"
        defaults["peak_exposure"]   = f"{sum(abs(float(p.market_value)) for p in raw_pos) / equity * 100:.1f}% of equity" if equity else "N/A"

    except Exception as exc:
        logger.warning("[postmarket] Alpaca P&L failed: %s", exc)

    return defaults


def _fetch_anomalies() -> list[str]:
    """Scan today's monitor log for ERROR-level lines."""
    anomalies = []
    try:
        log_path = pathlib.Path("logs/monitor.log")
        if not log_path.exists():
            return []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if today in line and " [ERROR   ]" in line:
                    anomalies.append(line.strip()[:120])
    except Exception:
        pass
    return anomalies[:10]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_context(adapter=None) -> dict:
    """Assemble the full data context for the post-market template."""
    # Sync fill status before reading DB — ensures FILLED/fill_price are current
    try:
        from src.execution.executor import reconcile_orders
        from src.execution.alpaca_adapter import AlpacaBrokerAdapter
        _reconcile_adapter = adapter or AlpacaBrokerAdapter()
        reconcile_orders(_reconcile_adapter)
    except Exception as exc:
        logger.warning("[postmarket] reconcile_orders failed: %s", exc)

    summary  = _fetch_pnl_summary(adapter=adapter)
    orders   = _fetch_executed_orders()
    monitor  = _fetch_monitor_summary()
    overnight = _fetch_overnight_positions(adapter=adapter)
    accuracy = _fetch_accuracy()
    anomalies = _fetch_anomalies()
    now_utc   = datetime.now(timezone.utc)

    return {
        "date":            now_utc.strftime("%Y-%m-%d"),
        "generated_at":   now_utc.strftime("%H:%M"),
        "summary":         summary,
        "executed_orders": orders,
        "monitor":         monitor,
        "overnight":       overnight,
        "accuracy":        accuracy,
        "anomalies":       anomalies,
    }


def build_html(ctx: dict) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("postmarket.html.j2")
    return tpl.render(**ctx)


def build_text(ctx: dict) -> str:
    s = ctx["summary"]
    lines = [
        f"ATHANOR ALPHA — EOD Digest — {ctx['date']}",
        "=" * 60,
        f"P&L today: {s['pl_str']}  |  vs SPY: {s['alpha_str']}",
        f"Fills: {s['fills']}  Exits: {s['exits']}  Rejects: {s['rejects']}",
        f"Equity EOD: {s['equity_eod']}",
        "",
    ]
    if ctx["executed_orders"]:
        lines.append("EXECUTED ORDERS")
        for o in ctx["executed_orders"]:
            lines.append(f"  {o['time']} {o['ticker']:6} {o['action']:12} qty={o['qty']} fill={o['fill_price']} {o['status']}")
    lines.append("")
    m = ctx["monitor"]
    lines.append(f"Monitor: {m['total_ticks']} ticks, {m['actions_taken']} actions taken")
    if ctx["overnight"]:
        lines.append("\nOVERNIGHT POSITIONS")
        for p in ctx["overnight"]:
            lines.append(f"  {p['ticker']:6} {p['side']:5} qty={p['qty']} unreal={p['pl_str']}")
    lines += ["", "─" * 60, "AI-generated. Not financial advice."]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build post-market EOD email")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Print HTML to stdout")
    mode.add_argument("--send",    action="store_true", help="Send the EOD digest via SMTP (real email)")
    parser.add_argument("--out", default="", help="Write HTML to file")
    args = parser.parse_args()

    ctx  = build_context()
    html = build_html(ctx)
    text = build_text(ctx)

    if args.out:
        pathlib.Path(args.out).write_text(html, encoding="utf-8")
        print(f"HTML written to {args.out}")

    if args.send:
        from src.alerts.email_sender import send_postmarket
        summary = ctx.get("summary", {}) or {}
        summary_line = ""
        pl = summary.get("pl_str")
        alpha = summary.get("alpha_str")
        if pl or alpha:
            summary_line = " | ".join(p for p in (pl, f"vs SPY {alpha}" if alpha else None) if p)
        ok = send_postmarket(html, text, date=ctx.get("date", ""), summary_line=summary_line)
        if ok:
            print(f"Post-market email sent ({ctx.get('date', '')}).")
            sys.exit(0)
        print("Post-market email send FAILED — see logs.", file=sys.stderr)
        sys.exit(1)

    if args.preview:
        sys.stdout.write(html)
    elif not args.out:
        print(text)


if __name__ == "__main__":
    _main()
