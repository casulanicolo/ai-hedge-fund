"""
src/alerts/premarket_builder.py
────────────────────────────────
Builds the pre-market briefing email (Section 1 of 2 daily emails).

Data sources:
  - yfinance: VIX history, yield curve (^TNX, ^IRX), SPY, earnings calendar
  - Alpaca: current open positions via BrokerAdapter
  - SQLite: last portfolio_decisions run, agent weights, outcomes

CLI:
    python -m src.alerts.premarket_builder --preview
    python -m src.alerts.premarket_builder --preview --out premarket.html
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = pathlib.Path(__file__).parent / "email_templates"


# ─────────────────────────────────────────────────────────────────────────────
# Internal data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_float(v: Optional[float], decimals: int = 2, prefix: str = "") -> str:
    if v is None:
        return "N/A"
    return f"{prefix}{v:,.{decimals}f}"


def _pl_color(v: float) -> str:
    return "#27ae60" if v >= 0 else "#c0392b"


def _fetch_vix() -> dict:
    try:
        import yfinance as yf
        hist = yf.Ticker("^VIX").history(period="20d", interval="1d")
        if hist.empty:
            return {}
        closes = hist["Close"]
        last   = float(closes.iloc[-1])
        prev10 = float(closes.iloc[-11]) if len(closes) >= 11 else float(closes.iloc[0])
        change = last - prev10
        trend  = "↑" if change > 0.5 else "↓" if change < -0.5 else "→"
        if last >= 30:
            color = "#c0392b"
        elif last >= 20:
            color = "#e8a83a"
        else:
            color = "#27ae60"
        return {
            "vix": f"{last:.1f}",
            "vix_date": closes.index[-1].strftime("%Y-%m-%d"),
            "vix_trend": trend,
            "vix_color": color,
            "vix_10d_change": f"{change:+.1f}",
            "vix_10d_color": "#c0392b" if change > 0 else "#27ae60",
        }
    except Exception as exc:
        logger.warning("[premarket] VIX fetch failed: %s", exc)
        return {"vix": "N/A", "vix_date": "—", "vix_trend": "→", "vix_color": "#888",
                "vix_10d_change": "N/A", "vix_10d_color": "#888"}


def _fetch_yield_curve() -> dict:
    try:
        import yfinance as yf
        t10  = yf.Ticker("^TNX").fast_info
        t3m  = yf.Ticker("^IRX").fast_info
        y10  = getattr(t10, "last_price", None)
        y3m  = getattr(t3m, "last_price", None)
        if y10 is None or y3m is None:
            raise ValueError("No yield data")
        spread = y10 - y3m
        inverted = spread < 0
        return {
            "yield_10y": f"{y10:.2f}",
            "yield_3m":  f"{y3m:.2f}",
            "yield_spread": f"{spread:+.2f}%",
            "yield_status": "INVERTED" if inverted else "NORMAL",
            "yield_color": "#c0392b" if inverted else "#27ae60",
        }
    except Exception as exc:
        logger.warning("[premarket] Yield curve fetch failed: %s", exc)
        return {"yield_10y": "N/A", "yield_3m": "N/A",
                "yield_spread": "N/A", "yield_status": "N/A", "yield_color": "#888"}


def _fetch_events(tickers: list[str]) -> list[str]:
    """Check earnings calendar for watchlist tickers."""
    events: list[str] = []
    try:
        import yfinance as yf
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for ticker in tickers[:8]:  # limit API calls
            try:
                cal = yf.Ticker(ticker).calendar
                if cal is None or cal.empty:
                    continue
                # calendar has 'Earnings Date' column
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"]
                    ed_str = str(ed.iloc[0])[:10] if hasattr(ed, "iloc") else str(ed)[:10]
                    if ed_str == today_str:
                        events.append(f"{ticker} Earnings")
            except Exception:
                pass
    except Exception as exc:
        logger.warning("[premarket] Events fetch failed: %s", exc)
    return events


def _fetch_regime_from_db() -> dict:
    """Read macro_regime from most recent pipeline_run JSONL log."""
    try:
        log_path = pathlib.Path("logs/runs.jsonl")
        if not log_path.exists():
            return {}
        import json
        last_run = None
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    last_run = json.loads(line)
                except Exception:
                    pass
        if not last_run:
            return {}
        regime = last_run.get("macro_regime") or last_run.get("regime", "UNKNOWN")
        return {"regime": regime}
    except Exception:
        return {}


def _fetch_open_positions(adapter=None) -> list[dict]:
    positions = []
    try:
        if adapter is None:
            from src.execution.alpaca_adapter import AlpacaBrokerAdapter
            adapter = AlpacaBrokerAdapter()
        raw = adapter.get_positions()
    except Exception as exc:
        logger.warning("[premarket] Alpaca positions failed: %s", exc)
        return []

    try:
        import yfinance as yf
        from src.db.init_db import get_connection
        conn = get_connection()
    except Exception:
        conn = None

    for p in raw:
        try:
            ticker = p.ticker
            entry  = float(p.avg_entry_price)
            qty    = float(p.qty)
            mv     = float(p.market_value)
            unreal = float(p.unrealized_pl)
            unreal_pct = float(p.unrealized_pl_pct) * 100 if hasattr(p, "unrealized_pl_pct") else 0.0

            # Previous close from yfinance
            prev_close = None
            try:
                hist = yf.Ticker(ticker).history(period="2d", interval="1d")
                if len(hist) >= 2:
                    prev_close = float(hist["Close"].iloc[-2])
                elif len(hist) == 1:
                    prev_close = float(hist["Close"].iloc[-1])
            except Exception:
                pass

            # Stop loss / take profit from DB
            sl = tp = None
            if conn:
                try:
                    row = conn.execute(
                        "SELECT stop_loss, take_profit, submitted_at FROM executed_orders "
                        "WHERE ticker=? AND action IN ('OPEN_LONG','OPEN_SHORT') "
                        "AND status NOT IN ('REJECTED','CANCELED') ORDER BY submitted_at DESC LIMIT 1",
                        (ticker,),
                    ).fetchone()
                    if row:
                        sl = row["stop_loss"]
                        tp = row["take_profit"]
                        opened_at_str = row["submitted_at"]
                        try:
                            opened_at = datetime.fromisoformat(opened_at_str.replace("Z", "+00:00"))
                            days_held = int((datetime.now(timezone.utc) - opened_at).days)
                        except Exception:
                            days_held = "?"
                    else:
                        days_held = "?"
                except Exception:
                    days_held = "?"
            else:
                days_held = "?"

            current = mv / qty if qty else entry
            dist_sl = f"{abs((current - sl) / current * 100):.1f}%" if sl else "—"
            dist_tp = f"{abs((tp - current) / current * 100):.1f}%" if tp else "—"

            positions.append({
                "ticker":    ticker,
                "side":      str(p.side),
                "qty":       int(abs(qty)),
                "entry":     f"{entry:.2f}",
                "close_prev": f"{prev_close:.2f}" if prev_close else "N/A",
                "pl_str":    f"${unreal:+,.2f} ({unreal_pct:+.2f}%)",
                "pl_color":  _pl_color(unreal),
                "days_held": days_held,
                "dist_sl":   dist_sl,
                "dist_tp":   dist_tp,
            })
        except Exception as exc:
            logger.warning("[premarket] Position enrich %s failed: %s", p.ticker, exc)

    if conn:
        try:
            conn.close()
        except Exception:
            pass
    return positions


def _fetch_trade_plan() -> list[dict]:
    """Read last run's portfolio_decisions from SQLite."""
    orders = []
    try:
        from src.db.init_db import get_connection
        conn = get_connection()
        rows = conn.execute(
            """
            SELECT pd.* FROM portfolio_decisions pd
            INNER JOIN (
                SELECT run_id FROM portfolio_decisions
                GROUP BY run_id ORDER BY MAX(timestamp) DESC LIMIT 1
            ) latest ON pd.run_id = latest.run_id
            ORDER BY pd.conviction DESC NULLS LAST
            """,
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("[premarket] Trade plan DB fetch failed: %s", exc)
        return []

    _ACTION_MAP = {
        "BUY":  ("OPEN_LONG",  "#d4f4e2", "#1a7a4a"),
        "SELL": ("OPEN_SHORT", "#fde0e0", "#b03030"),
        "HOLD": ("HOLD",       "#f9f9f9", "#888888"),
    }

    plan_counts = {"new_count": 0, "hold_count": 0, "close_count": 0}

    for row in rows:
        action_raw = row["action"]
        label, bg, fg = _ACTION_MAP.get(action_raw, (action_raw, "#f9f9f9", "#333"))

        if action_raw == "BUY":
            plan_counts["new_count"] += 1
        elif action_raw == "SELL":
            plan_counts["new_count"] += 1
        else:
            plan_counts["hold_count"] += 1

        entry = row["entry_price"] or row["info_entry"]
        sl    = row["stop_loss"] or row["info_sl"]
        tp    = row["take_profit"] or row["info_tp"]
        size  = row["size_usd"] or row["info_size_usd"]

        reasoning = (row["reasoning"] or "")[:120]

        orders.append({
            "ticker":    row["ticker"],
            "action":    label,
            "action_bg": bg,
            "action_fg": fg,
            "qty":       "—",
            "notional":  f"${size:,.0f}" if size else "—",
            "entry":     f"${entry:.2f}" if entry else "—",
            "sl":        f"${sl:.2f}"    if sl    else "—",
            "tp":        f"${tp:.2f}"    if tp    else "—",
            "conviction": f"{(row['conviction'] or 0)*100:.0f}%",
            "reasoning": reasoning,
        })

    return orders, plan_counts


def _fetch_kpis() -> dict:
    """Compute KPIs from SQLite data."""
    kpis = {
        "week_pl_str":   "N/A",
        "week_pl_color": "#888",
        "sharpe":        "N/A",
        "max_dd":        "N/A",
        "win_rate":      "N/A",
    }
    try:
        from src.db.init_db import get_connection
        conn = get_connection()

        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        # Win rate: last 20 CLOSE/SCALE_OUT fills with known fill_price
        rows = conn.execute(
            """
            SELECT eo.ticker, eo.action, eo.fill_price,
                   eo2.fill_price AS open_fill
            FROM executed_orders eo
            LEFT JOIN executed_orders eo2
                ON eo2.ticker = eo.ticker
               AND eo2.action IN ('OPEN_LONG','OPEN_SHORT')
               AND eo2.submitted_at < eo.submitted_at
            WHERE eo.action IN ('CLOSE','SCALE_OUT')
              AND eo.fill_price IS NOT NULL
              AND eo.status = 'FILLED'
            ORDER BY eo.submitted_at DESC LIMIT 20
            """
        ).fetchall()
        conn.close()

        if rows:
            wins = 0
            for r in rows:
                if r["fill_price"] and r["open_fill"]:
                    if r["action"] in ("CLOSE", "SCALE_OUT"):
                        pnl = r["fill_price"] - r["open_fill"]
                        if pnl > 0:
                            wins += 1
            kpis["win_rate"] = f"{wins}/{len(rows)} ({wins/len(rows)*100:.0f}%)"

    except Exception as exc:
        logger.warning("[premarket] KPI fetch failed: %s", exc)

    return kpis


def _fetch_devils_advocate() -> str:
    """Read devil's advocate output from last JSONL run."""
    try:
        import json
        log_path = pathlib.Path("logs/runs.jsonl")
        if not log_path.exists():
            return ""
        last = None
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    last = json.loads(line)
                except Exception:
                    pass
        if not last:
            return ""
        da = last.get("devil_advocate") or last.get("devils_advocate", "")
        return str(da)[:600] if da else ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_context(tickers: Optional[list[str]] = None, adapter=None) -> dict:
    """Assemble the full data context for the pre-market template."""
    tickers = tickers or []

    vix_data   = _fetch_vix()
    yield_data = _fetch_yield_curve()
    events     = _fetch_events(tickers)
    regime_data = _fetch_regime_from_db()

    regime = regime_data.get("regime", "UNKNOWN")
    regime_colors = {"RISK_ON": "#27ae60", "CAUTION": "#e8a83a", "RISK_OFF": "#c0392b"}
    regime_notes  = {
        "RISK_ON":  "VIX < 20, normal yield curve",
        "CAUTION":  "VIX 20–30 or inverted yield",
        "RISK_OFF": "VIX ≥ 30 or severe inversion",
    }

    positions       = _fetch_open_positions(adapter=adapter)
    trade_plan_raw  = _fetch_trade_plan()
    if isinstance(trade_plan_raw, tuple):
        trade_plan, plan_counts = trade_plan_raw
    else:
        trade_plan, plan_counts = trade_plan_raw, {"new_count": 0, "hold_count": 0, "close_count": 0}

    kpis          = _fetch_kpis()
    devils        = _fetch_devils_advocate()
    now_utc       = datetime.now(timezone.utc)

    return {
        "date":         now_utc.strftime("%Y-%m-%d"),
        "generated_at": now_utc.strftime("%H:%M"),
        "macro": {
            **vix_data,
            **yield_data,
            "regime":       regime,
            "regime_color": regime_colors.get(regime, "#888"),
            "regime_note":  regime_notes.get(regime, ""),
            "events":       events,
        },
        "plan":          plan_counts,
        "open_positions": positions,
        "trade_plan":     trade_plan,
        "devils_advocate": devils,
        "kpis":           kpis,
    }


def build_html(ctx: dict) -> str:
    """Render the Jinja2 premarket template with the given context."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("premarket.html.j2")
    return tpl.render(**ctx)


def build_text(ctx: dict) -> str:
    """Plain-text fallback."""
    lines = [
        f"ATHANOR ALPHA — Pre-Market Briefing — {ctx['date']}",
        "=" * 60,
        "",
        f"Regime: {ctx['macro']['regime']}",
        f"VIX: {ctx['macro']['vix']} {ctx['macro']['vix_trend']}",
        f"Yield curve: {ctx['macro']['yield_spread']} ({ctx['macro']['yield_status']})",
        "",
        f"Plan: {ctx['plan']['new_count']} NEW | {ctx['plan']['hold_count']} HOLD | {ctx['plan']['close_count']} CLOSE",
        "",
    ]
    if ctx["open_positions"]:
        lines.append("OPEN POSITIONS")
        for p in ctx["open_positions"]:
            lines.append(f"  {p['ticker']:6} {p['side']:5} qty={p['qty']} entry={p['entry']} P&L={p['pl_str']}")
    lines.append("")
    if ctx["trade_plan"]:
        lines.append("TRADE PLAN")
        for o in ctx["trade_plan"]:
            lines.append(f"  {o['ticker']:6} {o['action']:12} entry={o['entry']} SL={o['sl']} TP={o['tp']} conv={o['conviction']}")
    lines.append("")
    lines += [
        f"Win rate (20 trades): {ctx['kpis']['win_rate']}",
        "─" * 60,
        "AI-generated. Not financial advice.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build pre-market email")
    parser.add_argument("--preview", action="store_true", help="Print HTML to stdout")
    parser.add_argument("--out", default="", help="Write HTML to file")
    parser.add_argument("--tickers", nargs="*", default=[], help="Watchlist tickers for event check")
    args = parser.parse_args()

    ctx  = build_context(tickers=args.tickers)
    html = build_html(ctx)
    text = build_text(ctx)

    if args.out:
        pathlib.Path(args.out).write_text(html, encoding="utf-8")
        print(f"HTML written to {args.out}")
    elif args.preview:
        sys.stdout.write(html)
    else:
        print(text)


if __name__ == "__main__":
    _main()
