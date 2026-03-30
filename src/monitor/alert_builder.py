"""
src/monitor/alert_builder.py

Aggregates results from price_checker and news_checker for all positions,
decides which alerts fire, and builds a formatted alert message.

In Phase 5.1 alerts are printed to stdout / written to log.
In Phase 5.2 the message will be passed to email_sender.py.
"""

from datetime import datetime, timezone
from typing import Optional

from src.monitor.price_checker import check_price
from src.monitor.news_checker import check_news


# ── Alert severity levels ────────────────────────────────────────────────────
SEVERITY_HIGH   = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW    = "LOW"


def _overall_severity(price_alerts: list, news_result: dict) -> str:
    """Return the highest severity across price and news signals."""
    if price_alerts or news_result["top_urgency"] == "high":
        return SEVERITY_HIGH
    if news_result["top_urgency"] == "medium":
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def build_alerts_for_position(
    ticker: str,
    entry_price: float,
    shares: float,
    entry_date: str,
    notes: str = "",
) -> dict:
    """
    Run price + news checks for a single position and build a structured alert.

    Returns:
        ticker          str
        severity        str    HIGH / MEDIUM / LOW
        has_alert       bool   True if anything warrants notification
        price_result    dict   raw output from check_price
        news_result     dict   raw output from check_news
        all_alerts      list   combined alert strings
        summary_line    str    one-liner for log/digest
        message         str    full formatted message block
        error           str | None
    """
    result = {
        "ticker": ticker,
        "severity": SEVERITY_LOW,
        "has_alert": False,
        "price_result": {},
        "news_result": {},
        "all_alerts": [],
        "summary_line": "",
        "message": "",
        "error": None,
    }

    try:
        price_res = check_price(ticker, entry_price)
        news_res  = check_news(ticker)

        result["price_result"] = price_res
        result["news_result"]  = news_res

        if price_res.get("error"):
            result["error"] = f"price_checker: {price_res['error']}"
            return result

        all_alerts = price_res["alerts"] + news_res["alerts"]
        result["all_alerts"] = all_alerts

        severity = _overall_severity(price_res["alerts"], news_res)
        result["severity"]  = severity
        result["has_alert"] = bool(all_alerts) or news_res["top_urgency"] in ("high", "medium")

        # ── Summary line ─────────────────────────────────────────────────
        cp = price_res.get("current_price")
        pc = price_res.get("price_change")
        price_str = f"${cp:.2f} ({pc:+.2%})" if cp and pc is not None else "N/A"
        result["summary_line"] = (
            f"[{severity}] {ticker} | price {price_str} | "
            f"vol ratio {price_res.get('volume_ratio', 0):.2f}x | "
            f"news urgency {news_res['top_urgency'].upper()} | "
            f"{len(all_alerts)} alert(s)"
        )

        # ── Full message block ────────────────────────────────────────────
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"{'='*60}",
            f"  ATHANOR ALPHA — MONITOR ALERT",
            f"  {now_str}",
            f"{'='*60}",
            f"  Ticker     : {ticker}",
            f"  Severity   : {severity}",
            f"  Entry      : ${entry_price:.2f} on {entry_date}",
            f"  Shares     : {shares}",
            f"  Notes      : {notes}" if notes else "",
            f"",
            f"── Price ──────────────────────────────────────────────",
        ]
        if cp is not None:
            lines.append(f"  Current price  : ${cp:.2f}")
        if pc is not None:
            lines.append(f"  Change vs entry: {pc:+.2%}")
        if price_res.get("volume_ratio") is not None:
            lines.append(f"  Volume ratio   : {price_res['volume_ratio']:.2f}x 20d avg")
        if price_res.get("gap") is not None:
            lines.append(f"  Gap open/close : {price_res['gap']:+.2%}")

        lines.append("")
        lines.append("── News ───────────────────────────────────────────────")
        articles = news_res.get("articles", [])
        if articles:
            for art in articles[:5]:
                lines.append(f"  [{art['urgency'].upper():6}] {art['published']} — {art['title'][:72]}")
        else:
            lines.append("  No recent news in lookback window.")

        if all_alerts:
            lines.append("")
            lines.append("── Triggered Alerts ───────────────────────────────────")
            for a in all_alerts:
                lines.append(f"  ⚠ {a}")

        lines.append(f"{'='*60}")
        result["message"] = "\n".join(l for l in lines if l is not None)

    except Exception as exc:
        result["error"] = str(exc)

    return result


def run_all_positions(positions: list[dict]) -> list[dict]:
    """
    Run build_alerts_for_position for every entry in positions list.
    Returns list of alert dicts, sorted by severity (HIGH first).
    """
    results = []
    for pos in positions:
        alert = build_alerts_for_position(
            ticker      = pos["ticker"],
            entry_price = pos["entry_price"],
            shares      = pos.get("shares", 0),
            entry_date  = pos.get("entry_date", ""),
            notes       = pos.get("notes", ""),
        )
        results.append(alert)

    severity_order = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 1, SEVERITY_LOW: 2}
    results.sort(key=lambda r: severity_order.get(r["severity"], 99))
    return results


# ── Quick standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, pathlib

    positions_path = pathlib.Path("config/positions.json")
    positions = json.loads(positions_path.read_text())["positions"]

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Alert builder test")
    print(f"Checking {len(positions)} position(s)...\n")

    alerts = run_all_positions(positions)

    # Print summary table
    print("── Summary ────────────────────────────────────────────")
    for a in alerts:
        print(f"  {a['summary_line']}")
    print()

    # Print full message only for positions with alerts
    for a in alerts:
        if a["has_alert"]:
            print(a["message"])
            print()

    no_alert = [a for a in alerts if not a["has_alert"]]
    if no_alert:
        print(f"── No alerts for: {', '.join(a['ticker'] for a in no_alert)} ──")
