"""
src/monitor/daemon.py

Athanor Alpha — Monitor Daemon
Runs every 15 minutes during US market hours (14:30–21:00 UTC, Mon–Fri).

Each cycle:
  1. Load positions from config/positions.json
  2. Run price + news checks via alert_builder
  3. Send immediate email for HIGH alerts (rate limited: 1/ticker/hour)
  4. Send digest email if any MEDIUM alerts fired (or if --digest flag used)
  5. Print all results to stdout (logged via shell redirect)

Usage:
    python -m src.monitor.daemon           # normal mode (waits for market hours)
    python -m src.monitor.daemon --now     # force one cycle immediately (for testing)
    python -m src.monitor.daemon --digest  # force one cycle + always send digest email
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from datetime import datetime, timezone

from src.monitor.alert_builder import run_all_positions
from src.alerts.email_sender import send_alert, send_digest
from src.alerts.templates import build_alert_html, build_digest_html

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("athanor.monitor")

# ── Constants ────────────────────────────────────────────────────────────────
POSITIONS_PATH   = pathlib.Path("config/positions.json")
CYCLE_MINUTES    = 15
MARKET_OPEN_UTC  = (14, 30)   # 14:30 UTC = 09:30 EST
MARKET_CLOSE_UTC = (21,  0)   # 21:00 UTC = 16:00 EST
MARKET_DAYS      = {0, 1, 2, 3, 4}   # Mon=0 … Fri=4


# ── Market hours check ───────────────────────────────────────────────────────
def is_market_hours(now: datetime | None = None) -> bool:
    """Return True if `now` (UTC) falls within US equity market hours."""
    now = now or datetime.now(timezone.utc)
    if now.weekday() not in MARKET_DAYS:
        return False
    open_minutes    = MARKET_OPEN_UTC[0]  * 60 + MARKET_OPEN_UTC[1]
    close_minutes   = MARKET_CLOSE_UTC[0] * 60 + MARKET_CLOSE_UTC[1]
    current_minutes = now.hour * 60 + now.minute
    return open_minutes <= current_minutes < close_minutes


# ── Positions loader ─────────────────────────────────────────────────────────
def load_positions() -> list[dict]:
    """Load and validate positions from config/positions.json."""
    if not POSITIONS_PATH.exists():
        log.error(f"positions.json not found at {POSITIONS_PATH}")
        return []
    try:
        data = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
        positions = data.get("positions", [])
        valid = []
        for p in positions:
            if "ticker" not in p or "entry_price" not in p:
                log.warning(f"Skipping invalid position entry: {p}")
                continue
            valid.append(p)
        return valid
    except Exception as exc:
        log.error(f"Failed to load positions.json: {exc}")
        return []


# ── Single monitoring cycle ──────────────────────────────────────────────────
def run_cycle(force_digest: bool = False) -> None:
    """
    Execute one full monitoring cycle across all positions.
    force_digest=True → always send digest email regardless of alert count.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info(f"── Cycle start {now_str} ──────────────────────────────────")

    positions = load_positions()
    if not positions:
        log.warning("No valid positions found. Skipping cycle.")
        return

    log.info(f"Checking {len(positions)} position(s): "
             f"{', '.join(p['ticker'] for p in positions)}")

    alerts = run_all_positions(positions)

    # ── Counts ───────────────────────────────────────────────────────────
    high_alerts   = [a for a in alerts if a["severity"] == "HIGH"]
    medium_alerts = [a for a in alerts if a["severity"] == "MEDIUM"]
    low_alerts    = [a for a in alerts if a["severity"] == "LOW"]

    log.info(f"Results → HIGH: {len(high_alerts)} | "
             f"MEDIUM: {len(medium_alerts)} | "
             f"LOW: {len(low_alerts)}")

    # ── Process each position ─────────────────────────────────────────────
    for a in alerts:
        if a.get("error"):
            log.error(f"[{a['ticker']}] Error during check: {a['error']}")
            continue

        log.info(a["summary_line"])

        # Print full text block to stdout/log
        if a["has_alert"]:
            print(a["message"])
            print()

        # ── HIGH alert → immediate email ──────────────────────────────
        if a["severity"] == "HIGH" and a["has_alert"]:
            ticker    = a["ticker"]
            subject   = f"[HIGH ALERT] Athanor Alpha — {ticker}"
            body_text = a["message"]
            body_html = build_alert_html(a)

            sent = send_alert(
                ticker    = ticker,
                subject   = subject,
                body_text = body_text,
                body_html = body_html,
            )
            if sent:
                log.info(f"[{ticker}] HIGH alert email sent.")
            else:
                log.info(f"[{ticker}] HIGH alert email skipped (rate limited or error).")

    # ── MEDIUM or forced → digest email ──────────────────────────────────
    if medium_alerts or force_digest:
        reason = "forced --digest flag" if force_digest else f"{len(medium_alerts)} MEDIUM alert(s)"
        log.info(f"Sending digest email ({reason})...")

        now_str2  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subject   = f"[DIGEST] Athanor Alpha — {now_str2}"

        lines = [f"Athanor Alpha — Daily Digest — {now_str2}", ""]
        for a in alerts:
            lines.append(a.get("summary_line", a["ticker"]))
        body_text = "\n".join(lines)
        body_html = build_digest_html(alerts)

        sent = send_digest(subject=subject, body_text=body_text, body_html=body_html)
        if sent:
            log.info("Digest email sent.")
        else:
            log.warning("Digest email failed — check SMTP config.")

    if not any(a["has_alert"] for a in alerts) and not force_digest:
        log.info("✓ All clear — no alerts this cycle.")

    log.info("── Cycle complete ───────────────────────────────────────────")


# ── Main daemon loop ─────────────────────────────────────────────────────────
def main(force_now: bool = False, force_digest: bool = False) -> None:
    log.info("Athanor Alpha Monitor Daemon starting...")
    log.info(f"Positions file : {POSITIONS_PATH.resolve()}")
    log.info(f"Cycle interval : {CYCLE_MINUTES} minutes")
    log.info(f"Market hours   : {MARKET_OPEN_UTC[0]:02d}:{MARKET_OPEN_UTC[1]:02d}–"
             f"{MARKET_CLOSE_UTC[0]:02d}:{MARKET_CLOSE_UTC[1]:02d} UTC (Mon–Fri)")

    if force_now or force_digest:
        flag = "--digest" if force_digest else "--now"
        log.info(f"{flag} flag detected: running one cycle immediately.")
        run_cycle(force_digest=force_digest)
        log.info("Done. Exiting.")
        return

    log.info("Entering market-hours loop. Press Ctrl+C to stop.")
    try:
        while True:
            now = datetime.now(timezone.utc)
            if is_market_hours(now):
                run_cycle()
                log.info(f"Sleeping {CYCLE_MINUTES} minutes until next cycle...")
                time.sleep(CYCLE_MINUTES * 60)
            else:
                day_name = now.strftime("%A")
                log.info(
                    f"Outside market hours ({day_name} "
                    f"{now.strftime('%H:%M')} UTC). "
                    f"Waiting 5 min..."
                )
                time.sleep(5 * 60)
    except KeyboardInterrupt:
        log.info("Daemon stopped by user (Ctrl+C).")


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Athanor Alpha Monitor Daemon")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run one cycle immediately (ignore market hours) then exit.",
    )
    parser.add_argument(
        "--digest",
        action="store_true",
        help="Run one cycle immediately and always send digest email, then exit.",
    )
    args = parser.parse_args()
    main(force_now=args.now, force_digest=args.digest)
