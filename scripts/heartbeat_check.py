#!/usr/bin/env python3
"""
scripts/heartbeat_check.py
Verifica che il monitor daemon stia girando. Se l'ultimo tick [daemon]
è > 90 minuti fa durante market hours, invia mail di alert.

Idempotente: manda al massimo 1 mail per giorno (file .heartbeat_alerted).

Crontab:
  */30 14-21 * * 1-5 cd $PROJ && .venv/bin/python scripts/heartbeat_check.py >> logs/heartbeat.log 2>&1
"""

import argparse
import os
import re
import smtplib
import subprocess
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_FILE       = Path("logs/monitor.log")
ALERTED_FILE   = Path(".heartbeat_alerted")
SILENCE_THRESH = 90   # minuti
MARKET_OPEN    = (14, 30)   # UTC
MARKET_CLOSE   = (21,  0)   # UTC
MARKET_DAYS    = {0, 1, 2, 3, 4}   # Mon-Fri

SMTP_HOST       = os.getenv("SMTP_HOST", "")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER       = os.getenv("SMTP_USER", "")
SMTP_PASSWORD   = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT", "")

_DAEMON_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _is_market_hours(now: datetime) -> bool:
    if now.weekday() not in MARKET_DAYS:
        return False
    m       = now.hour * 60 + now.minute
    open_m  = MARKET_OPEN[0]  * 60 + MARKET_OPEN[1]
    close_m = MARKET_CLOSE[0] * 60 + MARKET_CLOSE[1]
    return open_m <= m < close_m


def _last_daemon_ts() -> "datetime | None":
    """Legge l'ultimo timestamp da una riga con '[daemon]' in monitor.log."""
    if not LOG_FILE.exists():
        return None
    last_ts = None
    try:
        with LOG_FILE.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "[daemon]" not in line:
                    continue
                m = _DAEMON_RE.match(line)
                if m:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    last_ts = ts.replace(tzinfo=timezone.utc)
    except Exception as exc:
        print(f"ERROR reading {LOG_FILE}: {exc}", file=sys.stderr)
    return last_ts


def _already_alerted_today() -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    if ALERTED_FILE.exists():
        return ALERTED_FILE.read_text().strip() == today
    return False


def _mark_alerted() -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    ALERTED_FILE.write_text(today)


def _send_alert(last_ts: "datetime | None", silence_min: int, dry_run: bool) -> None:
    last_str = (
        last_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        if last_ts else "MAI (nessuna riga [daemon] nel log)"
    )
    body = (
        f"Il monitor daemon di Athanor Alpha sembra offline.\n\n"
        f"Ultimo tick rilevato: {last_str}\n"
        f"Silenzio: {silence_min} minuti\n\n"
        f"Controlla la VPS e riavvia il daemon se necessario:\n"
        f"  .\\scripts\\start_monitor.ps1\n"
        f"  oppure: python -m src.monitor.daemon\n"
    )
    subject = "[Athanor] ⚠️ Monitor OFFLINE"

    if dry_run:
        # FIX: gestisci encoding su Windows (cp1252 non supporta emoji)
        def _p(s: str) -> None:
            try:
                print(s)
            except UnicodeEncodeError:
                print(s.encode("ascii", "replace").decode())
        _p(f"[dry-run] Subject : {subject}")
        _p(f"[dry-run] To      : {ALERT_RECIPIENT}")
        _p(f"[dry-run] Body    :\n{body}")
        return

    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_RECIPIENT]):
        print("ERROR: credenziali SMTP mancanti in .env — impossibile inviare alert", file=sys.stderr)
        sys.exit(1)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = ALERT_RECIPIENT
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
        print(f"[heartbeat] Alert inviato a {ALERT_RECIPIENT} (silence={silence_min}min)")
        _mark_alerted()
    except Exception as exc:
        print(f"ERROR invio alert: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Athanor heartbeat check")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stampa cosa farebbe senza inviare la mail",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)

    if not _is_market_hours(now):
        print(f"[heartbeat] Fuori market hours ({now.strftime('%H:%M')} UTC) — skip.")
        return

    last_ts = _last_daemon_ts()
    if last_ts is None:
        silence_min = 9999
    else:
        silence_min = int((now - last_ts).total_seconds() / 60)

    print(f"[heartbeat] Ultimo tick daemon: {last_ts} | silenzio: {silence_min}min")

    if silence_min < SILENCE_THRESH:
        print(f"[heartbeat] OK — entro soglia {SILENCE_THRESH}min.")
        return

    # Check systemd before alerting: daemon may be alive but idle (no positions)
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "athanor-monitor"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("[heartbeat] Daemon silenzioso ma systemd active — no alert.")
            return
    except FileNotFoundError:
        pass  # systemctl not available (Windows dev machine)

    if _already_alerted_today() and not args.dry_run:
        print("[heartbeat] Alert già inviato oggi — nessun duplicato.")
        return

    print(f"[heartbeat] ALERT: {silence_min}min > {SILENCE_THRESH}min — invio mail.")
    _send_alert(last_ts, silence_min, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
