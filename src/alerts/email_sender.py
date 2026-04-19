"""
src/alerts/email_sender.py
───────────────────────────
Exactly 3 public functions:

  send_premarket(html, text)     — no rate limit (cron-driven, once/day)
  send_postmarket(html, text)    — no rate limit (cron-driven, once/day)
  send_critical_alert(subj, body) — max 1/hour; for blocking runtime errors only

SMTP credentials from .env: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_RECIPIENT
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("athanor.email")

SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER       = os.getenv("SMTP_USER", "")
SMTP_PASSWORD   = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT", "")

# Rate limit for send_critical_alert only
_critical_sent_times: list[datetime] = []
CRITICAL_HOURLY_CAP = 1


def _is_critical_rate_limited() -> bool:
    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    while _critical_sent_times and _critical_sent_times[0] < cutoff:
        _critical_sent_times.pop(0)
    return len(_critical_sent_times) >= CRITICAL_HOURLY_CAP


def _send(subject: str, body_text: str, body_html: Optional[str] = None) -> bool:
    """Core SMTP send. Returns True on success."""
    if not SMTP_USER or not SMTP_PASSWORD or not ALERT_RECIPIENT:
        log.error("SMTP credentials not configured in .env")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = ALERT_RECIPIENT
        msg.attach(MIMEText(body_text, "plain", "utf-8"))
        if body_html:
            msg.attach(MIMEText(body_html, "html", "utf-8"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as srv:
            srv.ehlo(); srv.starttls(); srv.ehlo()
            srv.login(SMTP_USER, SMTP_PASSWORD)
            srv.sendmail(SMTP_USER, ALERT_RECIPIENT, msg.as_string())
        log.info("Email sent → %s | %s", ALERT_RECIPIENT, subject)
        return True
    except smtplib.SMTPAuthenticationError:
        log.error("SMTP auth failed — check SMTP_USER / SMTP_PASSWORD in .env")
        return False
    except Exception as exc:
        log.error("SMTP send failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def send_premarket(html: str, text: str, date: str = "") -> bool:
    """
    Send the daily pre-market briefing email.
    No rate limit — called once/day by the cron script.
    """
    subject = f"[Athanor] Pre-Market {date or ''}".strip()
    return _send(subject, text, html)


def send_postmarket(html: str, text: str, date: str = "", summary_line: str = "") -> bool:
    """
    Send the daily post-market EOD digest email.
    No rate limit — called once/day by the cron script.
    """
    subject = f"[Athanor] EOD {date or ''}"
    if summary_line:
        subject += f" — {summary_line}"
    return _send(subject.strip(), text, html)


def send_critical_alert(subject: str, body: str) -> bool:
    """
    Send a critical runtime error alert (e.g. Alpaca connection failed, DB corrupted).
    Rate limited: max 1 email per hour. Returns False if rate-limited or send fails.
    """
    if _is_critical_rate_limited():
        log.warning("Critical alert suppressed (rate limit 1/h): %s", subject)
        return False
    ok = _send(f"[Athanor CRITICAL] {subject}", body)
    if ok:
        _critical_sent_times.append(datetime.now(timezone.utc))
    return ok


def test_connection() -> bool:
    if not SMTP_USER or not SMTP_PASSWORD:
        log.error("SMTP credentials missing in .env")
        return False
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as srv:
            srv.ehlo(); srv.starttls(); srv.ehlo()
            srv.login(SMTP_USER, SMTP_PASSWORD)
        log.info("SMTP OK — logged in as %s", SMTP_USER)
        return True
    except Exception as exc:
        log.error("SMTP connection failed: %s", exc)
        return False


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ok = test_connection()
    sys.exit(0 if ok else 1)
