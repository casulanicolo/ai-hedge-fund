"""
src/alerts/email_sender.py

SMTP email sender for Athanor Alpha monitor alerts.

Features:
  - Reads credentials from .env (python-dotenv)
  - Sends plain-text + HTML multipart emails
  - Rate limiting: max 1 email per ticker per hour (in-memory)
  - Two send functions:
      send_alert(ticker, subject, body_text, body_html)  → immediate alert
      send_digest(subject, body_text, body_html)         → daily digest (no rate limit)
"""

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

# ── SMTP config from .env ────────────────────────────────────────────────────
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER      = os.getenv("SMTP_USER", "")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT", "")

# ── Rate limiting: {ticker: last_sent_datetime} ──────────────────────────────
_last_sent: dict[str, datetime] = {}
RATE_LIMIT_MINUTES = 60   # max 1 urgent email per ticker per hour


def _is_rate_limited(ticker: str) -> bool:
    """Return True if we sent an alert for this ticker within the rate limit window."""
    last = _last_sent.get(ticker)
    if last is None:
        return False
    elapsed = datetime.now(timezone.utc) - last
    return elapsed < timedelta(minutes=RATE_LIMIT_MINUTES)


def _mark_sent(ticker: str) -> None:
    _last_sent[ticker] = datetime.now(timezone.utc)


def _build_message(
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> MIMEMultipart:
    """Build a MIME multipart email message."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = ALERT_RECIPIENT

    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    return msg


def _send(msg: MIMEMultipart) -> bool:
    """
    Open SMTP connection and send message.
    Returns True on success, False on failure.
    """
    if not SMTP_USER or not SMTP_PASSWORD or not ALERT_RECIPIENT:
        log.error("SMTP credentials not configured in .env — cannot send email.")
        return False

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ALERT_RECIPIENT, msg.as_string())
        log.info(f"Email sent → {ALERT_RECIPIENT} | subject: {msg['Subject']}")
        return True
    except smtplib.SMTPAuthenticationError:
        log.error("SMTP authentication failed — check SMTP_USER and SMTP_PASSWORD in .env")
        return False
    except smtplib.SMTPException as exc:
        log.error(f"SMTP error: {exc}")
        return False
    except Exception as exc:
        log.error(f"Unexpected error sending email: {exc}")
        return False


# ── Public API ───────────────────────────────────────────────────────────────

def send_alert(
    ticker: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    Send an immediate alert email for `ticker`.

    Respects rate limiting (max 1 per ticker per hour) unless force=True.
    Returns True if email was sent, False if skipped or failed.
    """
    if not force and _is_rate_limited(ticker):
        last = _last_sent[ticker]
        minutes_ago = int((datetime.now(timezone.utc) - last).total_seconds() / 60)
        log.info(
            f"Rate limited [{ticker}] — last alert sent {minutes_ago} min ago. "
            f"Skipping. (use force=True to override)"
        )
        return False

    msg = _build_message(subject, body_text, body_html)
    success = _send(msg)
    if success:
        _mark_sent(ticker)
    return success


def send_digest(
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> bool:
    """
    Send a digest email (daily summary, weekly review).
    No rate limiting applied.
    Returns True if sent successfully.
    """
    msg = _build_message(subject, body_text, body_html)
    return _send(msg)


def test_connection() -> bool:
    """
    Test SMTP credentials by opening a connection without sending.
    Returns True if login succeeds.
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        log.error("SMTP credentials missing in .env")
        return False
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
        log.info(f"SMTP connection OK — logged in as {SMTP_USER}")
        return True
    except smtplib.SMTPAuthenticationError:
        log.error("SMTP authentication failed — check credentials in .env")
        return False
    except Exception as exc:
        log.error(f"SMTP connection failed: {exc}")
        return False


# ── Quick standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("Testing SMTP connection...")
    ok = test_connection()
    if not ok:
        print("Connection test FAILED. Check .env credentials.")
        sys.exit(1)

    print("\nSending test email...")
    sent = send_alert(
        ticker    = "TEST",
        subject   = "[TEST] Athanor Alpha — SMTP connection verified",
        body_text = (
            "This is a test email from Athanor Alpha Monitor.\n\n"
            "If you received this, SMTP is configured correctly.\n\n"
            f"Sent at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        ),
        force=True,   # bypass rate limit for test
    )

    if sent:
        print(f"\n✓ Test email sent to {ALERT_RECIPIENT}")
        print("Check your inbox.")
    else:
        print("\n✗ Failed to send test email. Check logs above.")
