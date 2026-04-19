"""
src/alerts/email_sender.py

SMTP email sender for Athanor Alpha monitor alerts.

Features:
  - Reads credentials from .env (python-dotenv)
  - Sends plain-text + HTML multipart emails
  - Rate limiting: max 1 email per ticker per hour (per-ticker)
  - Global rate limit: max 5 emails per hour across all tickers
  - Overflow goes into _digest_buffer; flushed at EOD via flush_digest_buffer()
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

# ── Per-ticker rate limiting ──────────────────────────────────────────────────
_last_sent: dict[str, datetime] = {}
RATE_LIMIT_MINUTES = 60   # max 1 urgent email per ticker per hour

# ── Global hourly rate limiting ───────────────────────────────────────────────
_global_sent_times: list[datetime] = []   # timestamps of emails sent in the last hour
GLOBAL_HOURLY_CAP = 5

# ── Digest buffer — accumulates emails that hit the global cap ────────────────
_digest_buffer: list[dict] = []           # list of {ticker, subject, body_text}


def _is_rate_limited(ticker: str) -> bool:
    """Return True if we sent an alert for this ticker within the rate limit window."""
    last = _last_sent.get(ticker)
    if last is None:
        return False
    elapsed = datetime.now(timezone.utc) - last
    return elapsed < timedelta(minutes=RATE_LIMIT_MINUTES)


def _mark_sent(ticker: str) -> None:
    _last_sent[ticker] = datetime.now(timezone.utc)


def _is_global_rate_limited() -> bool:
    """Return True if we've hit the global hourly cap."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    # Prune stale entries
    while _global_sent_times and _global_sent_times[0] < cutoff:
        _global_sent_times.pop(0)
    return len(_global_sent_times) >= GLOBAL_HOURLY_CAP


def _mark_global_sent() -> None:
    _global_sent_times.append(datetime.now(timezone.utc))


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

    Respects:
      - per-ticker rate limit (max 1/hour per ticker)
      - global rate limit (max 5 emails/hour total)
    Emails that exceed the global cap are buffered in _digest_buffer
    and flushed at EOD by flush_digest_buffer().
    Returns True if email was sent, False if skipped/buffered/failed.
    """
    if not force and _is_rate_limited(ticker):
        last = _last_sent[ticker]
        minutes_ago = int((datetime.now(timezone.utc) - last).total_seconds() / 60)
        log.info(
            f"Rate limited [{ticker}] — last alert sent {minutes_ago} min ago. Buffering."
        )
        _digest_buffer.append({"ticker": ticker, "subject": subject, "body_text": body_text})
        return False

    if not force and _is_global_rate_limited():
        log.info(
            f"Global hourly cap ({GLOBAL_HOURLY_CAP}/h) reached — buffering [{ticker}]: {subject}"
        )
        _digest_buffer.append({"ticker": ticker, "subject": subject, "body_text": body_text})
        return False

    msg = _build_message(subject, body_text, body_html)
    success = _send(msg)
    if success:
        _mark_sent(ticker)
        _mark_global_sent()
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


def flush_digest_buffer() -> bool:
    """
    Send all buffered alerts as a single digest email then clear the buffer.
    Called at EOD by the monitor daemon. Returns True if sent (or buffer empty).
    """
    if not _digest_buffer:
        log.info("[email_sender] Digest buffer empty — nothing to flush.")
        return True

    count = len(_digest_buffer)
    lines = [
        f"[{e['ticker']}] {e['subject']}\n{e['body_text']}\n{'─'*60}"
        for e in _digest_buffer
    ]
    body = f"Athanor Monitor — {count} buffered alert(s)\n\n" + "\n\n".join(lines)
    subject = f"[Athanor] EOD digest — {count} alert(s) not sent in real-time"

    success = send_digest(subject=subject, body_text=body)
    if success:
        log.info("[email_sender] Flushed %d buffered alert(s) as digest.", count)
        _digest_buffer.clear()
    else:
        log.warning("[email_sender] Digest flush failed — buffer retained (%d items).", count)
    return success


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
