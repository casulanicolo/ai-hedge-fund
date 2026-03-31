"""
email_report.py — Athanor Alpha Phase 3.5
Reads portfolio_recommendations + risk_report from state.
Formats an HTML email and sends it via SMTP (Gmail app password or any provider).
SMTP credentials come from .env:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, REPORT_EMAIL_TO
"""

from __future__ import annotations

import os
import smtplib
import json
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── SMTP config from environment ───────────────────────────────────────────────
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_TO      = os.getenv("REPORT_EMAIL_TO", SMTP_USER)


# ══════════════════════════════════════════════════════════════════════════════
# HTML builder
# ══════════════════════════════════════════════════════════════════════════════

# Action → colour
ACTION_COLOR = {
    "BUY":  "#1a7a4a",   # dark green
    "SELL": "#b03030",   # dark red
    "HOLD": "#7a6a1a",   # dark amber
}

ACTION_BG = {
    "BUY":  "#d4f4e2",
    "SELL": "#fde0e0",
    "HOLD": "#fdf6d4",
}


def _conviction_bar(conviction: float) -> str:
    """Returns an HTML mini progress bar 0-100% for conviction."""
    pct = int(conviction * 100)
    color = "#1a7a4a" if pct >= 60 else "#c47a00" if pct >= 35 else "#888"
    return (
        f'<div style="background:#e0e0e0;border-radius:4px;height:8px;width:120px;display:inline-block;vertical-align:middle;">'
        f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px;"></div>'
        f'</div> <span style="font-size:11px;color:{color};font-weight:600;">{pct}%</span>'
    )


def _trade_plan_rows(recommendations: list[dict]) -> str:
    """
    Generates HTML rows for the operational Trade Plan table.
    Only includes BUY/SELL recommendations that have ATR-based trade levels.
    Sorted by conviction descending (best setup first).
    """
    actionable = [
        r for r in recommendations
        if r.get("action") in ("BUY", "SELL") and r.get("entry_price") is not None
    ]
    actionable.sort(key=lambda r: -r.get("conviction", 0))

    if not actionable:
        return '<tr><td colspan="8" style="padding:12px;color:#888;font-style:italic;">No actionable trade levels available.</td></tr>'

    rows = ""
    for rec in actionable:
        ticker  = rec.get("ticker", "?")
        action  = rec.get("action", "?")
        entry   = rec.get("entry_price")
        sl      = rec.get("stop_loss")
        tp      = rec.get("take_profit")
        size    = rec.get("size_usd")
        rr      = rec.get("rr_ratio", 2.0)
        cons    = rec.get("consensus", "—")
        conv    = rec.get("conviction", 0.0)

        fg = ACTION_COLOR.get(action, "#333")
        bg = ACTION_BG.get(action, "#f9f9f9")

        entry_str = f"${entry:,.2f}" if entry is not None else "—"
        sl_str    = f"${sl:,.2f}"    if sl    is not None else "—"
        tp_str    = f"${tp:,.2f}"    if tp    is not None else "—"
        size_str  = f"${size:,.0f}"  if size  is not None and size > 0 else "—"
        rr_str    = f"1:{rr:.0f}"    if rr    is not None else "—"
        conv_pct  = f"{int(conv * 100)}%"

        # SL in red, TP in green for BUY; reversed for SELL
        sl_color = "#b03030" if action == "BUY" else "#1a7a4a"
        tp_color = "#1a7a4a" if action == "BUY" else "#b03030"

        rows += f"""
        <tr style="border-bottom:1px solid #e8e8e8;">
          <td style="padding:10px 10px;font-weight:700;font-size:14px;">{ticker}</td>
          <td style="padding:10px 8px;">
            <span style="background:{bg};color:{fg};font-weight:700;padding:3px 8px;
                         border-radius:4px;font-size:12px;border:1px solid {fg};">{action}</span>
          </td>
          <td style="padding:10px 8px;font-weight:600;font-size:13px;color:#333;">{entry_str}</td>
          <td style="padding:10px 8px;font-weight:600;font-size:13px;color:{sl_color};">{sl_str}</td>
          <td style="padding:10px 8px;font-weight:600;font-size:13px;color:{tp_color};">{tp_str}</td>
          <td style="padding:10px 8px;font-weight:700;font-size:13px;color:#0d1b2a;">{size_str}</td>
          <td style="padding:10px 8px;font-size:12px;font-weight:600;color:#555;">{rr_str}</td>
          <td style="padding:10px 8px;font-size:11px;color:#555;">{cons}</td>
        </tr>"""
    return rows


def _recommendation_rows(recommendations: list[dict]) -> str:
    rows = ""
    for rec in sorted(recommendations, key=lambda r: -r.get("sizing_pct", 0)):
        ticker   = rec.get("ticker", "?")
        action   = rec.get("action", "HOLD")
        sizing   = rec.get("sizing_pct", 0.0)
        conv     = rec.get("conviction", 0.0)
        reason   = rec.get("reasoning", "")

        fg  = ACTION_COLOR.get(action, "#333")
        bg  = ACTION_BG.get(action, "#f9f9f9")
        bar = _conviction_bar(conv)

        sizing_str = f"{sizing:.1f}%" if sizing > 0 else "—"

        rows += f"""
        <tr style="border-bottom:1px solid #e8e8e8;">
          <td style="padding:10px 12px;font-weight:700;font-size:14px;">{ticker}</td>
          <td style="padding:10px 12px;">
            <span style="background:{bg};color:{fg};font-weight:700;padding:3px 10px;
                         border-radius:4px;font-size:12px;border:1px solid {fg};">{action}</span>
          </td>
          <td style="padding:10px 12px;font-weight:600;font-size:13px;">{sizing_str}</td>
          <td style="padding:10px 12px;">{bar}</td>
          <td style="padding:10px 12px;font-size:12px;color:#555;max-width:320px;">{reason}</td>
        </tr>"""
    return rows


def _warning_block(warnings: list[str]) -> str:
    if not warnings:
        return '<p style="color:#1a7a4a;font-weight:600;">✅ No risk warnings.</p>'
    items = "".join(f"<li style='margin:4px 0;'>{w}</li>" for w in warnings)
    return f"""
    <div style="background:#fff8e1;border-left:4px solid #f5a623;padding:12px 16px;border-radius:4px;">
      <p style="margin:0 0 6px 0;font-weight:700;color:#b07000;">⚠ Risk Warnings</p>
      <ul style="margin:0;padding-left:18px;color:#555;">{items}</ul>
    </div>"""


def build_html(portfolio: dict, risk_report: dict, run_date: str) -> str:
    recs            = portfolio.get("recommendations", [])
    summary         = portfolio.get("portfolio_summary", "")
    risk_notes      = portfolio.get("risk_notes", "")
    total_long      = portfolio.get("total_long_pct", 0.0)
    n_long          = portfolio.get("n_long", 0)
    n_short         = portfolio.get("n_short", 0)
    n_hold          = portfolio.get("n_hold", 0)
    daily_var       = risk_report.get("daily_var_95", 0.0)
    max_dd          = risk_report.get("max_drawdown_estimate", 0.0)
    warnings        = risk_report.get("warnings", [])
    risk_ok         = risk_report.get("risk_ok", True)

    trade_rows = _trade_plan_rows(recs)
    rec_rows   = _recommendation_rows(recs)
    warn_block = _warning_block(warnings)
    risk_badge = (
        '<span style="color:#1a7a4a;font-weight:700;">✅ CLEAR</span>' if risk_ok
        else f'<span style="color:#b03030;font-weight:700;">⚠ {len(warnings)} WARNING{"S" if len(warnings)>1 else ""}</span>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Athanor Alpha — Daily Report</title></head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:'Segoe UI',Arial,sans-serif;">

<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:24px 0;">
<tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0"
       style="background:#ffffff;border-radius:8px;overflow:hidden;
              box-shadow:0 2px 8px rgba(0,0,0,0.10);">

  <!-- Header -->
  <tr>
    <td style="background:#0d1b2a;padding:24px 32px;">
      <h1 style="margin:0;color:#ffffff;font-size:22px;letter-spacing:1px;">
        ⚗ ATHANOR ALPHA
      </h1>
      <p style="margin:4px 0 0 0;color:#a0b4c8;font-size:13px;">
        Daily Portfolio Report &nbsp;·&nbsp; {run_date}
      </p>
    </td>
  </tr>

  <!-- Stats bar -->
  <tr>
    <td style="background:#f0f4f8;padding:16px 32px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="text-align:center;">
            <div style="font-size:22px;font-weight:700;color:#1a7a4a;">{n_long}</div>
            <div style="font-size:11px;color:#666;text-transform:uppercase;">Long</div>
          </td>
          <td style="text-align:center;">
            <div style="font-size:22px;font-weight:700;color:#b03030;">{n_short}</div>
            <div style="font-size:11px;color:#666;text-transform:uppercase;">Short</div>
          </td>
          <td style="text-align:center;">
            <div style="font-size:22px;font-weight:700;color:#888;">{n_hold}</div>
            <div style="font-size:11px;color:#666;text-transform:uppercase;">Hold</div>
          </td>
          <td style="text-align:center;">
            <div style="font-size:22px;font-weight:700;color:#0d1b2a;">{total_long:.1f}%</div>
            <div style="font-size:11px;color:#666;text-transform:uppercase;">Gross Long</div>
          </td>
          <td style="text-align:center;">
            <div style="font-size:16px;font-weight:700;color:#c47a00;">{daily_var*100:.1f}%</div>
            <div style="font-size:11px;color:#666;text-transform:uppercase;">VaR 95%</div>
          </td>
          <td style="text-align:center;">
            <div style="font-size:16px;font-weight:700;color:#c47a00;">{max_dd*100:.1f}%</div>
            <div style="font-size:11px;color:#666;text-transform:uppercase;">Max DD est.</div>
          </td>
          <td style="text-align:center;">
            <div style="font-size:13px;">{risk_badge}</div>
            <div style="font-size:11px;color:#666;text-transform:uppercase;">Risk Status</div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Summary -->
  <tr>
    <td style="padding:24px 32px 12px 32px;">
      <h2 style="margin:0 0 8px 0;font-size:15px;color:#0d1b2a;">Portfolio Stance</h2>
      <p style="margin:0;font-size:13px;color:#444;line-height:1.6;">{summary}</p>
    </td>
  </tr>

  <!-- Trade Plan table -->
  <tr>
    <td style="padding:12px 32px 8px 32px;">
      <h2 style="margin:0 0 4px 0;font-size:15px;color:#0d1b2a;">Trade Plan</h2>
      <p style="margin:0 0 10px 0;font-size:11px;color:#888;">
        Entry = last close &nbsp;·&nbsp; SL = entry &plusmn; 2&times;ATR(14) &nbsp;·&nbsp;
        TP = entry &plusmn; 4&times;ATR(14) &nbsp;·&nbsp; Size = 1% risk per trade
      </p>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border-collapse:collapse;font-size:12px;">
        <thead>
          <tr style="background:#0d1b2a;text-align:left;">
            <th style="padding:7px 10px;font-weight:600;color:#a0b4c8;">Ticker</th>
            <th style="padding:7px 8px;font-weight:600;color:#a0b4c8;">Dir</th>
            <th style="padding:7px 8px;font-weight:600;color:#a0b4c8;">Entry</th>
            <th style="padding:7px 8px;font-weight:600;color:#e07070;">SL</th>
            <th style="padding:7px 8px;font-weight:600;color:#70c070;">TP</th>
            <th style="padding:7px 8px;font-weight:600;color:#a0b4c8;">Size $</th>
            <th style="padding:7px 8px;font-weight:600;color:#a0b4c8;">R:R</th>
            <th style="padding:7px 8px;font-weight:600;color:#a0b4c8;">Agents</th>
          </tr>
        </thead>
        <tbody>
          {trade_rows}
        </tbody>
      </table>
    </td>
  </tr>

  <!-- All signals table -->
  <tr>
    <td style="padding:12px 32px 24px 32px;">
      <h2 style="margin:0 0 12px 0;font-size:15px;color:#0d1b2a;">All Signals</h2>
      <table width="100%" cellpadding="0" cellspacing="0"
             style="border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#f0f4f8;text-align:left;">
            <th style="padding:8px 12px;font-weight:600;color:#555;">Ticker</th>
            <th style="padding:8px 12px;font-weight:600;color:#555;">Action</th>
            <th style="padding:8px 12px;font-weight:600;color:#555;">Size %</th>
            <th style="padding:8px 12px;font-weight:600;color:#555;">Conviction</th>
            <th style="padding:8px 12px;font-weight:600;color:#555;">Reasoning</th>
          </tr>
        </thead>
        <tbody>
          {rec_rows}
        </tbody>
      </table>
    </td>
  </tr>

  <!-- Risk warnings -->
  <tr>
    <td style="padding:0 32px 24px 32px;">
      <h2 style="margin:0 0 8px 0;font-size:15px;color:#0d1b2a;">Risk Notes</h2>
      {warn_block}
      {"" if not risk_notes else f'<p style="margin:10px 0 0 0;font-size:12px;color:#666;">{risk_notes}</p>'}
    </td>
  </tr>

  <!-- Footer -->
  <tr>
    <td style="background:#0d1b2a;padding:16px 32px;">
      <p style="margin:0;color:#a0b4c8;font-size:11px;">
        Athanor Alpha · AI-generated report · For informational purposes only ·
        No positions are executed automatically · Generated {run_date} UTC
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Plain-text fallback
# ══════════════════════════════════════════════════════════════════════════════

def build_plaintext(portfolio: dict, risk_report: dict, run_date: str) -> str:
    recs     = portfolio.get("recommendations", [])
    summary  = portfolio.get("portfolio_summary", "")
    warnings = risk_report.get("warnings", [])
    var      = risk_report.get("daily_var_95", 0.0)
    max_dd   = risk_report.get("max_drawdown_estimate", 0.0)

    lines = [
        f"ATHANOR ALPHA — Daily Report — {run_date}",
        "=" * 72,
        "",
        "PORTFOLIO STANCE",
        summary,
        "",
        f"VaR 95%: {var*100:.1f}%  |  Max DD estimate: {max_dd*100:.1f}%",
        "",
        "TRADE PLAN  (Entry=last close, SL=entry+-2xATR, TP=entry+-4xATR, Size=1% risk)",
        f"{'Ticker':<7} {'Dir':<5} {'Entry':>8}  {'SL':>8}  {'TP':>8}  {'Size$':>7}  {'R:R':<5}  Agents",
        "-" * 72,
    ]
    actionable = [
        r for r in recs
        if r.get("action") in ("BUY", "SELL") and r.get("entry_price") is not None
    ]
    actionable.sort(key=lambda r: -r.get("conviction", 0))
    for rec in actionable:
        t    = rec.get("ticker", "?")
        a    = rec.get("action", "?")
        e    = rec.get("entry_price")
        sl   = rec.get("stop_loss")
        tp   = rec.get("take_profit")
        sz   = rec.get("size_usd")
        rr   = rec.get("rr_ratio", 2.0)
        cons = rec.get("consensus", "—")
        e_s  = f"${e:,.2f}"   if e  is not None else "—"
        sl_s = f"${sl:,.2f}"  if sl is not None else "—"
        tp_s = f"${tp:,.2f}"  if tp is not None else "—"
        sz_s = f"${sz:,.0f}"  if sz is not None and sz > 0 else "—"
        rr_s = f"1:{rr:.0f}"  if rr is not None else "—"
        lines.append(f"{t:<7} {a:<5} {e_s:>8}  {sl_s:>8}  {tp_s:>8}  {sz_s:>7}  {rr_s:<5}  {cons}")

    lines += [
        "",
        "ALL SIGNALS",
        f"{'Ticker':<8} {'Action':<6} {'Size':>6}  {'Conv':>5}  Reasoning",
        "-" * 72,
    ]
    for rec in sorted(recs, key=lambda r: -r.get("sizing_pct", 0)):
        t  = rec.get("ticker", "?")
        a  = rec.get("action", "HOLD")
        s  = rec.get("sizing_pct", 0.0)
        c  = rec.get("conviction", 0.0)
        r  = rec.get("reasoning", "")[:55]
        lines.append(f"{t:<8} {a:<6} {s:>5.1f}%  {c:>4.0%}  {r}")

    if warnings:
        lines += ["", "RISK WARNINGS"]
        for w in warnings:
            lines.append(f"  ⚠  {w}")
    else:
        lines += ["", "Risk Status: CLEAR"]

    lines += [
        "",
        "─" * 60,
        "AI-generated report. No positions are executed automatically.",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Send function
# ══════════════════════════════════════════════════════════════════════════════

def send_daily_report(portfolio: dict, risk_report: dict) -> bool:
    """
    Build and send the daily HTML report email.
    Returns True if sent successfully, False on error.
    Call this after portfolio_manager_agent completes.
    """
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_long   = portfolio.get("n_long", 0)
    n_short  = portfolio.get("n_short", 0)
    risk_ok  = risk_report.get("risk_ok", True)
    risk_tag = "✅ CLEAR" if risk_ok else f"⚠ {len(risk_report.get('warnings', []))} WARNINGS"

    subject = (
        f"[Athanor Alpha] Daily Report {run_date[:10]} — "
        f"{n_long}L/{n_short}S — Risk: {risk_tag}"
    )

    html_body  = build_html(portfolio, risk_report, run_date)
    plain_body = build_plaintext(portfolio, risk_report, run_date)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = EMAIL_TO
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    if not SMTP_USER or not SMTP_PASSWORD:
        print("[email_report] SMTP_USER or SMTP_PASSWORD not set in .env — skipping send.")
        print(plain_body)
        return False

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, EMAIL_TO, msg.as_string())
        print(f"[email_report] Report sent to {EMAIL_TO}")
        return True
    except Exception as e:
        print(f"[email_report] Failed to send email: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# CLI preview (run directly to see HTML without sending)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Minimal fake data for preview (includes trade levels from Phase 3.5)
    fake_portfolio = {
        "recommendations": [
            {"ticker": "AAPL", "action": "BUY",  "sizing_pct": 14.2, "conviction": 0.78,
             "reasoning": "Strong multi-agent consensus with high fundamental score.",
             "entry_price": 172.50, "stop_loss": 168.30, "take_profit": 180.90,
             "size_usd": 1294.0, "rr_ratio": 2.0, "consensus": "7/9 bullish"},
            {"ticker": "NVDA", "action": "BUY",  "sizing_pct": 11.5, "conviction": 0.65,
             "reasoning": "Bullish technical regime with momentum confirmation.",
             "entry_price": 875.20, "stop_loss": 856.80, "take_profit": 912.00,
             "size_usd": 954.0, "rr_ratio": 2.0, "consensus": "6/9 bullish"},
            {"ticker": "JPM",  "action": "HOLD", "sizing_pct": 0.0,  "conviction": 0.30,
             "reasoning": "Mixed signals; no clear edge detected.",
             "entry_price": None, "stop_loss": None, "take_profit": None,
             "size_usd": None, "rr_ratio": None, "consensus": "9 agents / mixed"},
            {"ticker": "TSLA", "action": "SELL", "sizing_pct": 6.0,  "conviction": 0.55,
             "reasoning": "Bearish sentiment and weakening fundamentals.",
             "entry_price": 245.00, "stop_loss": 253.80, "take_profit": 227.40,
             "size_usd": 487.0, "rr_ratio": 2.0, "consensus": "6/9 bearish"},
        ],
        "portfolio_summary": (
            "The portfolio is cautiously bullish, concentrated in Technology. "
            "Two long positions are recommended with high conviction. "
            "One short position in TSLA reflects macro headwinds."
        ),
        "risk_notes": "Sector concentration in Technology exceeds 50% of bullish signals.",
        "total_long_pct": 25.7,
        "n_long": 2, "n_short": 1, "n_hold": 1,
    }
    fake_risk = {
        "daily_var_95": 0.028,
        "max_drawdown_estimate": 0.11,
        "warnings": ["Sector concentration: 67% of bullish signals are Technology"],
        "risk_ok": False,
    }

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = build_html(fake_portfolio, fake_risk, run_date)

    preview_path = "email_preview.html"
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Preview written to {preview_path} — open it in your browser.")
    print()
    print(build_plaintext(fake_portfolio, fake_risk, run_date))
