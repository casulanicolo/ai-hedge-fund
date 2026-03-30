"""
src/alerts/templates.py

HTML email templates for Athanor Alpha alerts.

Two templates:
  build_alert_html(alert_dict)   → urgent single-ticker alert
  build_digest_html(alerts_list) → end-of-day summary of all positions
"""

from datetime import datetime, timezone


# ── Shared style ─────────────────────────────────────────────────────────────
_CSS = """
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 20px; }
    .container { max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
    .header { padding: 20px 28px; color: #ffffff; }
    .header-high   { background: #c0392b; }
    .header-medium { background: #d35400; }
    .header-low    { background: #27ae60; }
    .header-digest { background: #2c3e50; }
    .header h1 { margin: 0; font-size: 20px; letter-spacing: 0.5px; }
    .header p  { margin: 4px 0 0; font-size: 13px; opacity: 0.85; }
    .body { padding: 24px 28px; }
    .section-title { font-size: 12px; font-weight: 700; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; margin: 20px 0 8px; border-bottom: 1px solid #ecf0f1; padding-bottom: 4px; }
    .kv-table { width: 100%; border-collapse: collapse; margin-bottom: 8px; }
    .kv-table td { padding: 6px 4px; font-size: 14px; vertical-align: top; }
    .kv-table td:first-child { color: #7f8c8d; width: 40%; }
    .kv-table td:last-child  { color: #2c3e50; font-weight: 600; }
    .alert-box { background: #fdf2f2; border-left: 4px solid #c0392b; padding: 10px 14px; border-radius: 0 4px 4px 0; margin: 8px 0; font-size: 13px; color: #922b21; }
    .alert-box-medium { background: #fef5ec; border-left-color: #d35400; color: #935116; }
    .news-item { padding: 6px 0; border-bottom: 1px solid #f0f0f0; font-size: 13px; }
    .news-item:last-child { border-bottom: none; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700; margin-right: 6px; }
    .badge-high   { background: #fadbd8; color: #922b21; }
    .badge-medium { background: #fdebd0; color: #935116; }
    .badge-low    { background: #eafaf1; color: #1e8449; }
    .ticker-card { border: 1px solid #ecf0f1; border-radius: 6px; padding: 14px 16px; margin: 12px 0; }
    .ticker-card-high   { border-left: 4px solid #c0392b; }
    .ticker-card-medium { border-left: 4px solid #d35400; }
    .ticker-card-low    { border-left: 4px solid #27ae60; }
    .ticker-name { font-size: 16px; font-weight: 700; color: #2c3e50; }
    .price-up   { color: #27ae60; font-weight: 700; }
    .price-down { color: #c0392b; font-weight: 700; }
    .footer { background: #f8f9fa; padding: 14px 28px; font-size: 11px; color: #95a5a6; text-align: center; }
"""


def _header_class(severity: str) -> str:
    return {
        "HIGH":   "header-high",
        "MEDIUM": "header-medium",
        "LOW":    "header-low",
    }.get(severity.upper(), "header-digest")


def _badge(urgency: str) -> str:
    cls = f"badge-{urgency.lower()}"
    return f'<span class="badge {cls}">{urgency.upper()}</span>'


def _price_span(change: float | None) -> str:
    if change is None:
        return "N/A"
    cls = "price-up" if change >= 0 else "price-down"
    sign = "+" if change >= 0 else ""
    return f'<span class="{cls}">{sign}{change:.2%}</span>'


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Template 1: single-ticker urgent alert ───────────────────────────────────

def build_alert_html(alert: dict) -> str:
    """
    Build HTML for a single-ticker HIGH/MEDIUM alert.
    `alert` is a dict returned by alert_builder.build_alerts_for_position().
    """
    ticker   = alert.get("ticker", "N/A")
    severity = alert.get("severity", "HIGH")
    pr       = alert.get("price_result", {})
    nr       = alert.get("news_result", {})

    cp      = pr.get("current_price")
    pc      = pr.get("price_change")
    vr      = pr.get("volume_ratio")
    gap     = pr.get("gap")
    ep      = pr.get("entry_price") or alert.get("entry_price", 0)

    # Price rows
    price_rows = ""
    if cp:
        price_rows += f"<tr><td>Current price</td><td>${cp:.2f}</td></tr>"
    if pc is not None:
        price_rows += f"<tr><td>Change vs entry</td><td>{_price_span(pc)}</td></tr>"
    if vr is not None:
        color = "#c0392b" if vr >= 2.0 else "#2c3e50"
        price_rows += f'<tr><td>Volume ratio</td><td style="color:{color};font-weight:700">{vr:.2f}x 20d avg</td></tr>'
    if gap is not None:
        price_rows += f"<tr><td>Gap open/prev close</td><td>{_price_span(gap)}</td></tr>"

    # Alert boxes
    alert_boxes = ""
    for a in alert.get("all_alerts", []):
        cls = "alert-box" if severity == "HIGH" else "alert-box alert-box-medium"
        alert_boxes += f'<div class="{cls}">⚠ {a}</div>'

    # News items
    news_html = ""
    articles = nr.get("articles", [])
    if articles:
        for art in articles[:5]:
            news_html += (
                f'<div class="news-item">'
                f'{_badge(art["urgency"])} '
                f'<span style="color:#7f8c8d;font-size:12px">{art["published"]}</span> '
                f'— {art["title"][:80]}'
                f'</div>'
            )
    else:
        news_html = '<p style="color:#95a5a6;font-size:13px">No recent news in lookback window.</p>'

    hdr_cls = _header_class(severity)

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_CSS}</style></head>
<body>
<div class="container">
  <div class="header {hdr_cls}">
    <h1>⚡ Athanor Alpha — {severity} Alert</h1>
    <p>{ticker} · {_now_utc()}</p>
  </div>
  <div class="body">
    <div class="section-title">Position</div>
    <table class="kv-table">
      <tr><td>Ticker</td><td>{ticker}</td></tr>
      <tr><td>Severity</td><td>{severity}</td></tr>
    </table>

    <div class="section-title">Price</div>
    <table class="kv-table">{price_rows}</table>

    {f'<div class="section-title">Triggered Alerts</div>{alert_boxes}' if alert_boxes else ''}

    <div class="section-title">Recent News</div>
    {news_html}
  </div>
  <div class="footer">
    Athanor Alpha Monitor · {_now_utc()} · This is an automated alert. No trade has been executed.
  </div>
</div>
</body>
</html>"""
    return html


# ── Template 2: end-of-day digest ────────────────────────────────────────────

def build_digest_html(alerts: list[dict]) -> str:
    """
    Build HTML for an end-of-day digest covering all positions.
    `alerts` is the list returned by alert_builder.run_all_positions().
    """
    high_count   = sum(1 for a in alerts if a["severity"] == "HIGH")
    medium_count = sum(1 for a in alerts if a["severity"] == "MEDIUM")
    low_count    = sum(1 for a in alerts if a["severity"] == "LOW")

    # Position cards
    cards_html = ""
    for a in alerts:
        ticker   = a.get("ticker", "N/A")
        severity = a.get("severity", "LOW")
        pr       = a.get("price_result", {})
        cp       = pr.get("current_price")
        pc       = pr.get("price_change")
        vr       = pr.get("volume_ratio")
        nr       = a.get("news_result", {})

        card_cls = f"ticker-card ticker-card-{severity.lower()}"
        price_str = f"${cp:.2f} {_price_span(pc)}" if cp and pc is not None else "N/A"
        vol_str   = f"{vr:.2f}x" if vr is not None else "N/A"
        news_urg  = nr.get("top_urgency", "low").upper()
        n_alerts  = len(a.get("all_alerts", []))

        cards_html += f"""
        <div class="{card_cls}">
          <div class="ticker-name">{ticker} <span style="font-size:12px;font-weight:400;color:#7f8c8d">— {severity}</span></div>
          <table class="kv-table" style="margin-top:8px">
            <tr><td>Price</td><td>{price_str}</td></tr>
            <tr><td>Volume ratio</td><td>{vol_str}</td></tr>
            <tr><td>News urgency</td><td>{_badge(news_urg)}</td></tr>
            <tr><td>Alerts fired</td><td>{"⚠ " + str(n_alerts) if n_alerts else "✓ None"}</td></tr>
          </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_CSS}</style></head>
<body>
<div class="container">
  <div class="header header-digest">
    <h1>📊 Athanor Alpha — Daily Digest</h1>
    <p>{_now_utc()} · {len(alerts)} position(s) monitored</p>
  </div>
  <div class="body">
    <div class="section-title">Summary</div>
    <table class="kv-table">
      <tr><td>HIGH alerts</td><td><span style="color:#c0392b;font-weight:700">{high_count}</span></td></tr>
      <tr><td>MEDIUM alerts</td><td><span style="color:#d35400;font-weight:700">{medium_count}</span></td></tr>
      <tr><td>LOW / clear</td><td><span style="color:#27ae60;font-weight:700">{low_count}</span></td></tr>
    </table>

    <div class="section-title">Positions</div>
    {cards_html}
  </div>
  <div class="footer">
    Athanor Alpha Monitor · {_now_utc()} · This is an automated digest. No trade has been executed.
  </div>
</div>
</body>
</html>"""
    return html


# ── Quick standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib

    # Build a fake alert dict for preview
    fake_alert = {
        "ticker": "AAPL",
        "severity": "HIGH",
        "has_alert": True,
        "all_alerts": ["PRICE ALERT [AAPL] UP +41.77% vs entry $175.50 → current $248.80"],
        "price_result": {
            "current_price": 248.80,
            "price_change": 0.4177,
            "volume_ratio": 2.35,
            "gap": -0.0039,
        },
        "news_result": {
            "top_urgency": "low",
            "articles": [
                {"urgency": "low", "published": "06:03 UTC", "title": "Apple announces new product line"},
            ],
        },
    }

    fake_alerts = [
        fake_alert,
        {
            "ticker": "MSFT",
            "severity": "LOW",
            "has_alert": False,
            "all_alerts": [],
            "price_result": {"current_price": 356.77, "price_change": -0.1407, "volume_ratio": 1.13, "gap": -0.0136},
            "news_result": {"top_urgency": "medium", "articles": []},
        },
    ]

    alert_html  = build_alert_html(fake_alert)
    digest_html = build_digest_html(fake_alerts)

    pathlib.Path("alert_preview.html").write_text(alert_html,  encoding="utf-8")
    pathlib.Path("digest_preview.html").write_text(digest_html, encoding="utf-8")

    print("✓ alert_preview.html  — open in browser to preview single alert")
    print("✓ digest_preview.html — open in browser to preview daily digest")
