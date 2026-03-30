"""
src/monitor/news_checker.py

Fetches recent news headlines for a ticker via yfinance and assigns
a heuristic urgency score based on keyword matching.

Urgency levels:
  high   → immediate alert warranted
  medium → worth noting in summary
  low    → routine news, no alert
"""

import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Optional


# ── Keyword lists ────────────────────────────────────────────────────────────
HIGH_URGENCY_KEYWORDS = [
    "crash", "collapse", "bankrupt", "fraud", "sec investigation",
    "class action", "delisted", "halt", "suspended", "emergency",
    "recall", "breach", "hack", "cyberattack", "indicted", "arrested",
    "profit warning", "revenue warning", "guidance cut", "downgrade",
    "missed", "miss", "disappoints", "layoffs", "ceo resign", "cfo resign",
    "acquisition", "merger", "buyout", "takeover", "hostile bid",
    "earnings miss", "beat expectations",
]

MEDIUM_URGENCY_KEYWORDS = [
    "earnings", "revenue", "guidance", "outlook", "forecast",
    "dividend", "buyback", "split", "spinoff", "ipo", "offering",
    "partnership", "contract", "deal", "lawsuit", "settlement",
    "fda", "approval", "regulatory", "investigation", "probe",
    "upgrade", "raised", "lowered", "target price", "analyst",
    "insider", "filing", "sec", "10-k", "10-q",
]

# How many hours back to look for news
NEWS_LOOKBACK_HOURS = 4


def _score_headline(headline: str) -> tuple[str, list[str]]:
    """
    Return (urgency_level, matched_keywords) for a single headline.
    Priority: high > medium > low.
    """
    lower = headline.lower()
    matched_high = [kw for kw in HIGH_URGENCY_KEYWORDS if kw in lower]
    if matched_high:
        return "high", matched_high

    matched_medium = [kw for kw in MEDIUM_URGENCY_KEYWORDS if kw in lower]
    if matched_medium:
        return "medium", matched_medium

    return "low", []


def check_news(ticker: str) -> dict:
    """
    Fetch recent news for `ticker` and evaluate urgency.

    Returns a dict with:
        ticker          str
        articles        list[dict]   all fetched articles (title, published, urgency, keywords)
        high_urgency    list[dict]   only high-urgency articles
        medium_urgency  list[dict]   only medium-urgency articles
        top_urgency     str          overall max urgency ("high"/"medium"/"low")
        alerts          list[str]    human-readable alert messages
        error           str | None
    """
    result = {
        "ticker": ticker,
        "articles": [],
        "high_urgency": [],
        "medium_urgency": [],
        "top_urgency": "low",
        "alerts": [],
        "error": None,
    }

    try:
        tk = yf.Ticker(ticker)
        raw_news = tk.news  # list of dicts

        if not raw_news:
            return result

        cutoff = datetime.now(timezone.utc) - timedelta(hours=NEWS_LOOKBACK_HOURS)

        for item in raw_news:
            # yfinance news item keys vary slightly by version
            content = item.get("content", {})
            title = (
                content.get("title")
                or item.get("title")
                or ""
            )
            if not title:
                continue

            # Published timestamp
            pub_raw = (
                content.get("pubDate")
                or item.get("providerPublishTime")
                or None
            )
            published: Optional[datetime] = None
            if isinstance(pub_raw, (int, float)):
                published = datetime.fromtimestamp(pub_raw, tz=timezone.utc)
            elif isinstance(pub_raw, str):
                try:
                    published = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                except ValueError:
                    pass

            # Skip articles outside lookback window
            if published and published < cutoff:
                continue

            urgency, keywords = _score_headline(title)
            pub_str = published.strftime("%H:%M UTC") if published else "unknown time"

            article = {
                "title": title,
                "published": pub_str,
                "urgency": urgency,
                "keywords": keywords,
            }
            result["articles"].append(article)

            if urgency == "high":
                result["high_urgency"].append(article)
            elif urgency == "medium":
                result["medium_urgency"].append(article)

        # ── Overall urgency ──────────────────────────────────────────────
        if result["high_urgency"]:
            result["top_urgency"] = "high"
        elif result["medium_urgency"]:
            result["top_urgency"] = "medium"

        # ── Build alerts ─────────────────────────────────────────────────
        for art in result["high_urgency"]:
            result["alerts"].append(
                f"NEWS HIGH [{ticker}] {art['published']} — {art['title']} "
                f"(keywords: {', '.join(art['keywords'])})"
            )

        for art in result["medium_urgency"]:
            result["alerts"].append(
                f"NEWS MEDIUM [{ticker}] {art['published']} — {art['title']} "
                f"(keywords: {', '.join(art['keywords'])})"
            )

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── Quick standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT"]

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] News checker test")
    print(f"Looking back {NEWS_LOOKBACK_HOURS}h\n")

    for ticker in test_tickers:
        res = check_news(ticker)
        print(f"── {ticker} ──────────────────────────")
        if res["error"]:
            print(f"  ERROR: {res['error']}")
        else:
            total = len(res["articles"])
            print(f"  Articles in window : {total}")
            print(f"  Top urgency        : {res['top_urgency'].upper()}")
            if res["alerts"]:
                print(f"  ⚠ Alerts:")
                for a in res["alerts"]:
                    print(f"    → {a}")
            else:
                print(f"  ✓ No alerts")
            if total > 0:
                print(f"  Recent headlines:")
                for art in res["articles"][:3]:
                    print(f"    [{art['urgency'].upper():6}] {art['published']} — {art['title'][:80]}")
        print()
