"""
Athanor Alpha — Sentiment Agent (Phase 3.3)
============================================
Sources:
  - yfinance .news  → recent headlines + summaries (free, no API key)
  - SEC 8-K filings → material events via sec_edgar.py cache

Output written to state["data"]["analyst_signals"]["sentiment_agent"]:
  {
      "signal":          "bullish" | "bearish" | "neutral",
      "confidence":      0.0 – 1.0,
      "sentiment_score": -1.0 – 1.0,
      "event_type":      "earnings" | "M&A" | "regulatory" | "macro" | "other",
      "urgency":         "low" | "medium" | "high",
      "reasoning":       str,
  }
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import yfinance as yf
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

AGENT_ID = "sentiment_agent"

# How many calendar days back to look for news
NEWS_LOOKBACK_DAYS = 7

# Urgency → weight when averaging sentiment scores
URGENCY_WEIGHTS: dict[str, float] = {
    "high": 3.0,
    "medium": 1.5,
    "low": 1.0,
}

# Keywords that map a headline to an event type (checked in order)
EVENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("earnings", [
        "earnings", "eps", "revenue", "quarterly result", "q1", "q2", "q3", "q4",
        "profit", "loss", "beat", "miss", "guidance", "outlook",
    ]),
    ("M&A", [
        "acqui", "merger", "takeover", "bid", "deal", "buyout",
        "spinoff", "spin-off", "divestiture",
    ]),
    ("regulatory", [
        "sec ", "ftc", "doj", "fda", "ruling", "lawsuit", "settlement",
        "fine", "penalty", "antitrust", "regulation", "compliance",
        "investigation", "subpoena",
    ]),
    ("macro", [
        "fed", "federal reserve", "interest rate", "inflation", "gdp",
        "recession", "tariff", "trade war", "geopolit", "ukraine", "china",
        "macro", "economy", "unemployment",
    ]),
]

# Keywords that push urgency to HIGH
HIGH_URGENCY_KEYWORDS = [
    "breaking", "urgent", "halt", "suspend", "bankrupt", "fraud",
    "restatement", "sec investigation", "criminal", "class action",
    "emergency", "catastrophic",
]

# Keywords that suggest MEDIUM urgency
MEDIUM_URGENCY_KEYWORDS = [
    "earnings", "guidance", "merger", "acqui", "settlement", "fine",
    "downgrade", "upgrade", "layoff", "recall", "ceo", "cfo",
]


# ──────────────────────────────────────────────
# Pydantic output model
# ──────────────────────────────────────────────

class SentimentSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall sentiment signal for the ticker."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the signal, 0 = uncertain, 1 = very confident."
    )
    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Aggregated sentiment score: -1 very negative, +1 very positive."
    )
    event_type: Literal["earnings", "M&A", "regulatory", "macro", "other"] = Field(
        description="Dominant event category driving the sentiment."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How time-sensitive this information is."
    )
    reasoning: str = Field(
        description="Concise explanation of the sentiment call (2-4 sentences)."
    )


# ──────────────────────────────────────────────
# Helper: classify a single headline
# ──────────────────────────────────────────────

def _classify_event_type(text: str) -> str:
    """Return the event_type label that best matches the text."""
    lower = text.lower()
    for event_type, keywords in EVENT_PATTERNS:
        if any(kw in lower for kw in keywords):
            return event_type
    return "other"


def _classify_urgency(text: str) -> str:
    """Return urgency level based on keyword presence."""
    lower = text.lower()
    if any(kw in lower for kw in HIGH_URGENCY_KEYWORDS):
        return "high"
    if any(kw in lower for kw in MEDIUM_URGENCY_KEYWORDS):
        return "medium"
    return "low"


# ──────────────────────────────────────────────
# Helper: fetch yfinance news for a ticker
# ──────────────────────────────────────────────

def _fetch_yf_news(ticker: str) -> list[dict[str, Any]]:
    """
    Returns a list of dicts with keys:
      title, summary, published_utc, source, url
    Only items published within NEWS_LOOKBACK_DAYS are included.

    Supports both yfinance structures:
      - Legacy: flat dict with providerPublishTime (int), title, summary
      - New:    nested under item["content"] with pubDate (ISO string)
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)
    try:
        yf_ticker = yf.Ticker(ticker)
        raw = yf_ticker.news or []
    except Exception:
        return []

    items: list[dict[str, Any]] = []
    for item in raw:
        # ── New yfinance structure: everything inside item["content"] ──
        content = item.get("content") or {}
        if isinstance(content, dict) and content.get("title"):
            title   = content.get("title", "")
            summary = content.get("summary") or content.get("description") or ""
            source  = (content.get("provider") or {}).get("displayName", "unknown")
            url     = (content.get("canonicalUrl") or {}).get("url", "")

            pub_str = content.get("pubDate") or content.get("displayTime") or ""
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pub_dt = datetime.now(tz=timezone.utc)

        # ── Legacy yfinance structure: flat dict ──
        else:
            title   = item.get("title") or item.get("headline") or ""
            summary = item.get("summary") or item.get("description") or ""
            source  = item.get("publisher") or item.get("source") or "unknown"
            url     = item.get("link") or item.get("url") or ""

            ts = item.get("providerPublishTime") or item.get("publishTime") or 0
            try:
                pub_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except (ValueError, OSError):
                pub_dt = datetime.now(tz=timezone.utc)

        if not title or pub_dt < cutoff:
            continue

        items.append({
            "title": title,
            "summary": summary,
            "published_utc": pub_dt.isoformat(),
            "source": source,
            "url": url,
        })

    return items


# ──────────────────────────────────────────────
# Helper: fetch 8-K events from state (SEC cache)
# ──────────────────────────────────────────────

def _fetch_8k_events(ticker: str, state: AgentState) -> list[dict[str, Any]]:
    """
    Reads 8-K filings from prefetched SEC data in state if available,
    otherwise returns an empty list (graceful degradation).
    """
    try:
        prefetched = state.get("data", {}).get("prefetched_data", {})
        ticker_data = prefetched.get(ticker, {})
        sec_data = ticker_data.get("sec", {})
        filings_8k = sec_data.get("8-K", []) or []

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS * 4)
        events: list[dict[str, Any]] = []
        for filing in filings_8k[:10]:  # cap at 10 most recent
            filed_str = filing.get("filingDate") or filing.get("filed") or ""
            try:
                filed_dt = datetime.strptime(filed_str[:10], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                filed_dt = datetime.now(tz=timezone.utc)

            if filed_dt < cutoff:
                continue

            description = (
                filing.get("primaryDocDescription")
                or filing.get("description")
                or filing.get("form")
                or "8-K Material Event"
            )
            events.append({
                "title": f"8-K: {description}",
                "summary": filing.get("items") or "",
                "published_utc": filed_dt.isoformat(),
                "source": "SEC EDGAR",
                "url": filing.get("url") or "",
            })
        return events
    except Exception:
        return []


# ──────────────────────────────────────────────
# Core scoring logic
# ──────────────────────────────────────────────

def _score_items(
    items: list[dict[str, Any]]
) -> tuple[float, str, str, list[dict[str, Any]]]:
    """
    Given a list of news/filing items, compute:
      - weighted_score: float in [-1, 1]
      - dominant_event_type: str
      - dominant_urgency: str
      - annotated_items: list with added event_type, urgency fields

    Scoring heuristic (before LLM):
      We give each item a raw score based on simple positive/negative
      financial keywords in title+summary. The LLM will refine this.
    """
    POSITIVE_KW = [
        "beat", "exceed", "record", "growth", "surge", "upgrade", "buy",
        "strong", "profit", "gain", "rally", "approve", "win", "bullish",
        "raised guidance", "raise guidance", "dividend", "buyback",
    ]
    NEGATIVE_KW = [
        "miss", "disappoint", "downgrade", "sell", "loss", "decline",
        "investigation", "lawsuit", "fraud", "recall", "bankrupt", "cut",
        "lower guidance", "reduce guidance", "layoff", "fine", "penalty",
        "bearish", "warning", "risk",
    ]

    annotated: list[dict[str, Any]] = []
    event_type_counts: dict[str, float] = {}
    urgency_counts: dict[str, float] = {}

    total_weight = 0.0
    weighted_sum = 0.0

    for item in items:
        text = (item.get("title") or "") + " " + (item.get("summary") or "")
        lower = text.lower()

        event_type = _classify_event_type(text)
        urgency = _classify_urgency(text)
        weight = URGENCY_WEIGHTS[urgency]

        # Raw polarity score [-1, 1]
        pos = sum(1 for kw in POSITIVE_KW if kw in lower)
        neg = sum(1 for kw in NEGATIVE_KW if kw in lower)
        total_kw = pos + neg
        if total_kw == 0:
            raw_score = 0.0
        else:
            raw_score = (pos - neg) / total_kw  # [-1, 1]

        weighted_sum += raw_score * weight
        total_weight += weight

        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + weight
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + weight

        annotated.append({
            **item,
            "event_type": event_type,
            "urgency": urgency,
            "raw_score": round(raw_score, 3),
        })

    if total_weight == 0:
        return 0.0, "other", "low", annotated

    weighted_score = max(-1.0, min(1.0, weighted_sum / total_weight))
    dominant_event = max(event_type_counts, key=event_type_counts.get)  # type: ignore
    dominant_urgency = max(urgency_counts, key=urgency_counts.get)  # type: ignore

    return round(weighted_score, 4), dominant_event, dominant_urgency, annotated


# ──────────────────────────────────────────────
# Main agent function
# ──────────────────────────────────────────────

def sentiment_agent(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node — Sentiment Agent.

    Reads news + 8-K events for each ticker, computes a heuristic
    sentiment score, then asks the LLM to interpret and produce the
    final structured signal.
    """
    data: dict[str, Any] = state.get("data", {})
    tickers: list[str] = data.get("tickers", [])
    analyst_signals: dict[str, Any] = data.setdefault("analyst_signals", {})
    analyst_signals.setdefault(AGENT_ID, {})

    for ticker in tickers:
        progress.update_status(AGENT_ID, ticker, "fetching news")

        # ── 1. Collect raw items ──────────────────────────────────
        yf_items = _fetch_yf_news(ticker)
        sec_items = _fetch_8k_events(ticker, state)
        all_items = yf_items + sec_items

        if not all_items:
            progress.update_status(AGENT_ID, ticker, "no news found → neutral")
            analyst_signals[AGENT_ID][ticker] = {
                "signal": "neutral",
                "confidence": 0.3,
                "sentiment_score": 0.0,
                "event_type": "other",
                "urgency": "low",
                "reasoning": "No recent news or filings found. Defaulting to neutral.",
            }
            continue

        # ── 2. Heuristic scoring ──────────────────────────────────
        progress.update_status(AGENT_ID, ticker, "scoring sentiment")
        heuristic_score, dominant_event, dominant_urgency, annotated = _score_items(all_items)

        # ── 3. Build LLM prompt ───────────────────────────────────
        progress.update_status(AGENT_ID, ticker, "asking LLM")

        # Limit to top 8 items to keep prompt short
        top_items = sorted(annotated, key=lambda x: URGENCY_WEIGHTS[x["urgency"]], reverse=True)[:8]
        items_json = json.dumps(
            [
                {
                    "title": it["title"],
                    "summary": it["summary"][:300] if it["summary"] else "",
                    "published_utc": it["published_utc"],
                    "source": it["source"],
                    "event_type": it["event_type"],
                    "urgency": it["urgency"],
                    "raw_score": it["raw_score"],
                }
                for it in top_items
            ],
            indent=2,
        )

        prompt = f"""You are an expert financial sentiment analyst.

Ticker: {ticker}
Analysis date: {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")}
Heuristic sentiment score (keyword-based, -1 to +1): {heuristic_score}
Dominant event type detected: {dominant_event}
Dominant urgency detected: {dominant_urgency}

Recent news items and SEC 8-K filings (up to 8, sorted by urgency):
{items_json}

Instructions:
1. Interpret the news holistically. Consider the heuristic score as a starting point
   but use your judgment — it can be wrong.
2. Determine the overall sentiment signal: bullish, bearish, or neutral.
3. Choose the dominant event_type: earnings | M&A | regulatory | macro | other.
4. Assess urgency: high (act now), medium (monitor), low (background noise).
5. Estimate confidence: how clear-cut is the signal?
6. Write a short reasoning (2-4 sentences).

Return ONLY a valid JSON object matching this schema — no extra text:
{{
  "signal":          "bullish" | "bearish" | "neutral",
  "confidence":      0.0 to 1.0,
  "sentiment_score": -1.0 to 1.0,
  "event_type":      "earnings" | "M&A" | "regulatory" | "macro" | "other",
  "urgency":         "low" | "medium" | "high",
  "reasoning":       "string"
}}
"""

        # ── 4. Call LLM with Pydantic output ─────────────────────
        result: SentimentSignal | None = call_llm(
            prompt=prompt,
            pydantic_model=SentimentSignal,
            agent_name=AGENT_ID,
            state=state,
        )

        if result is None:
            # Fallback: use heuristic only
            if heuristic_score > 0.15:
                signal = "bullish"
            elif heuristic_score < -0.15:
                signal = "bearish"
            else:
                signal = "neutral"

            analyst_signals[AGENT_ID][ticker] = {
                "signal": signal,
                "confidence": 0.4,
                "sentiment_score": heuristic_score,
                "event_type": dominant_event,
                "urgency": dominant_urgency,
                "reasoning": "LLM call failed. Using keyword heuristic fallback.",
            }
        else:
            analyst_signals[AGENT_ID][ticker] = {
                "signal": result.signal,
                "confidence": result.confidence,
                "sentiment_score": result.sentiment_score,
                "event_type": result.event_type,
                "urgency": result.urgency,
                "reasoning": result.reasoning,
            }

        progress.update_status(
            AGENT_ID,
            ticker,
            f"{analyst_signals[AGENT_ID][ticker]['signal'].upper()} "
            f"(score={analyst_signals[AGENT_ID][ticker]['sentiment_score']:+.2f}, "
            f"urgency={analyst_signals[AGENT_ID][ticker]['urgency']})",
        )

    # ── 5. Emit LangGraph message ─────────────────────────────────
    show_agent_reasoning(analyst_signals[AGENT_ID], AGENT_ID)

    msg = HumanMessage(
        content=json.dumps(analyst_signals[AGENT_ID]),
        name=AGENT_ID,
    )
    return {"messages": [msg], "data": data}
