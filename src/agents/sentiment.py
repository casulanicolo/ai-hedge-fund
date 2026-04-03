"""
Athanor Alpha – Sentiment Agent (Phase 3.3)
============================================
Sources:
  - yfinance .news  → recent headlines + summaries (free, no API key)
  - SEC 8-K filings → material events via sec_edgar.py cache

Output written to state["data"]["analyst_signals"]["sentiment_agent"]:
  {
      "direction":       "LONG" | "SHORT" | "NEUTRAL",
      "expected_return": float,   # -0.10 to +0.10
      "confidence":      float,   # 0.1 to 1.0
      "sentiment_score": float,   # -1.0 to 1.0
      "event_type":      "earnings" | "M&A" | "regulatory" | "macro" | "other",
      "urgency":         "low" | "medium" | "high",
      "reasoning":       str,
  }
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import yfinance as yf
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress

# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────

AGENT_ID = "sentiment_agent"

NEWS_LOOKBACK_DAYS = 7

URGENCY_WEIGHTS: dict[str, float] = {
    "high":   3.0,
    "medium": 1.5,
    "low":    1.0,
}

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

HIGH_URGENCY_KEYWORDS = [
    "breaking", "urgent", "halt", "suspend", "bankrupt", "fraud",
    "restatement", "sec investigation", "criminal", "class action",
    "emergency", "catastrophic",
]

MEDIUM_URGENCY_KEYWORDS = [
    "earnings", "guidance", "merger", "acqui", "settlement", "fine",
    "downgrade", "upgrade", "layoff", "recall", "ceo", "cfo",
]


# ────────────────────────────────────────────────
# Pydantic output model
# ────────────────────────────────────────────────

class SentimentSignal(BaseModel):
    direction: Literal["LONG", "SHORT", "NEUTRAL"] = Field(
        description="Trading direction based on sentiment: LONG, SHORT, or NEUTRAL."
    )
    expected_return: float = Field(
        ge=-0.10, le=0.10,
        description="Realistic return estimate for a 3-4 day swing trade. E.g. 0.02 = +2%."
    )
    confidence: float = Field(
        ge=0.1, le=1.0,
        description="Confidence in the signal, 0.1 = uncertain, 1.0 = very confident."
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


# ────────────────────────────────────────────────
# Helper: classify a single headline
# ────────────────────────────────────────────────

def _classify_event_type(text: str) -> str:
    lower = text.lower()
    for event_type, keywords in EVENT_PATTERNS:
        if any(kw in lower for kw in keywords):
            return event_type
    return "other"


def _classify_urgency(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in HIGH_URGENCY_KEYWORDS):
        return "high"
    if any(kw in lower for kw in MEDIUM_URGENCY_KEYWORDS):
        return "medium"
    return "low"


# ────────────────────────────────────────────────
# Helper: fetch yfinance news
# ────────────────────────────────────────────────

def _fetch_yf_news(ticker: str) -> list[dict[str, Any]]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)
    try:
        yf_ticker = yf.Ticker(ticker)
        raw = yf_ticker.news or []
    except Exception:
        return []

    items: list[dict[str, Any]] = []
    for item in raw:
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
            "title":         title,
            "summary":       summary,
            "published_utc": pub_dt.isoformat(),
            "source":        source,
            "url":           url,
        })

    return items


# ────────────────────────────────────────────────
# Helper: fetch 8-K events from state
# ────────────────────────────────────────────────

def _fetch_8k_events(ticker: str, state: AgentState) -> list[dict[str, Any]]:
    try:
        prefetched  = state.get("data", {}).get("prefetched_data", {})
        ticker_data = prefetched.get(ticker, {})
        sec_data    = ticker_data.get("sec", {})
        filings_8k  = sec_data.get("8-K", []) or []

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS * 4)
        events: list[dict[str, Any]] = []
        for filing in filings_8k[:10]:
            filed_str = filing.get("filingDate") or filing.get("filed") or ""
            try:
                filed_dt = datetime.strptime(filed_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
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
                "title":         f"8-K: {description}",
                "summary":       filing.get("items") or "",
                "published_utc": filed_dt.isoformat(),
                "source":        "SEC EDGAR",
                "url":           filing.get("url") or "",
            })
        return events
    except Exception:
        return []


# ────────────────────────────────────────────────
# Core scoring logic
# ────────────────────────────────────────────────

def _score_items(
    items: list[dict[str, Any]]
) -> tuple[float, str, str, list[dict[str, Any]]]:
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
    urgency_counts:    dict[str, float] = {}
    total_weight  = 0.0
    weighted_sum  = 0.0

    for item in items:
        text  = (item.get("title") or "") + " " + (item.get("summary") or "")
        lower = text.lower()

        event_type = _classify_event_type(text)
        urgency    = _classify_urgency(text)
        weight     = URGENCY_WEIGHTS[urgency]

        pos      = sum(1 for kw in POSITIVE_KW if kw in lower)
        neg      = sum(1 for kw in NEGATIVE_KW if kw in lower)
        total_kw = pos + neg
        raw_score = 0.0 if total_kw == 0 else (pos - neg) / total_kw

        weighted_sum  += raw_score * weight
        total_weight  += weight

        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + weight
        urgency_counts[urgency]       = urgency_counts.get(urgency, 0) + weight

        annotated.append({**item, "event_type": event_type, "urgency": urgency, "raw_score": round(raw_score, 3)})

    if total_weight == 0:
        return 0.0, "other", "low", annotated

    weighted_score   = max(-1.0, min(1.0, weighted_sum / total_weight))
    dominant_event   = max(event_type_counts, key=event_type_counts.get)   # type: ignore
    dominant_urgency = max(urgency_counts,    key=urgency_counts.get)       # type: ignore

    return round(weighted_score, 4), dominant_event, dominant_urgency, annotated


# ────────────────────────────────────────────────
# Main agent function
# ────────────────────────────────────────────────

def sentiment_agent(state: AgentState) -> dict[str, Any]:
    """LangGraph node – Sentiment Agent."""
    data:            dict[str, Any] = state.get("data", {})
    tickers:         list[str]      = data.get("tickers", [])
    analyst_signals: dict[str, Any] = data.setdefault("analyst_signals", {})
    analyst_signals.setdefault(AGENT_ID, {})

    for ticker in tickers:
        data["current_ticker"] = ticker
        progress.update_status(AGENT_ID, ticker, "fetching news")

        # 1. Collect raw items
        yf_items  = _fetch_yf_news(ticker)
        sec_items = _fetch_8k_events(ticker, state)
        all_items = yf_items + sec_items

        if not all_items:
            progress.update_status(AGENT_ID, ticker, "no news found → neutral")
            analyst_signals[AGENT_ID][ticker] = {
                "direction":       "NEUTRAL",
                "expected_return": 0.0,
                "confidence":      0.1,
                "sentiment_score": 0.0,
                "event_type":      "other",
                "urgency":         "low",
                "reasoning":       "No recent news or filings found. Defaulting to neutral.",
            }
            continue

        # 2. Heuristic scoring
        progress.update_status(AGENT_ID, ticker, "scoring sentiment")
        heuristic_score, dominant_event, dominant_urgency, annotated = _score_items(all_items)

        # 3. Build LLM prompt
        progress.update_status(AGENT_ID, ticker, "asking LLM")
        top_items = sorted(annotated, key=lambda x: URGENCY_WEIGHTS[x["urgency"]], reverse=True)[:8]
        items_json = json.dumps(
            [
                {
                    "title":         it["title"],
                    "summary":       it["summary"][:300] if it["summary"] else "",
                    "published_utc": it["published_utc"],
                    "source":        it["source"],
                    "event_type":    it["event_type"],
                    "urgency":       it["urgency"],
                    "raw_score":     it["raw_score"],
                }
                for it in top_items
            ],
            indent=2,
        )

        prompt = f"""You are an expert financial sentiment analyst evaluating {ticker} for a 3-4 day swing trade.

Analysis date: {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")}
Heuristic sentiment score (keyword-based, -1 to +1): {heuristic_score}
Dominant event type detected: {dominant_event}
Dominant urgency detected: {dominant_urgency}

Recent news items and SEC 8-K filings (up to 8, sorted by urgency):
{items_json}

Instructions:
1. Interpret the news holistically. Use the heuristic score as a starting point.
2. Determine direction: LONG (positive sentiment), SHORT (negative sentiment), NEUTRAL.
3. Estimate expected_return: realistic % return for a 3-4 day swing trade based on sentiment strength (e.g. 0.02 = +2%). Must reflect a realistic target, not a theoretical maximum. Range: -0.10 to +0.10. expected_return must reflect the direction (positive for LONG, negative for SHORT, ~0 for NEUTRAL).
4. Assess confidence from 0.1 to 1.0.
5. Choose dominant event_type: earnings | M&A | regulatory | macro | other.
6. Assess urgency: high (act now), medium (monitor), low (background noise).
7. Write a short reasoning (2-4 sentences).

Return ONLY a valid JSON object matching this schema – no extra text:
{{
  "direction":       "LONG" | "SHORT" | "NEUTRAL",
  "expected_return": float between -0.10 and +0.10,
  "confidence":      float between 0.1 and 1.0,
  "sentiment_score": float between -1.0 and +1.0,
  "event_type":      "earnings" | "M&A" | "regulatory" | "macro" | "other",
  "urgency":         "low" | "medium" | "high",
  "reasoning":       "string"
}}
"""

        # 4. Call LLM
        result: SentimentSignal | None = call_llm(
            prompt=prompt,
            pydantic_model=SentimentSignal,
            agent_name=AGENT_ID,
            state=state,
        )

        if result is None:
            if heuristic_score > 0.15:
                direction = "LONG"
                expected_return = round(heuristic_score * 0.05, 4)
            elif heuristic_score < -0.15:
                direction = "SHORT"
                expected_return = round(heuristic_score * 0.05, 4)
            else:
                direction = "NEUTRAL"
                expected_return = 0.0

            analyst_signals[AGENT_ID][ticker] = {
                "direction":       direction,
                "expected_return": expected_return,
                "confidence":      0.1,
                "sentiment_score": heuristic_score,
                "event_type":      dominant_event,
                "urgency":         dominant_urgency,
                "reasoning":       "LLM call failed. Using keyword heuristic fallback.",
            }
        else:
            analyst_signals[AGENT_ID][ticker] = {
                "direction":       result.direction,
                "expected_return": result.expected_return,
                "confidence":      result.confidence,
                "sentiment_score": result.sentiment_score,
                "event_type":      result.event_type,
                "urgency":         result.urgency,
                "reasoning":       result.reasoning,
            }

        sig = analyst_signals[AGENT_ID][ticker]
        progress.update_status(
            AGENT_ID, ticker,
            f"{sig['direction']} er={sig['expected_return']:+.3f} conf={sig['confidence']:.2f} urgency={sig['urgency']}",
        )

    # 5. Emit LangGraph message
    show_agent_reasoning(analyst_signals[AGENT_ID], AGENT_ID)
    msg = HumanMessage(content=json.dumps(analyst_signals[AGENT_ID]), name=AGENT_ID)
    return {"messages": [msg], "data": data}
