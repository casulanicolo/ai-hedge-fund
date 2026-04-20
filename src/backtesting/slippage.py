"""
src/backtesting/slippage.py
───────────────────────────
Cost model for backtest fills. Applied at execution time to convert
mid/quote prices into realistic effective fills.

Tiers (round-trip cost = 2 × bps because applied per side):
  - large cap   : 3 bps   (S&P 500 / mega-cap)
  - mid  cap    : 8 bps   (Russell 2000 / liquid mid)
  - crypto      : 15 bps  (24/7, fragmented liquidity)

Side convention: BUY pays UP, SELL pays DOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AssetTier(str, Enum):
    LARGE_CAP = "large_cap"
    MID_CAP   = "mid_cap"
    CRYPTO    = "crypto"


SLIPPAGE_BPS: dict[AssetTier, float] = {
    AssetTier.LARGE_CAP: 3.0,
    AssetTier.MID_CAP:   8.0,
    AssetTier.CRYPTO:    15.0,
}


# Hardcoded large-cap universe (S&P 500 leaders) — extend as needed.
# Anything not in this set, not crypto, and below the cap threshold = mid cap.
_LARGE_CAP_TICKERS: frozenset[str] = frozenset({
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "BRK.B", "BRK-B", "JPM", "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD",
    "CVX", "ABBV", "PFE", "AVGO", "LLY", "MRK", "WMT", "KO", "PEP", "COST",
    "CSCO", "NFLX", "ADBE", "CRM", "ORCL", "INTC", "AMD", "QCOM", "TXN",
    "IBM", "DIS", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW",
    "NKE", "MCD", "SBUX", "T", "VZ", "TMUS", "BA", "CAT", "GE", "RTX",
    "HON", "UPS", "FDX", "LMT", "ABT", "TMO", "DHR", "BMY", "AMGN", "CVS",
    "SPY", "QQQ", "IWM", "DIA",
})


@dataclass(frozen=True)
class SlippageResult:
    side: str               # "BUY" | "SELL"
    tier: AssetTier
    requested_price: float
    fill_price: float
    bps: float
    cost_total: float       # signed cash impact ($)


def classify(ticker: str, market_cap_usd: Optional[float] = None) -> AssetTier:
    """
    Classify a ticker into a slippage tier.

    Order of precedence:
      1. Crypto suffix `-USD` (Alpaca convention)
      2. Hardcoded large-cap set
      3. market_cap ≥ 50B → LARGE_CAP, else MID_CAP
      4. Fallback when no cap data: MID_CAP (conservative)
    """
    sym = ticker.upper()
    if sym.endswith("-USD"):
        return AssetTier.CRYPTO
    if sym in _LARGE_CAP_TICKERS:
        return AssetTier.LARGE_CAP
    if market_cap_usd is not None and market_cap_usd >= 50_000_000_000:
        return AssetTier.LARGE_CAP
    return AssetTier.MID_CAP


def apply_slippage(
    ticker: str,
    side: str,
    quoted_price: float,
    quantity: float,
    *,
    market_cap_usd: Optional[float] = None,
    tier_override: Optional[AssetTier] = None,
) -> SlippageResult:
    """
    Compute the effective fill price after slippage.

    BUY  → fill higher than quote.
    SELL → fill lower than quote.
    """
    if quantity <= 0 or quoted_price <= 0:
        raise ValueError(f"slippage: invalid qty={quantity} price={quoted_price}")
    side_u = side.upper()
    if side_u not in ("BUY", "SELL"):
        raise ValueError(f"slippage: side must be BUY or SELL, got {side}")

    tier = tier_override or classify(ticker, market_cap_usd)
    bps = SLIPPAGE_BPS[tier]
    factor = bps / 10_000.0

    if side_u == "BUY":
        fill = quoted_price * (1.0 + factor)
        cost = (fill - quoted_price) * quantity   # positive = paid more
    else:  # SELL
        fill = quoted_price * (1.0 - factor)
        cost = (quoted_price - fill) * quantity   # positive = received less

    return SlippageResult(
        side=side_u,
        tier=tier,
        requested_price=round(quoted_price, 6),
        fill_price=round(fill, 6),
        bps=bps,
        cost_total=round(cost, 4),
    )
