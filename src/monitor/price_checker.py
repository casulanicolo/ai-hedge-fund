"""
src/monitor/price_checker.py

Fetches current intraday price and volume for a given position,
then checks for alert conditions:
  - price_change > ±3% vs entry price
  - volume spike > 2x 20-day average
  - gap up/down > 1.5% (open vs previous close)
"""

import yfinance as yf
from datetime import datetime
from typing import Optional


# ── Alert thresholds ────────────────────────────────────────────────────────
PRICE_CHANGE_THRESHOLD = 0.03   # ±3%
VOLUME_SPIKE_MULTIPLIER = 2.0   # 2x 20-day average volume
GAP_THRESHOLD = 0.015           # 1.5%


def check_price(ticker: str, entry_price: float) -> dict:
    """
    Fetch current price/volume for `ticker` and evaluate alert conditions
    against `entry_price`.

    Returns a dict with:
        ticker          str
        current_price   float | None
        price_change    float | None   (fraction, e.g. 0.04 = +4%)
        volume          int   | None
        avg_volume_20d  float | None
        volume_ratio    float | None
        today_open      float | None
        prev_close      float | None
        gap             float | None   (fraction)
        alerts          list[str]      (human-readable alert messages)
        error           str   | None
    """
    result = {
        "ticker": ticker,
        "current_price": None,
        "price_change": None,
        "volume": None,
        "avg_volume_20d": None,
        "volume_ratio": None,
        "today_open": None,
        "prev_close": None,
        "gap": None,
        "alerts": [],
        "error": None,
    }

    try:
        tk = yf.Ticker(ticker)

        # ── Current price + today's volume ───────────────────────────────
        # fast_info is lightweight; falls back to info if unavailable
        fast = tk.fast_info
        current_price: Optional[float] = getattr(fast, "last_price", None)
        today_open: Optional[float]    = getattr(fast, "open", None)
        prev_close: Optional[float]    = getattr(fast, "previous_close", None)
        volume: Optional[int]          = getattr(fast, "last_volume", None)

        if current_price is None:
            # fallback: use 1-day intraday history
            hist_1d = tk.history(period="1d", interval="1m")
            if not hist_1d.empty:
                current_price = float(hist_1d["Close"].iloc[-1])
                volume = int(hist_1d["Volume"].sum())

        result["current_price"] = current_price
        result["today_open"]    = today_open
        result["prev_close"]    = prev_close
        result["volume"]        = volume

        # ── 20-day average volume ────────────────────────────────────────
        hist_20d = tk.history(period="25d", interval="1d")
        if not hist_20d.empty and len(hist_20d) >= 5:
            # exclude today's partial session if present
            avg_vol = float(hist_20d["Volume"].iloc[:-1].tail(20).mean())
            result["avg_volume_20d"] = avg_vol
            if volume and avg_vol > 0:
                result["volume_ratio"] = volume / avg_vol

        # ── Price change vs entry ────────────────────────────────────────
        if current_price is not None and entry_price > 0:
            price_change = (current_price - entry_price) / entry_price
            result["price_change"] = price_change

            if abs(price_change) >= PRICE_CHANGE_THRESHOLD:
                direction = "UP" if price_change > 0 else "DOWN"
                result["alerts"].append(
                    f"PRICE ALERT [{ticker}] {direction} {price_change:+.2%} "
                    f"vs entry ${entry_price:.2f} → current ${current_price:.2f}"
                )

        # ── Volume spike ─────────────────────────────────────────────────
        if result["volume_ratio"] is not None:
            if result["volume_ratio"] >= VOLUME_SPIKE_MULTIPLIER:
                result["alerts"].append(
                    f"VOLUME SPIKE [{ticker}] {result['volume_ratio']:.1f}x "
                    f"20-day avg "
                    f"(today: {volume:,} vs avg: {result['avg_volume_20d']:,.0f})"
                )

        # ── Gap up/down ──────────────────────────────────────────────────
        if today_open is not None and prev_close is not None and prev_close > 0:
            gap = (today_open - prev_close) / prev_close
            result["gap"] = gap
            if abs(gap) >= GAP_THRESHOLD:
                direction = "UP" if gap > 0 else "DOWN"
                result["alerts"].append(
                    f"GAP {direction} [{ticker}] {gap:+.2%} "
                    f"(open ${today_open:.2f} vs prev close ${prev_close:.2f})"
                )

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ── Quick standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    test_positions = [
        {"ticker": "AAPL", "entry_price": 175.50},
        {"ticker": "MSFT", "entry_price": 415.20},
    ]

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Price checker test\n")

    for pos in test_positions:
        res = check_price(pos["ticker"], pos["entry_price"])
        print(f"── {res['ticker']} ──────────────────────────")
        if res["error"]:
            print(f"  ERROR: {res['error']}")
        else:
            print(f"  Current price : ${res['current_price']:.2f}" if res["current_price"] else "  Current price : N/A")
            print(f"  Price change  : {res['price_change']:+.2%}" if res["price_change"] is not None else "  Price change  : N/A")
            print(f"  Volume ratio  : {res['volume_ratio']:.2f}x" if res["volume_ratio"] is not None else "  Volume ratio  : N/A")
            print(f"  Gap           : {res['gap']:+.2%}" if res["gap"] is not None else "  Gap           : N/A")
            if res["alerts"]:
                print(f"  ⚠ Alerts:")
                for a in res["alerts"]:
                    print(f"    → {a}")
            else:
                print(f"  ✓ No alerts")
        print()
