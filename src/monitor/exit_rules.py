"""
src/monitor/exit_rules.py
──────────────────────────
Pure, stateless exit rule functions.

Each function returns ExitDecision | None:
  - None  → rule not applicable (conditions not met)
  - ExitDecision → rule fires; daemon executes the action

Priority order used by daemon:
  1. evaluate_setup_broken       (urgent: invalidated setup)
  2. evaluate_time_decay         (capital efficiency)
  3. evaluate_partial_profit_take (scale out)
  4. evaluate_trailing_stop      (SL adjustment, no exit)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from src.execution.broker_adapter import ClockInfo
from src.monitor.monitor_state import EnrichedPosition, ExitDecision

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Exponential-smoothed RSI (Wilder method)."""
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _minutes_to_close(clock: ClockInfo) -> float:
    """Minutes remaining until market close. Returns large number if market closed."""
    if not clock.is_open or clock.next_close is None:
        return 9999.0
    now = datetime.now(timezone.utc)
    delta = clock.next_close - now
    return max(delta.total_seconds() / 60.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 1 — Trailing stop adjustment
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_trailing_stop(
    pos: EnrichedPosition,
    atr: float,
) -> ExitDecision | None:
    """
    Raise SL progressively as profit grows:
      profit > 0.5×ATR → SL = breakeven (entry)
      profit > 1.0×ATR → SL = entry + 0.5×ATR
      profit > 1.5×ATR → trailing: SL = max(current_SL, price - 1.0×ATR)

    Returns ADJUST_TRAILING_STOP with new_stop, or None.
    """
    if atr <= 0:
        return None

    price      = pos.effective_price()
    entry      = pos.avg_entry_price
    current_sl = pos.stop_loss
    is_long    = pos.side == "long"

    profit = (price - entry) if is_long else (entry - price)
    profit_in_atr = profit / atr

    if profit_in_atr <= 0.5:
        return None   # not profitable enough yet

    if profit_in_atr > 1.5:
        # Tight trailing stop
        new_sl = (price - 1.0 * atr) if is_long else (price + 1.0 * atr)
        if current_sl is not None:
            new_sl = max(new_sl, current_sl) if is_long else min(new_sl, current_sl)
        rule = "trailing_1_5x"
    elif profit_in_atr > 1.0:
        new_sl = (entry + 0.5 * atr) if is_long else (entry - 0.5 * atr)
        if current_sl is not None and (
            (is_long  and new_sl <= current_sl) or
            (not is_long and new_sl >= current_sl)
        ):
            return None   # already at or above this level
        rule = "trailing_1x"
    else:
        # profit > 0.5×ATR: move to breakeven
        new_sl = entry
        if current_sl is not None and (
            (is_long  and new_sl <= current_sl) or
            (not is_long and new_sl >= current_sl)
        ):
            return None   # already at or above breakeven
        rule = "trailing_breakeven"

    return ExitDecision(
        action="ADJUST_TRAILING_STOP",
        new_stop=round(new_sl, 4),
        reason=f"profit={profit_in_atr:.2f}×ATR → raise SL (rule: {rule})",
        rule=rule,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rule 2 — Partial profit take
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_partial_profit_take(
    pos: EnrichedPosition,
    entry: float,
    atr: float,
    current_price: float,
    prior_partials_count: int = 0,
) -> ExitDecision | None:
    """
    Scale out as profit reaches ATR milestones:
      1st partial (33%) at profit = 1×ATR
      2nd partial (50% of remaining) at profit = 1.5×ATR

    prior_partials_count: how many PARTIAL_EXIT ticks already executed today.
    """
    if atr <= 0:
        return None

    is_long = pos.side == "long"
    profit  = (current_price - entry) if is_long else (entry - current_price)
    profit_in_atr = profit / atr

    if prior_partials_count == 0 and profit_in_atr >= 1.0:
        return ExitDecision(
            action="PARTIAL_EXIT",
            percentage=0.33,
            reason=f"profit={profit_in_atr:.2f}×ATR — taking first 33% off table",
            rule="partial_1x_atr",
        )

    if prior_partials_count == 1 and profit_in_atr >= 1.5:
        return ExitDecision(
            action="PARTIAL_EXIT",
            percentage=0.50,
            reason=f"profit={profit_in_atr:.2f}×ATR — taking 50% of remaining position",
            rule="partial_1_5x_atr",
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Rule 3 — Setup broken detection
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_setup_broken(
    pos: EnrichedPosition,
    ohlcv_5min: pd.DataFrame,
    clock: ClockInfo,
    avg_daily_volume: Optional[float] = None,
) -> ExitDecision | None:
    """
    Detect invalidated setup via RSI(14) inversion > 20 points against
    position in last 3 bars (15 min).

    ohlcv_5min must have columns: open, high, low, close (lowercase).
    """
    if ohlcv_5min is None or ohlcv_5min.empty or len(ohlcv_5min) < 20:
        return None

    df = ohlcv_5min.rename(columns=str.lower)
    if "close" not in df.columns:
        return None

    is_long = pos.side == "long"

    # ── RSI inversion check ───────────────────────────────────────────────────
    try:
        rsi = _rsi(df["close"], period=14)
        if len(rsi) >= 4:
            rsi_now  = float(rsi.iloc[-1])
            rsi_prev = float(rsi.iloc[-4])   # 15 minutes ago (3 bars back)
            rsi_delta = rsi_now - rsi_prev    # positive = RSI rising

            # Long position: RSI dropping > 20 pts = bearish signal
            # Short position: RSI rising > 20 pts = bullish signal (against short)
            if is_long and rsi_delta < -20:
                return ExitDecision(
                    action="FULL_EXIT",
                    reason=f"RSI dropped {abs(rsi_delta):.1f} pts in 15min — setup broken (long)",
                    rule="setup_broken_rsi",
                )
            if not is_long and rsi_delta > 20:
                return ExitDecision(
                    action="FULL_EXIT",
                    reason=f"RSI rose {rsi_delta:.1f} pts in 15min — setup broken (short)",
                    rule="setup_broken_rsi",
                )
    except Exception as exc:
        logger.debug("[exit_rules] RSI computation failed: %s", exc)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Rule 4 — Time decay
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_time_decay(
    pos: EnrichedPosition,
    atr: float,
    days_open: Optional[float] = None,
) -> ExitDecision | None:
    """
    Exit if position is stale: open > 3 days without meaningful profit (< 0.3×ATR).
    days_open overrides pos.days_open if provided.
    """
    if atr <= 0:
        return None

    age = days_open if days_open is not None else pos.days_open
    if age < 3.0:
        return None

    profit = pos.profit_per_share()
    profit_in_atr = profit / atr

    if profit_in_atr < 0.3:
        return ExitDecision(
            action="FULL_EXIT",
            reason=(
                f"Time decay: position open {age:.1f} days, "
                f"profit={profit_in_atr:.2f}×ATR < 0.3×ATR threshold"
            ),
            rule="time_decay",
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluator (used by daemon)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_rules(
    pos: EnrichedPosition,
    clock: ClockInfo,
    ohlcv_5min: Optional[pd.DataFrame],
    atr: float,
    avg_daily_volume: Optional[float] = None,
) -> ExitDecision:
    """
    Evaluate all rules in priority order. Returns first applicable decision,
    or HOLD if no rule fires.
    """
    price = pos.effective_price()

    # Priority 1 — setup broken
    if ohlcv_5min is not None and not ohlcv_5min.empty:
        d = evaluate_setup_broken(pos, ohlcv_5min, clock, avg_daily_volume)
        if d:
            return d

    # Priority 2 — time decay
    d = evaluate_time_decay(pos, atr)
    if d:
        return d

    # Priority 3 — partial profit take
    d = evaluate_partial_profit_take(
        pos,
        entry=pos.avg_entry_price,
        atr=atr,
        current_price=price,
        prior_partials_count=pos.prior_partials_count,
    )
    if d:
        return d

    # Priority 4 — trailing stop (no exit, just SL move)
    d = evaluate_trailing_stop(pos, atr)
    if d:
        return d

    return ExitDecision(action="HOLD", reason="no rule triggered", rule="hold")
