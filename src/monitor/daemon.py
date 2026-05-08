"""
src/monitor/daemon.py
──────────────────────
Athanor Alpha — Active Position Monitor Daemon (Fase 3)

Runs every CYCLE_SECONDS during US market hours (14:30–21:00 UTC, Mon–Fri).
Reads live positions from Alpaca, evaluates exit rules, executes decisions.

Exit conditions:
  - .kill_monitor file created in project root → clean shutdown
  - Market closes (16:00 ET / 21:00 UTC) → sys.exit(0)
  - --now flag → single cycle then exit

Usage:
    python -m src.monitor.daemon                    # normal daemon (market hours)
    python -m src.monitor.daemon --now              # one cycle immediately, then exit
    python -m src.monitor.daemon --llm-monitor      # enable LLM review layer
    python -m src.monitor.daemon --cycle 30         # 30-second cycle
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from datetime import datetime, timezone
from math import floor
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
_LOG_DIR = pathlib.Path("logs")
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "monitor.log"

_log_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
log = logging.getLogger("athanor.monitor")

# ── Constants ────────────────────────────────────────────────────────────────
KILL_SWITCH_PATH        = pathlib.Path(".kill_monitor")
ATHANOR_KILL_PATH       = pathlib.Path(".athanor_kill")
DEFAULT_CYCLE_SEC = 60
MARKET_OPEN_UTC   = (14, 30)   # 14:30 UTC = 09:30 ET
MARKET_CLOSE_UTC  = (21,  0)   # 21:00 UTC = 16:00 ET
MARKET_DAYS       = {0, 1, 2, 3, 4}


# ─────────────────────────────────────────────────────────────────────────────
# Market hours helpers
# ─────────────────────────────────────────────────────────────────────────────

def _market_minutes(now: datetime) -> int:
    return now.hour * 60 + now.minute


def is_market_hours(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    if now.weekday() not in MARKET_DAYS:
        return False
    m = _market_minutes(now)
    open_m  = MARKET_OPEN_UTC[0]  * 60 + MARKET_OPEN_UTC[1]
    close_m = MARKET_CLOSE_UTC[0] * 60 + MARKET_CLOSE_UTC[1]
    return open_m <= m < close_m


def is_market_just_closed(now: Optional[datetime] = None) -> bool:
    """Return True if market closed in the last 2 minutes (post-session flush)."""
    now = now or datetime.now(timezone.utc)
    m = _market_minutes(now)
    close_m = MARKET_CLOSE_UTC[0] * 60 + MARKET_CLOSE_UTC[1]
    return close_m <= m <= close_m + 2


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_position_db_data(ticker: str) -> dict:
    """
    Look up stop_loss, take_profit, run_id, opened_at from executed_orders.
    Returns the most recent OPEN_LONG / OPEN_SHORT record for this ticker.
    """
    try:
        from src.db.init_db import get_connection
        conn = get_connection()
        row = conn.execute(
            """
            SELECT stop_loss, take_profit, run_id, submitted_at, quantity
            FROM executed_orders
            WHERE ticker = ? AND action IN ('OPEN_LONG', 'OPEN_SHORT')
              AND status NOT IN ('REJECTED', 'CANCELED')
            ORDER BY submitted_at DESC LIMIT 1
            """,
            (ticker,),
        ).fetchone()
        conn.close()
        if row:
            return {
                "stop_loss":    row["stop_loss"],
                "take_profit":  row["take_profit"],
                "run_id":       row["run_id"],
                "opened_at":    row["submitted_at"],
                "original_qty": row["quantity"],
            }
    except Exception as exc:
        log.warning("[daemon] _get_position_db_data %s failed: %s", ticker, exc)
    return {}


def _count_prior_partials(ticker: str) -> int:
    """Count PARTIAL_EXIT ticks for ticker today."""
    try:
        from src.db.init_db import get_connection
        conn = get_connection()
        today = datetime.now(timezone.utc).date().isoformat()
        row = conn.execute(
            """
            SELECT COUNT(*) FROM monitor_ticks
            WHERE ticker = ? AND decision = 'PARTIAL_EXIT'
              AND DATE(timestamp) = ?
            """,
            (ticker, today),
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _log_tick_to_db(tick) -> None:
    try:
        from src.db.init_db import get_connection, insert_monitor_tick
        conn = get_connection()
        insert_monitor_tick(conn, tick)
        conn.close()
    except Exception as exc:
        log.warning("[daemon] _log_tick_to_db failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Position enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _trading_days_since(opened_at: "datetime") -> float:
    """Count trading days (Mon-Fri) between opened_at and now UTC.
    Returns a float: integer part = full trading days elapsed,
    decimal part = fraction of current trading day elapsed.
    Weekend days count as 0.
    """
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    if now <= opened_at:
        return 0.0
    start_date = opened_at.date()
    end_date   = now.date()
    trading_days = 0
    current = start_date
    while current < end_date:
        if current.weekday() < 5:
            trading_days += 1
        current += timedelta(days=1)
    if now.date().weekday() < 5:
        market_open_utc  = now.replace(hour=14, minute=30, second=0, microsecond=0)
        market_close_utc = now.replace(hour=21, minute=0,  second=0, microsecond=0)
        if now >= market_open_utc:
            elapsed = min(
                (now - market_open_utc).total_seconds(),
                (market_close_utc - market_open_utc).total_seconds(),
            )
            trading_days += elapsed / (6.5 * 3600)
    return float(trading_days)


def _enrich_position(raw_pos, current_price: float) -> "EnrichedPosition":
    from src.monitor.monitor_state import EnrichedPosition

    db_data   = _get_position_db_data(raw_pos.ticker)
    opened_at = None
    days_open  = 0.0

    if db_data.get("opened_at"):
        try:
            from datetime import datetime, timezone
            opened_at = datetime.fromisoformat(db_data["opened_at"].replace("Z", "+00:00"))
            days_open = _trading_days_since(opened_at)
        except Exception:
            pass

    prior_partials = _count_prior_partials(raw_pos.ticker)

    return EnrichedPosition(
        ticker=raw_pos.ticker,
        qty=float(raw_pos.qty),
        market_value=float(raw_pos.market_value),
        avg_entry_price=float(raw_pos.avg_entry_price),
        unrealized_pl=float(raw_pos.unrealized_pl),
        side=str(raw_pos.side),
        current_price=current_price,
        stop_loss=db_data.get("stop_loss"),
        take_profit=db_data.get("take_profit"),
        run_id=db_data.get("run_id"),
        opened_at=opened_at,
        days_open=days_open,
        original_qty=db_data.get("original_qty"),
        prior_partials_count=prior_partials,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decision execution
# ─────────────────────────────────────────────────────────────────────────────

def _execute_decision(
    pos,
    decision,
    adapter,
    run_id: str,
) -> tuple[bool, Optional[str]]:
    """
    Submit the TradeOrder corresponding to the ExitDecision.
    Returns (action_taken, broker_order_id).
    """
    from math import floor as _floor
    from src.execution.orders import TradeOrder
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc)

    if decision.action == "HOLD":
        return False, None

    if decision.action == "ADJUST_TRAILING_STOP":
        order = TradeOrder(
            ticker=pos.ticker,
            action="ADJUST_SL",
            quantity=None,
            notional_usd=None,
            order_type="MARKET",
            limit_price=None,
            stop_loss=decision.new_stop,
            take_profit=None,
            time_in_force="DAY",
            conviction=0.0,
            weighted_conviction=0.0,
            regime_at_decision="CAUTION",
            reasoning=decision.reason[:500],
            agent_contributions={},
            created_at=ts,
            run_id=run_id,
        )
        result = adapter.submit_order(order)
        return result.success, result.broker_order_id

    if decision.action in ("FULL_EXIT", "PARTIAL_EXIT"):
        # FIX: Guard — salta exit se posizione già chiusa sul broker (previene LLY zombie)
        try:
            live_positions = adapter.get_positions()
            live_tickers = {p.ticker for p in live_positions}
            if pos.ticker not in live_tickers:
                log.warning("[daemon] %s: position already closed on broker — skipping exit", pos.ticker)
                return False, None
        except Exception:
            pass  # se non riusciamo a verificare, procediamo normalmente
        qty = int(abs(pos.qty))
        if decision.action == "PARTIAL_EXIT" and decision.percentage:
            qty = max(1, _floor(abs(pos.qty) * decision.percentage))

        order = TradeOrder(
            ticker=pos.ticker,
            action="SCALE_OUT" if decision.action == "PARTIAL_EXIT" else "CLOSE",
            quantity=qty,
            notional_usd=None,
            order_type="MARKET",
            limit_price=None,
            stop_loss=None,
            take_profit=None,
            time_in_force="DAY",
            conviction=0.0,
            weighted_conviction=0.0,
            regime_at_decision="CAUTION",
            reasoning=decision.reason[:500],
            agent_contributions={},
            created_at=ts,
            run_id=run_id,
        )
        result = adapter.submit_order(order)

        submitted_at = datetime.now(timezone.utc).isoformat()
        try:
            from src.db.init_db import get_connection, insert_executed_order
            conn = get_connection()
            # Use the original open order's run_id (exists in pipeline_runs).
            # Fallback: ensure monitor run_id has a pipeline_runs row (FK constraint).
            insert_run_id = getattr(pos, "run_id", None) or run_id
            if insert_run_id == run_id:
                conn.execute(
                    "INSERT OR IGNORE INTO pipeline_runs"
                    " (run_id, started_at, status, tickers) VALUES (?, ?, ?, ?)",
                    (run_id, submitted_at, "running", "[]"),
                )
                conn.commit()
            insert_executed_order(conn, insert_run_id, order, result, submitted_at)
            conn.close()
        except Exception as _e:
            log.warning("[daemon] insert_executed_order failed: %s", _e)

        return result.success, result.broker_order_id

    return False, None


# ─────────────────────────────────────────────────────────────────────────────
# Daemon
# ─────────────────────────────────────────────────────────────────────────────

class ActiveMonitorDaemon:
    def __init__(
        self,
        cycle_seconds: int = DEFAULT_CYCLE_SEC,
        llm_monitor: bool = False,
        run_id: str = "monitor",
        adapter=None,   # injectable for testing; defaults to AlpacaBrokerAdapter()
    ):
        if adapter is not None:
            self._adapter = adapter
        else:
            from src.execution.alpaca_adapter import AlpacaBrokerAdapter
            self._adapter = AlpacaBrokerAdapter()
        self._cycle_sec   = cycle_seconds
        self._llm_monitor = llm_monitor
        self._run_id      = run_id

    def _should_kill(self) -> bool:
        if KILL_SWITCH_PATH.exists():
            log.info("[daemon] .kill_monitor detected — shutting down cleanly.")
            return True
        if ATHANOR_KILL_PATH.exists():
            log.critical("[daemon] .athanor_kill detected — closing all positions and exiting.")
            try:
                from src.risk.kill_switch import close_all_and_exit
                close_all_and_exit(self._adapter)
            except SystemExit:
                raise
            except Exception as exc:
                log.error("[daemon] close_all_and_exit failed: %s", exc)
                sys.exit(1)
        return False

    # ── Single tick ───────────────────────────────────────────────────────────

    def _run_circuit_breaker_checks(self, raw_positions: list) -> None:
        """Run CB checks each tick; handle CB2 force-closes inline."""
        try:
            from src.risk.circuit_breakers import check_all, force_close_cb2_positions
            statuses = check_all(self._adapter)
            triggered = [s for s in statuses if s.triggered]
            for s in triggered:
                log.warning("[daemon] %s TRIGGERED: %s", s.cb_id, s.reason)
            cb2 = [s for s in statuses if s.cb_id == "CB2" and s.triggered]
            if cb2:
                force_close_cb2_positions(self._adapter, cb2)
        except Exception as exc:
            log.warning("[daemon] CB check error: %s", exc)

    def run_tick(self) -> list:
        from src.monitor.monitor_state import MonitorTick
        from src.monitor.exit_rules import evaluate_all_rules
        from src.monitor.price_checker import (
            get_ohlcv_5min, get_daily_atr, get_avg_daily_volume, get_current_price,
        )

        try:
            clock     = self._adapter.get_clock()
            raw_positions = self._adapter.get_positions()
        except Exception as exc:
            log.error("[daemon] Failed to get positions/clock from Alpaca: %s", exc)
            return []

        # ── Circuit breaker checks (Fase 8) ──────────────────────────────────
        self._run_circuit_breaker_checks(raw_positions)

        if not raw_positions:
            log.info("[daemon] No open positions — nothing to evaluate.")
            return []

        log.info("[daemon] Evaluating %d position(s): %s",
                 len(raw_positions), [p.ticker for p in raw_positions])

        ticks = []
        for raw in raw_positions:
            ticker = raw.ticker
            try:
                current_price = get_current_price(ticker) or (
                    abs(float(raw.market_value) / float(raw.qty))
                    if raw.qty else float(raw.avg_entry_price)
                )
                pos         = _enrich_position(raw, current_price)
                ohlcv_5min  = get_ohlcv_5min(ticker)
                atr         = get_daily_atr(ticker)
                avg_vol     = get_avg_daily_volume(ticker)
                decision = evaluate_all_rules(
                    pos=pos,
                    clock=clock,
                    ohlcv_5min=ohlcv_5min,
                    atr=atr,
                    avg_daily_volume=avg_vol,
                )

                # Optional LLM layer
                if (
                    self._llm_monitor
                    and decision.action == "HOLD"
                    and atr > 0
                ):
                    from src.monitor.llm_reviewer import should_review, review_position
                    if should_review(pos, atr):
                        llm_d = review_position(pos, atr, ohlcv_5min)
                        if llm_d:
                            decision = llm_d
                            log.info("[daemon] LLM override for %s: %s", ticker, decision.action)

                # Execute
                pl_pct = (
                    float(raw.unrealized_pl) / (float(raw.avg_entry_price) * abs(float(raw.qty)))
                    if raw.avg_entry_price and raw.qty else None
                )

                action_taken, broker_id = False, None
                if decision.action != "HOLD":
                    log.info(
                        "[daemon] %s → %s | %s",
                        ticker, decision.action, decision.reason,
                    )
                    action_taken, broker_id = _execute_decision(
                        pos, decision, self._adapter, self._run_id
                    )

                tick = MonitorTick(
                    timestamp=datetime.now(timezone.utc),
                    ticker=ticker,
                    current_price=current_price,
                    unrealized_pl_pct=pl_pct,
                    decision=decision.action,
                    reason=decision.reason,
                    action_taken=action_taken,
                    broker_order_id=broker_id,
                    rule=decision.rule,
                )
                _log_tick_to_db(tick)
                ticks.append(tick)

                log.info(
                    "[daemon] %s | price=%.2f | P&L=%.2f%% | %s | action_taken=%s",
                    ticker, current_price,
                    (pl_pct or 0) * 100,
                    decision.action,
                    action_taken,
                )

            except Exception as exc:
                log.error("[daemon] Error processing %s: %s", ticker, exc, exc_info=True)

        return ticks

    # Email removed from daemon — all reporting via EOD cron (postmarket_builder).

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("Monitor started | cycle=%ds | llm=%s | log=%s",
                 self._cycle_sec, self._llm_monitor, _LOG_FILE)
        log.info("Kill switch: create file '%s' to stop cleanly.", KILL_SWITCH_PATH)

        try:
            while True:
                if self._should_kill():
                    sys.exit(0)

                now = datetime.now(timezone.utc)

                if not is_market_hours(now):
                    if is_market_just_closed(now):
                        log.info("[daemon] Market closed — exiting (EOD email handled by cron).")
                        sys.exit(0)
                    log.debug("[daemon] Outside market hours (%s UTC) — sleeping 5min.", now.strftime("%H:%M"))
                    time.sleep(5 * 60)
                    continue

                self.run_tick()

                if self._should_kill():
                    sys.exit(0)

                time.sleep(self._cycle_sec)

        except KeyboardInterrupt:
            log.info("[daemon] Stopped by user (Ctrl+C).")
            sys.exit(0)



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import uuid

    parser = argparse.ArgumentParser(description="Athanor Alpha Active Monitor Daemon")
    parser.add_argument("--now", action="store_true",
                        help="Run one cycle immediately then exit (ignore market hours).")
    parser.add_argument("--llm-monitor", action="store_true",
                        help="Enable optional LLM review layer (Haiku).")
    parser.add_argument("--cycle", type=int, default=DEFAULT_CYCLE_SEC,
                        help=f"Cycle interval in seconds (default: {DEFAULT_CYCLE_SEC}).")
    args = parser.parse_args()

    run_id = f"monitor-{uuid.uuid4().hex[:8]}"
    daemon = ActiveMonitorDaemon(
        cycle_seconds=args.cycle,
        llm_monitor=args.llm_monitor,
        run_id=run_id,
    )

    if args.now:
        log.info("--now: running single cycle then exiting.")
        daemon.run_tick()
        log.info("Done.")
        sys.exit(0)

    daemon.run()


if __name__ == "__main__":
    main()
