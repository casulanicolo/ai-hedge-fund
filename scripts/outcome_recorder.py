"""
src/scripts/outcome_recorder.py

A6 Audit — Outcome Recorder
Chiude le posizioni reali aperte nel DB e registra gli outcome effettivi
nella tabella outcomes, collegando il prezzo di entrata reale al prezzo
di uscita reale per alimentare correttamente il loop EWA.

Logica:
  1. Legge tutte le posizioni OPEN dalla tabella positions nel DB
  2. Per ognuna scarica il prezzo corrente via yfinance
  3. Valuta tre condizioni di chiusura:
       a. take_profit raggiunto (prezzo >= take_profit per LONG, <= per SHORT)
       b. stop_loss raggiunto   (prezzo <= stop_loss  per LONG, >= per SHORT)
       c. time_exit: posizione aperta da >= 4 sessioni di borsa senza exit
  4. Se una condizione è soddisfatta:
       - Aggiorna positions nel DB: status='CLOSED', close_price, pnl_usd,
         pnl_pct, closed_at, note (motivo chiusura)
  5. Cerca la prediction corrispondente (stesso ticker, timestamp >= opened_at)
     e scrive una riga nella tabella outcomes con il ritorno reale calcolato
     dal prezzo di entrata reale (NON da yfinance retroattivo)
  6. Stampa riepilogo delle posizioni chiuse e degli outcome scritti

Eseguibile manualmente:
    python -m src.scripts.outcome_recorder

Schedulabile con cron dopo la chiusura dei mercati (es. 22:00 UTC).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = Path("db") / "hedge_fund.db"

# Soglia sessioni di borsa per time-based exit (deve coincidere con exit_checker.py)
TIME_EXIT_SESSIONS = 4

# Buffer calendario per scaricare prezzi (copre weekend + festivi)
CALENDAR_BUFFER_DAYS = 35

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# yfinance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _yf_download(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download OHLCV da yfinance, sopprimendo i log rumorosi."""
    yf_loggers = [
        logging.getLogger(name)
        for name in logging.Logger.manager.loggerDict
        if "yfinance" in name
    ]
    yf_loggers.append(logging.getLogger("yfinance"))
    original_levels = {lg: lg.level for lg in yf_loggers}
    for lg in yf_loggers:
        lg.setLevel(logging.CRITICAL)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        log.warning(f"yfinance errore per {ticker}: {e}")
        return None
    finally:
        for lg, lvl in original_levels.items():
            lg.setLevel(lvl)

    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df if "Close" in df.columns else None


def get_current_price(ticker: str) -> Optional[float]:
    """Restituisce il prezzo di chiusura più recente disponibile."""
    end   = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    start = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
    df = _yf_download(ticker, start, end)
    if df is None:
        return None
    closes = df["Close"].dropna()
    return float(closes.iloc[-1]) if not closes.empty else None


def count_trading_sessions(opened_at_str: str) -> int:
    """
    Conta quante sessioni di borsa (giorni con dato yfinance disponibile)
    sono passate dalla data di apertura della posizione a oggi.
    Usa SPY come proxy per i giorni di borsa aperti.
    """
    try:
        opened_dt = datetime.fromisoformat(opened_at_str)
        if opened_dt.tzinfo is None:
            opened_dt = opened_dt.replace(tzinfo=timezone.utc)
        start = opened_dt.date().isoformat()
        end   = (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
        df = _yf_download("SPY", start, end)
        if df is None:
            return 0
        # Esclude il giorno di apertura stesso (conta sessioni DOPO l'apertura)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        opened_naive = pd.Timestamp(opened_dt.date())
        after_open = df[df.index > opened_naive]
        return len(after_open)
    except Exception as e:
        log.warning(f"Errore conteggio sessioni: {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# P&L helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_pnl(direction: str, entry_price: float, close_price: float,
                size_usd: float) -> tuple[float, float]:
    """
    Calcola P&L in USD e % per una posizione.

    Returns:
        (pnl_usd, pnl_pct)
    """
    if entry_price == 0:
        return 0.0, 0.0

    if direction.upper() == "LONG":
        pnl_pct = (close_price - entry_price) / entry_price
    else:  # SHORT
        pnl_pct = (entry_price - close_price) / entry_price

    pnl_usd = size_usd * pnl_pct
    return round(pnl_usd, 4), round(pnl_pct, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Exit condition evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_exit(pos: sqlite3.Row, current_price: float,
                  sessions: int) -> Optional[str]:
    """
    Valuta se la posizione deve essere chiusa.

    Returns:
        Stringa con il motivo di chiusura, o None se la posizione
        deve restare aperta.
    """
    direction    = str(pos["direction"]).upper()
    entry_price  = float(pos["entry_price"])
    take_profit  = pos["take_profit"]
    stop_loss    = pos["stop_loss"]

    # --- Take profit ---
    if take_profit is not None:
        tp = float(take_profit)
        if direction == "LONG"  and current_price >= tp:
            return "take_profit"
        if direction == "SHORT" and current_price <= tp:
            return "take_profit"

    # --- Stop loss ---
    if stop_loss is not None:
        sl = float(stop_loss)
        if direction == "LONG"  and current_price <= sl:
            return "stop_loss"
        if direction == "SHORT" and current_price >= sl:
            return "stop_loss"

    # --- Time-based exit ---
    if sessions >= TIME_EXIT_SESSIONS:
        return f"time_exit_{sessions}s"

    return None  # posizione ancora aperta


# ─────────────────────────────────────────────────────────────────────────────
# DB writers
# ─────────────────────────────────────────────────────────────────────────────

def close_position_in_db(conn: sqlite3.Connection, pos_id: int,
                         close_price: float, pnl_usd: float,
                         pnl_pct: float, note: str) -> None:
    """Aggiorna la posizione nel DB come CLOSED."""
    now_utc = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE positions
        SET status      = 'CLOSED',
            close_price = ?,
            pnl_usd     = ?,
            pnl_pct     = ?,
            closed_at   = ?,
            note        = ?
        WHERE id = ?
        """,
        (close_price, pnl_usd, pnl_pct, now_utc, note, pos_id)
    )


def write_real_outcome(conn: sqlite3.Connection, ticker: str,
                       opened_at: str, entry_price: float,
                       close_price: float, direction: str) -> bool:
    """
    Trova la prediction corrispondente e scrive un outcome reale
    nella tabella outcomes per le finestre 1d, 5d, 20d (tutte con
    lo stesso ritorno reale — segnala che la posizione è stata chiusa).

    Returns True se almeno un outcome è stato scritto.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    # Calcola il ritorno reale basato sui prezzi reali di entrata/uscita
    if entry_price == 0:
        return False

    if direction.upper() == "LONG":
        actual_return = (close_price - entry_price) / entry_price
    else:  # SHORT
        actual_return = (entry_price - close_price) / entry_price

    # Cerca la prediction più vicina per questo ticker dopo l'apertura della posizione
    query = """
        SELECT id
        FROM predictions
        WHERE ticker = ?
          AND timestamp >= ?
        ORDER BY timestamp ASC
        LIMIT 1
    """
    row = conn.execute(query, (ticker, opened_at)).fetchone()
    if row is None:
        log.warning(f"  Nessuna prediction trovata per {ticker} >= {opened_at} — outcome non scritto.")
        return False

    prediction_id = row["id"]

    # Controlla se esiste già un outcome per questa prediction (evita duplicati)
    existing = conn.execute(
        "SELECT id FROM outcomes WHERE prediction_id = ? AND window = '1d'",
        (prediction_id,)
    ).fetchone()
    if existing:
        log.info(f"  Outcome 1d già presente per prediction_id={prediction_id} — skip.")
        return False

    # Scrivi outcome per tutte e 3 le finestre con il ritorno reale
    for window, col in [("1d", "actual_return_1d"), ("5d", "actual_return_5d"), ("20d", "actual_return_20d")]:
        existing_w = conn.execute(
            "SELECT id FROM outcomes WHERE prediction_id = ? AND window = ?",
            (prediction_id, window)
        ).fetchone()
        if existing_w:
            continue  # finestra già calcolata

        r1d  = actual_return if window == "1d"  else None
        r5d  = actual_return if window == "5d"  else None
        r20d = actual_return if window == "20d" else None

        conn.execute(
            """
            INSERT INTO outcomes
                (prediction_id, ticker, actual_return_1d, actual_return_5d,
                 actual_return_20d, window, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (prediction_id, ticker, r1d, r5d, r20d, window, now_utc)
        )

    log.info(
        f"  Outcome reale scritto: prediction_id={prediction_id} | "
        f"{ticker} | direction={direction} | "
        f"entry={entry_price:.4f} → close={close_price:.4f} | "
        f"return={actual_return:+.4f} ({actual_return*100:+.2f}%)"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_outcome_recorder() -> None:
    log.info("=" * 60)
    log.info("Outcome Recorder (A6) — avvio")
    log.info("=" * 60)

    if not DB_PATH.exists():
        log.error(f"Database non trovato: {DB_PATH}")
        log.error("Esegui dalla root del progetto athanor-alpha.")
        return

    conn = _get_connection()
    closed_count  = 0
    outcome_count = 0

    try:
        # 1. Carica tutte le posizioni OPEN
        open_positions = conn.execute(
            "SELECT id, ticker, direction, entry_price, stop_loss, take_profit, size_usd, opened_at FROM positions WHERE status = 'OPEN'"
        ).fetchall()

        if not open_positions:
            log.info("Nessuna posizione OPEN nel DB.")
            _print_summary(conn)
            return

        log.info(f"Posizioni OPEN trovate: {len(open_positions)}")

        for pos in open_positions:
            ticker      = pos["ticker"]
            pos_id      = pos["id"]
            direction   = pos["direction"]
            entry_price = float(pos["entry_price"])
            size_usd    = float(pos["size_usd"]) if pos["size_usd"] else 0.0
            opened_at   = pos["opened_at"]

            log.info(f"\n  [{ticker}] id={pos_id} | {direction} | entry={entry_price:.4f} | opened={opened_at}")

            # 2. Prezzo corrente
            current_price = get_current_price(ticker)
            if current_price is None:
                log.warning(f"  [{ticker}] Prezzo corrente non disponibile — skip.")
                continue
            log.info(f"  [{ticker}] Prezzo corrente: {current_price:.4f}")

            # 3. Conta sessioni aperte
            sessions = count_trading_sessions(opened_at)
            log.info(f"  [{ticker}] Sessioni aperte: {sessions}")

            # 4. Valuta condizione di uscita
            exit_reason = evaluate_exit(pos, current_price, sessions)
            if exit_reason is None:
                log.info(f"  [{ticker}] Nessuna condizione di uscita — posizione ancora aperta.")
                continue

            log.info(f"  [{ticker}] Condizione di uscita: {exit_reason}")

            # 5. Calcola P&L
            pnl_usd, pnl_pct = compute_pnl(direction, entry_price, current_price, size_usd)
            log.info(f"  [{ticker}] P&L: {pnl_usd:+.2f} USD ({pnl_pct*100:+.2f}%)")

            # 6. Chiudi la posizione nel DB
            close_position_in_db(conn, pos_id, current_price, pnl_usd, pnl_pct, exit_reason)
            conn.commit()
            closed_count += 1
            log.info(f"  [{ticker}] Posizione chiusa nel DB (id={pos_id}).")

            # 7. Scrivi outcome reale
            written = write_real_outcome(conn, ticker, opened_at, entry_price, current_price, direction)
            if written:
                conn.commit()
                outcome_count += 1

    finally:
        conn.close()

    log.info("\n" + "=" * 60)
    log.info(f"Posizioni chiuse:  {closed_count}")
    log.info(f"Outcome scritti:   {outcome_count}")
    log.info("=" * 60)


def _print_summary(conn: sqlite3.Connection) -> None:
    """Stampa un riepilogo dello stato attuale del DB."""
    open_count   = conn.execute("SELECT COUNT(id) FROM positions WHERE status = 'OPEN'").fetchone()[0]
    closed_count = conn.execute("SELECT COUNT(id) FROM positions WHERE status = 'CLOSED'").fetchone()[0]
    outcome_rows = conn.execute("SELECT COUNT(id) FROM outcomes").fetchone()[0]
    log.info(f"\nStato DB → posizioni OPEN: {open_count} | CLOSED: {closed_count} | outcomes: {outcome_rows}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_outcome_recorder()
