"""
src/feedback/outcome_tracker.py

Outcome Tracker — Fase 4.2
Legge le predictions senza outcome dal SQLite, calcola il rendimento
effettivo a T+1d, T+5d, T+20d usando yfinance, scrive in 'outcomes',
e stampa le statistiche di performance per agente.

Eseguibile manualmente:
    python -m src.feedback.outcome_tracker

Oppure schedulabile con cron (notte, dopo la chiusura mercati).
"""

import sqlite3
import os
import contextlib
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import yfinance as yf
import pandas as pd

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH = os.path.join("db", "hedge_fund.db")

# Finestre di valutazione (giorni di borsa)
WINDOWS = {
    "1d":  1,
    "5d":  5,
    "20d": 20,
}

# Giorni calendario extra per coprire weekend + festivi
CALENDAR_BUFFER = 35

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Utilità date
# ─────────────────────────────────────────────

def parse_timestamp(ts_str: str) -> datetime:
    """Converte una stringa timestamp ISO 8601 in datetime UTC."""
    ts_str = ts_str.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f+00:00",
        "%Y-%m-%dT%H:%M:%S+00:00",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        raise ValueError(f"Impossibile parsare timestamp: {ts_str!r}")


def _yf_download(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Wrapper attorno a yfinance.download.
    Sopprime i log rumorosi di yfinance (es. "possibly delisted" per date
    future) alzando temporaneamente il livello a CRITICAL sui suoi logger.
    Ritorna None se il DataFrame è vuoto.
    """
    # Silenzia tutti i logger di yfinance durante il download
    yf_loggers = [
        logging.getLogger(name)
        for name in logging.Logger.manager.loggerDict
        if "yfinance" in name
    ]
    # Aggiungi anche il root yfinance logger anche se non ancora registrato
    yf_loggers.append(logging.getLogger("yfinance"))
    original_levels = {lg: lg.level for lg in yf_loggers}
    for lg in yf_loggers:
        lg.setLevel(logging.CRITICAL)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
    except Exception as e:
        log.warning(f"yfinance errore per {ticker}: {e}")
        return None
    finally:
        # Ripristina i livelli originali
        for lg, lvl in original_levels.items():
            lg.setLevel(lvl)

    if df is None or df.empty:
        return None

    # Normalizza MultiIndex (yfinance >= 0.2.x con più ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df if "Close" in df.columns else None


def get_nth_trading_day_close(ticker: str, start_date: datetime, n_trading_days: int) -> Optional[float]:
    """
    Restituisce il prezzo di chiusura dell'N-esimo giorno di borsa
    successivo a start_date. Ritorna None se non ancora disponibile.
    """
    fetch_start = (start_date.date() + timedelta(days=1)).isoformat()
    fetch_end   = (start_date.date() + timedelta(days=CALENDAR_BUFFER)).isoformat()

    df = _yf_download(ticker, fetch_start, fetch_end)
    if df is None:
        return None

    closes = df["Close"].dropna()
    if len(closes) < n_trading_days:
        log.debug(
            f"  {ticker}: {len(closes)} giorni disponibili "
            f"(richiesti {n_trading_days}) — finestra ancora aperta"
        )
        return None

    return float(closes.iloc[n_trading_days - 1])


def get_price_at_date(ticker: str, pred_date: datetime) -> Optional[float]:
    """
    Restituisce il prezzo di chiusura nel giorno di borsa
    più vicino (ma non successivo) a pred_date.
    """
    fetch_start = (pred_date.date() - timedelta(days=5)).isoformat()
    fetch_end   = (pred_date.date() + timedelta(days=1)).isoformat()

    df = _yf_download(ticker, fetch_start, fetch_end)
    if df is None:
        return None

    closes = df["Close"].dropna()
    if closes.empty:
        return None

    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    pred_naive = pd.Timestamp(pred_date.date())
    valid = closes[closes.index <= pred_naive]
    return float(valid.iloc[-1]) if not valid.empty else None


# ─────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────

def load_pending_predictions(conn: sqlite3.Connection) -> list[dict]:
    """
    Ritorna le predictions che non hanno ancora tutte e 3 le finestre calcolate.
    """
    query = """
        SELECT
            p.id          AS prediction_id,
            p.ticker      AS ticker,
            p.signal      AS signal,
            p.confidence  AS confidence,
            p.timestamp   AS timestamp,
            p.agent_id    AS agent_id
        FROM predictions p
        WHERE (
            SELECT COUNT(*)
            FROM outcomes o
            WHERE o.prediction_id = p.id
        ) < 3
        ORDER BY p.timestamp ASC
    """
    rows = conn.execute(query).fetchall()
    return [
        {
            "prediction_id": r[0],
            "ticker":        r[1],
            "signal":        r[2],
            "confidence":    r[3],
            "timestamp":     r[4],
            "agent_id":      r[5],
        }
        for r in rows
    ]


def already_computed_windows(conn: sqlite3.Connection, prediction_id: int) -> set[str]:
    """Finestre già calcolate per questa prediction."""
    rows = conn.execute(
        "SELECT window FROM outcomes WHERE prediction_id = ?",
        (prediction_id,)
    ).fetchall()
    return {r[0] for r in rows}


def compute_return(price_base: float, price_target: float) -> float:
    if price_base == 0:
        return 0.0
    return (price_target - price_base) / price_base


def is_correct(signal: str, actual_return: float) -> bool:
    """BUY → corretto se sale, SELL → corretto se scende, HOLD → sempre neutro."""
    if signal == "BUY":
        return actual_return > 0
    elif signal == "SELL":
        return actual_return < 0
    return True  # HOLD


def write_outcome(
    conn: sqlite3.Connection,
    prediction_id: int,
    ticker: str,
    window: str,
    actual_return: Optional[float],
    evaluated_at: str,
) -> None:
    """Scrive una riga nella tabella outcomes."""
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
        (prediction_id, ticker, r1d, r5d, r20d, window, evaluated_at),
    )


# ─────────────────────────────────────────────
# Performance statistics
# ─────────────────────────────────────────────

def compute_agent_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """Statistiche per agente sulla finestra 1d (esclude HOLD dal hit rate)."""
    query = """
        SELECT p.agent_id, p.ticker, p.signal, o.actual_return_1d
        FROM predictions p
        JOIN outcomes o ON o.prediction_id = p.id
        WHERE o.window = '1d' AND o.actual_return_1d IS NOT NULL
        ORDER BY p.agent_id
    """
    rows = conn.execute(query).fetchall()
    if not rows:
        return pd.DataFrame()

    records = [
        {
            "agent_id": r[0],
            "ticker":   r[1],
            "signal":   r[2],
            "return":   r[3],
            "correct":  is_correct(r[2], r[3]) if r[2] != "HOLD" else None,
        }
        for r in rows
    ]

    df = pd.DataFrame(records)
    df_eval = df[df["signal"] != "HOLD"].copy()
    if df_eval.empty:
        return pd.DataFrame()

    stats = (
        df_eval.groupby("agent_id")
        .apply(lambda g: pd.Series({
            "total":             len(g),
            "hit_rate":          g["correct"].mean(),
            "avg_return_correct": g.loc[g["correct"] == True,  "return"].mean()
                                  if g["correct"].any()   else float("nan"),
            "avg_loss_wrong":    g.loc[g["correct"] == False, "return"].mean()
                                  if (~g["correct"]).any() else float("nan"),
        }))
        .reset_index()
    )
    return stats


# ─────────────────────────────────────────────
# Main job
# ─────────────────────────────────────────────

def run_outcome_tracker() -> None:
    log.info("=" * 60)
    log.info("Outcome Tracker — avvio")
    log.info("=" * 60)

    if not os.path.exists(DB_PATH):
        log.error(f"Database non trovato: {DB_PATH}")
        log.error("Esegui dalla root del progetto athanor-alpha.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    try:
        pending = load_pending_predictions(conn)
        log.info(f"Predictions da elaborare: {len(pending)}")

        if not pending:
            log.info("Nessuna prediction pendente.")
            _print_stats(conn)
            return

        now_utc = datetime.now(timezone.utc).isoformat()
        written_total = 0

        for pred in pending:
            pred_id  = pred["prediction_id"]
            ticker   = pred["ticker"]
            signal   = pred["signal"]
            agent_id = pred["agent_id"]

            try:
                pred_dt = parse_timestamp(pred["timestamp"])
            except ValueError as e:
                log.warning(f"  prediction_id={pred_id}: {e} — saltato")
                continue

            # Salta subito se T+1 non è ancora passato (nessuna finestra è calcolabile)
            now_date = datetime.now(timezone.utc).date()
            if pred_dt.date() >= now_date:
                log.info(f"  Salto prediction_id={pred_id} | {agent_id} | {ticker} | data futura ({pred_dt.date()})")
                continue

            log.info(f"  Elaboro prediction_id={pred_id} | {agent_id} | {ticker} | {signal} | {pred_dt.date()}")

            price_base = get_price_at_date(ticker, pred_dt)
            if price_base is None:
                log.info(f"    Prezzo base non ancora disponibile per {ticker} @ {pred_dt.date()} — skip")
                continue

            done_windows = already_computed_windows(conn, pred_id)
            written_this = 0

            for window_name, n_days in WINDOWS.items():
                if window_name in done_windows:
                    continue

                price_target = get_nth_trading_day_close(ticker, pred_dt, n_days)
                if price_target is None:
                    log.info(f"    {window_name}: dati non ancora disponibili — skip")
                    continue

                ret = compute_return(price_base, price_target)
                correct = is_correct(signal, ret)
                log.info(
                    f"    {window_name}: ret={ret:+.4f} | "
                    f"{'✅ HIT' if correct else '❌ MISS'} "
                    f"(base={price_base:.2f}, target={price_target:.2f})"
                )

                write_outcome(conn, pred_id, ticker, window_name, ret, now_utc)
                written_this += 1

            if written_this > 0:
                conn.commit()
                written_total += written_this

        log.info(f"\n{'='*60}")
        log.info(f"Outcome scritti: {written_total}")
        log.info(f"{'='*60}")
        _print_stats(conn)

    finally:
        conn.close()
        log.info("Connessione DB chiusa.")


def _print_stats(conn: sqlite3.Connection) -> None:
    total_outcomes = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    log.info(f"\nOutcome totali nel DB: {total_outcomes}")

    if total_outcomes == 0:
        log.info("Nessuna statistica disponibile (outcomes vuoto).")
        return

    stats = compute_agent_stats(conn)
    if stats.empty:
        log.info("Nessuna statistica (solo HOLD o dati mancanti).")
        return

    log.info("\n--- Performance per agente (finestra 1d) ---")
    log.info(f"{'Agente':<30} {'Pred':>5} {'HitRate':>8} {'AvgRet+':>9} {'AvgLoss-':>9}")
    log.info("-" * 65)
    for _, row in stats.iterrows():
        hr  = f"{row['hit_rate']:.1%}"             if pd.notna(row["hit_rate"])           else "N/A"
        arc = f"{row['avg_return_correct']:+.2%}"  if pd.notna(row["avg_return_correct"]) else "N/A"
        alw = f"{row['avg_loss_wrong']:+.2%}"      if pd.notna(row["avg_loss_wrong"])      else "N/A"
        log.info(f"  {row['agent_id']:<28} {int(row['total']):>5} {hr:>8} {arc:>9} {alw:>9}")


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_outcome_tracker()
