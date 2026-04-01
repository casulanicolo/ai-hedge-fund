"""
signal_cache.py — Cache segnali agente/ticker su SQLite (Fase 2)

Tabella: signal_cache
Colonne: agent_id, ticker, signal, confidence, reasoning, created_at
"""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Percorso al database (relativo alla root del progetto)
_DB_PATH = Path(__file__).resolve().parents[2] / "db" / "hedge_fund.db"


def _get_connection() -> sqlite3.Connection:
    """Apre una connessione al database SQLite."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_table_exists() -> None:
    """
    Crea la tabella signal_cache se non esiste ancora.
    Chiamata automaticamente dalle altre funzioni — non serve invocarla a mano.
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS signal_cache (
        agent_id    TEXT    NOT NULL,
        ticker      TEXT    NOT NULL,
        signal      TEXT    NOT NULL,
        confidence  REAL    NOT NULL DEFAULT 0.0,
        reasoning   TEXT    NOT NULL DEFAULT '',
        created_at  TEXT    NOT NULL,
        PRIMARY KEY (agent_id, ticker)
    );
    """
    try:
        with _get_connection() as conn:
            conn.execute(create_sql)
            conn.commit()
    except Exception as e:
        logger.error(f"[signal_cache] Errore creazione tabella: {e}")


def get_cached_signal(
    agent_id: str,
    ticker: str,
    max_age_hours: float = 24.0,
) -> dict | None:
    """
    Restituisce il segnale cached per (agent_id, ticker) se esiste
    ed è stato generato nelle ultime `max_age_hours` ore.

    Ritorna:
        dict con chiavi {signal, confidence, reasoning, created_at}
        oppure None se non trovato o scaduto.
    """
    ensure_table_exists()

    select_sql = """
    SELECT signal, confidence, reasoning, created_at
    FROM   signal_cache
    WHERE  agent_id = ? AND ticker = ?
    LIMIT  1;
    """
    try:
        with _get_connection() as conn:
            row = conn.execute(select_sql, (agent_id, ticker)).fetchone()

        if row is None:
            return None

        # Verifica età della cache
        created_at = datetime.fromisoformat(row["created_at"])
        # Assicura che created_at sia timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        now = datetime.now(tz=timezone.utc)
        age_hours = (now - created_at).total_seconds() / 3600.0

        if age_hours > max_age_hours:
            logger.debug(
                f"[signal_cache] EXPIRED ({age_hours:.1f}h > {max_age_hours}h) "
                f"— {agent_id}/{ticker}"
            )
            return None

        logger.debug(
            f"[signal_cache] HIT ({age_hours:.1f}h old) — {agent_id}/{ticker} "
            f"→ {row['signal']} ({row['confidence']:.0f}%)"
        )
        return {
            "signal":     row["signal"],
            "confidence": row["confidence"],
            "reasoning":  row["reasoning"],
            "created_at": row["created_at"],
        }

    except Exception as e:
        logger.warning(f"[signal_cache] Errore lettura cache {agent_id}/{ticker}: {e}")
        return None


def save_signal(
    agent_id: str,
    ticker: str,
    signal: str,
    confidence: float,
    reasoning: str,
) -> None:
    """
    Salva (o aggiorna) il segnale per (agent_id, ticker).
    Usa INSERT OR REPLACE per sovrascrivere record esistenti.
    """
    ensure_table_exists()

    upsert_sql = """
    INSERT OR REPLACE INTO signal_cache
        (agent_id, ticker, signal, confidence, reasoning, created_at)
    VALUES
        (?, ?, ?, ?, ?, ?);
    """
    created_at = datetime.now(tz=timezone.utc).isoformat()

    try:
        with _get_connection() as conn:
            conn.execute(
                upsert_sql,
                (agent_id, ticker, signal, confidence, reasoning, created_at),
            )
            conn.commit()
        logger.debug(
            f"[signal_cache] SAVED — {agent_id}/{ticker} → {signal} ({confidence:.0f}%)"
        )
    except Exception as e:
        logger.warning(f"[signal_cache] Errore salvataggio {agent_id}/{ticker}: {e}")
