"""
cache_guard.py – Decide se usare il segnale cached o chiamare il LLM (Fase 2)

Logica: salta il LLM se TUTTE e tre queste condizioni sono vere:
    1. Esiste un segnale in cache per (agent_id, ticker) generato OGGI (data UTC)
    2. Variazione prezzo nelle ultime 24h < PRICE_CHANGE_THRESHOLD
    3. Nessun filing 8-K nelle ultime 24h

Fix rispetto alla versione precedente:
    - La cache è invalidata se il segnale è stato generato in un giorno di
      calendario diverso da oggi (UTC), anche se sono passate meno di 24 ore.
      Questo impedisce che un segnale di ieri sera venga riutilizzato oggi.
    - Aggiunto flag FORCE_REFRESH (env var ATHANOR_FORCE_REFRESH=1) per
      bypassare completamente la cache nei test manuali.

Funzione esportata:
    should_use_cache(agent_id, ticker, state) → bool
"""

import logging
import os
from datetime import datetime, timezone

from src.data.signal_cache import get_cached_signal
from src.data.market_events import get_price_change_pct, has_recent_8k

logger = logging.getLogger(__name__)

# Soglie configurabili
PRICE_CHANGE_THRESHOLD = 1.5   # % – sopra questa soglia invalida la cache
CACHE_MAX_AGE_HOURS    = 24.0  # ore – età massima assoluta del segnale cached

# Agenti esclusi dalla cache (aggregatori che devono sempre ricalcolare)
CACHE_EXCLUDED_AGENTS = {
    "portfolio_manager",
    "portfolio_manager_agent",
    "risk_manager",
    "risk_manager_agent",
}

# Flag di override: impostare ATHANOR_FORCE_REFRESH=1 per bypassare la cache
FORCE_REFRESH = os.environ.get("ATHANOR_FORCE_REFRESH", "0").strip() == "1"


def should_use_cache(agent_id: str, ticker: str, state: dict) -> bool:
    """
    Restituisce True se il segnale cached è valido e il LLM può essere saltato.
    Restituisce False se bisogna chiamare il LLM.

    Parametri:
        agent_id – es. "warren_buffett_agent"
        ticker   – es. "AAPL"
        state    – AgentState LangGraph (deve contenere state["data"]["prefetched_data"])
    """
    # --- Override forzato: bypassa tutta la cache ---
    if FORCE_REFRESH:
        logger.info(f"[cache_guard] FORCE_REFRESH attivo — LLM forzato per {agent_id}/{ticker}")
        return False

    # --- Agenti esclusi dalla cache ---
    if agent_id in CACHE_EXCLUDED_AGENTS:
        logger.debug(f"[cache_guard] EXCLUDED — {agent_id} non usa cache")
        return False

    # --- Controllo 1: esiste un segnale in cache valido (< 24h)? ---
    cached = get_cached_signal(agent_id, ticker, max_age_hours=CACHE_MAX_AGE_HOURS)
    if cached is None:
        logger.debug(f"[cache_guard] MISS (no cache) — {agent_id}/{ticker}")
        return False

    # --- Controllo 1b: il segnale è stato generato OGGI (data UTC)? ---
    # Anche se è dentro le 24h, un segnale di ieri non è valido per oggi.
    if not _is_same_day_utc(cached["created_at"]):
        logger.info(
            f"[cache_guard] MISS (giorno diverso) — {agent_id}/{ticker} "
            f"(cached il {cached['created_at'][:10]}, oggi è {_today_utc()})"
        )
        return False

    # --- Estrai prefetched_data dallo stato ---
    prefetched_data = _get_prefetched_data(state)
    if prefetched_data is None:
        logger.debug(f"[cache_guard] MISS (no prefetched_data) — {agent_id}/{ticker}")
        return False

    # --- Controllo 2: variazione prezzo < soglia? ---
    price_change = get_price_change_pct(ticker, prefetched_data)
    if price_change >= PRICE_CHANGE_THRESHOLD:
        logger.info(
            f"[cache_guard] MISS (price move {price_change:.2f}% >= "
            f"{PRICE_CHANGE_THRESHOLD}%) — {agent_id}/{ticker}"
        )
        return False

    # --- Controllo 3: nessun 8-K recente? ---
    if has_recent_8k(ticker, prefetched_data):
        logger.info(f"[cache_guard] MISS (8-K recente) — {agent_id}/{ticker}")
        return False

    # --- Tutti i controlli passati: usa la cache ---
    logger.info(
        f"[cache_guard] HIT — salto LLM per {agent_id}/{ticker} "
        f"(signal={cached['signal']}, conf={cached['confidence']:.0f}%, "
        f"price_change={price_change:.2f}%)"
    )
    return True


def get_cached_signal_safe(agent_id: str, ticker: str) -> dict:
    """
    Restituisce il segnale cached (chiamare solo dopo should_use_cache=True).
    Garantisce sempre un dict valido anche in caso di errore imprevisto.
    """
    cached = get_cached_signal(agent_id, ticker, max_age_hours=CACHE_MAX_AGE_HOURS)
    if cached is not None:
        return cached
    # Fallback di sicurezza (non dovrebbe mai arrivare qui)
    return {
        "signal":     "neutral",
        "confidence": 0.0,
        "reasoning":  "[cache_guard] Fallback: segnale cached non disponibile.",
        "created_at": "",
    }


# ---------------------------------------------------------------------------
# Helper privati
# ---------------------------------------------------------------------------

def _today_utc() -> str:
    """Restituisce la data odierna in formato YYYY-MM-DD (UTC)."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _is_same_day_utc(created_at_iso: str) -> bool:
    """
    Verifica se il timestamp ISO corrisponde al giorno odierno (UTC).
    Restituisce False in caso di errore di parsing (cache invalidata per sicurezza).
    """
    try:
        created_at = datetime.fromisoformat(created_at_iso)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return created_at.strftime("%Y-%m-%d") == _today_utc()
    except Exception as e:
        logger.warning(f"[cache_guard] Errore parsing created_at '{created_at_iso}': {e}")
        return False  # invalida la cache per sicurezza


def _get_prefetched_data(state: dict) -> dict | None:
    """
    Estrae prefetched_data dall'AgentState LangGraph.
    Gestisce sia la struttura normale che eventuali varianti.
    """
    try:
        # Struttura standard: state["data"]["prefetched_data"]
        data = state.get("data", {})
        if isinstance(data, dict):
            prefetched = data.get("prefetched_data")
            if isinstance(prefetched, dict) and prefetched:
                return prefetched

        # Fallback: state["prefetched_data"] (struttura flat)
        prefetched = state.get("prefetched_data")
        if isinstance(prefetched, dict) and prefetched:
            return prefetched

        return None

    except Exception as e:
        logger.warning(f"[cache_guard] Errore estrazione prefetched_data: {e}")
        return None
