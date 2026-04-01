"""
market_events.py — Verifica eventi di mercato rilevanti per un ticker (Fase 2)

Usa esclusivamente i dati già prefetchati nello stato LangGraph.
Non effettua chiamate esterne aggiuntive.

Funzioni esportate:
    get_price_change_pct(ticker, prefetched_data)  → float (variazione % giornaliera)
    has_recent_8k(ticker, prefetched_data)         → bool (8-K nelle ultime 24h)
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def get_price_change_pct(ticker: str, prefetched_data: dict) -> float:
    """
    Calcola la variazione percentuale del prezzo di chiusura
    tra l'ultimo giorno e il giorno precedente, usando i dati OHLCV
    già in memoria.

    Ritorna:
        float — variazione % assoluta (es. 2.3 = 2.3%).
        0.0   — se i dati non sono disponibili o insufficienti.
    """
    try:
        ticker_data = prefetched_data.get(ticker, {})
        ohlcv: pd.DataFrame = ticker_data.get("ohlcv_daily")

        if ohlcv is None or ohlcv.empty:
            logger.debug(f"[market_events] Nessun dato OHLCV per {ticker}")
            return 0.0

        # Assicura ordinamento cronologico
        ohlcv = ohlcv.sort_index()

        if len(ohlcv) < 2:
            logger.debug(f"[market_events] Meno di 2 righe OHLCV per {ticker}")
            return 0.0

        # Prendi colonna close (case-insensitive)
        close_col = None
        for col in ohlcv.columns:
            if str(col).lower() == "close":
                close_col = col
                break

        if close_col is None:
            logger.debug(f"[market_events] Colonna 'close' non trovata per {ticker}")
            return 0.0

        last_close = float(ohlcv[close_col].iloc[-1])
        prev_close = float(ohlcv[close_col].iloc[-2])

        if prev_close == 0:
            return 0.0

        pct_change = abs((last_close - prev_close) / prev_close * 100)
        logger.debug(
            f"[market_events] {ticker} price change: "
            f"{prev_close:.2f} → {last_close:.2f} = {pct_change:.2f}%"
        )
        return pct_change

    except Exception as e:
        logger.warning(f"[market_events] Errore calcolo price change per {ticker}: {e}")
        return 0.0


def has_recent_8k(ticker: str, prefetched_data: dict) -> bool:
    """
    Verifica se ci sono stati filing 8-K nelle ultime 24 ore
    per il ticker specificato, usando i dati SEC già fetchati.

    Controlla il campo 'sec_filings' dentro prefetched_data[ticker],
    che è una lista di dict con almeno la chiave 'filing_date' (ISO-8601).

    Ritorna:
        True  — se almeno un 8-K è stato depositato nelle ultime 24 ore.
        False — se nessun 8-K recente o dati non disponibili.
    """
    try:
        ticker_data = prefetched_data.get(ticker, {})
        sec_filings = ticker_data.get("sec_filings", [])

        if not sec_filings:
            logger.debug(f"[market_events] Nessun filing SEC in cache per {ticker}")
            return False

        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)

        for filing in sec_filings:
            # Accetta sia dict che oggetti con attributi
            if isinstance(filing, dict):
                form_type   = str(filing.get("form_type", "")).upper()
                filing_date = filing.get("filing_date", "")
            else:
                form_type   = str(getattr(filing, "form_type", "")).upper()
                filing_date = getattr(filing, "filing_date", "")

            if "8-K" not in form_type:
                continue

            # Parsing data (gestisce vari formati)
            try:
                parsed_date = _parse_date(filing_date)
            except Exception:
                continue

            if parsed_date >= cutoff:
                logger.debug(
                    f"[market_events] 8-K recente trovato per {ticker}: "
                    f"{form_type} del {filing_date}"
                )
                return True

        logger.debug(f"[market_events] Nessun 8-K recente per {ticker}")
        return False

    except Exception as e:
        logger.warning(f"[market_events] Errore controllo 8-K per {ticker}: {e}")
        return False


# ---------------------------------------------------------------------------
# Helper privato
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> datetime:
    """
    Prova a parsare una stringa data in vari formati comuni.
    Ritorna sempre un datetime timezone-aware (UTC).
    """
    if not date_str:
        raise ValueError("Stringa data vuota")

    date_str = str(date_str).strip()

    # Formati da provare in ordine
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",   # ISO con timezone
        "%Y-%m-%dT%H:%M:%S",     # ISO senza timezone
        "%Y-%m-%d %H:%M:%S",     # datetime con spazio
        "%Y-%m-%d",               # solo data
        "%Y%m%d",                 # compatto
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    raise ValueError(f"Impossibile parsare la data: {date_str!r}")
