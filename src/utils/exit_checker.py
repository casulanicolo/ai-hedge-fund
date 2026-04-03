"""
exit_checker.py
---------------
Utility per la Time-Based Exit Rule (Fase 5).

Funzioni esportate:
  - count_market_sessions(entry_date_str) -> int
  - check_time_based_exits(positions) -> list[dict]
"""

import pandas_market_calendars as mcal
import pandas as pd
from datetime import date, datetime


# Soglia: numero di sessioni di mercato prima di inviare l'alert
SESSION_THRESHOLD = 4

# Calendario NYSE (New York Stock Exchange)
NYSE = mcal.get_calendar("NYSE")


def count_market_sessions(entry_date_str: str) -> int:
    """
    Conta quante sessioni di mercato NYSE sono passate
    dalla data di ingresso (inclusa) a oggi (esclusa).

    Args:
        entry_date_str: data in formato 'YYYY-MM-DD'

    Returns:
        numero intero di sessioni aperte trascorse
    """
    try:
        entry_date = pd.Timestamp(entry_date_str)
    except Exception:
        # Se la data non è parsabile, restituiamo 0 per sicurezza
        return 0

    today = pd.Timestamp(date.today())

    # Se la data di ingresso è oggi o futura, 0 sessioni passate
    if entry_date >= today:
        return 0

    # get_valid_days restituisce tutti i giorni in cui NYSE era aperto
    schedule = NYSE.valid_days(start_date=entry_date, end_date=today - pd.Timedelta(days=1))

    return len(schedule)


def check_time_based_exits(positions: list) -> list:
    """
    Scorre la lista delle posizioni aperte e restituisce
    quelle che soddisfano entrambe le condizioni:
      1. sessioni aperte >= SESSION_THRESHOLD
      2. alert_sent != True

    Args:
        positions: lista di dict, come da config/positions.json

    Returns:
        lista di dict delle posizioni da alertare (sottoinsieme di positions)
    """
    to_alert = []

    for pos in positions:
        # Salta se l'alert è già stato inviato
        if pos.get("alert_sent", False):
            continue

        entry_date = pos.get("entry_date")
        if not entry_date:
            # Posizione senza data di ingresso: non possiamo calcolare, saltiamo
            continue

        sessions = count_market_sessions(entry_date)

        if sessions >= SESSION_THRESHOLD:
            # Aggiungiamo il conteggio al dict per usarlo nel messaggio email
            pos_with_count = dict(pos)
            pos_with_count["sessions_open"] = sessions
            to_alert.append(pos_with_count)

    return to_alert
