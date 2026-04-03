"""
time_exit_agent.py
------------------
Agente LangGraph per la Time-Based Exit Rule (Fase 5).

Logica:
  1. Carica le posizioni aperte da config/positions.json
  2. Chiama check_time_based_exits() per trovare posizioni >= 4 sessioni
  3. Se ne trova, compone e invia un'email di alert via SMTP
  4. Aggiorna alert_sent=True nel file positions.json per evitare duplicati

Viene eseguito solo nel modo 'review' (slot 21:00 UTC).
"""

import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from dotenv import load_dotenv

from src.utils.exit_checker import check_time_based_exits

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Percorso al file delle posizioni
POSITIONS_PATH = Path(__file__).resolve().parents[2] / "config" / "positions.json"


def _load_positions() -> list:
    """Carica la lista delle posizioni da positions.json."""
    if not POSITIONS_PATH.exists():
        return []
    with open(POSITIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("positions", [])


def _save_positions(positions: list) -> None:
    """Salva la lista delle posizioni aggiornata in positions.json."""
    with open(POSITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump({"positions": positions}, f, indent=2, ensure_ascii=False)


def _build_email_html(to_alert: list) -> str:
    """Compone il corpo HTML dell'email di alert."""
    rows = ""
    for pos in to_alert:
        ticker = pos.get("ticker", "N/A")
        direction = pos.get("direction", "N/A")
        entry_date = pos.get("entry_date", "N/A")
        entry_price = pos.get("entry_price", "N/A")
        target_price = pos.get("target_price", "N/A")
        stop_loss = pos.get("stop_loss", "N/A")
        sessions = pos.get("sessions_open", "N/A")

        rows += f"""
        <tr>
          <td style="padding:8px;border:1px solid #ddd;font-weight:bold;">{ticker}</td>
          <td style="padding:8px;border:1px solid #ddd;">{direction}</td>
          <td style="padding:8px;border:1px solid #ddd;">{entry_date}</td>
          <td style="padding:8px;border:1px solid #ddd;">{entry_price}</td>
          <td style="padding:8px;border:1px solid #ddd;">{target_price}</td>
          <td style="padding:8px;border:1px solid #ddd;">{stop_loss}</td>
          <td style="padding:8px;border:1px solid #ddd;color:#c0392b;font-weight:bold;">{sessions}</td>
        </tr>
        """

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;color:#222;">
      <h2 style="color:#c0392b;">&#9888; Athanor Alpha &mdash; Time-Based Exit Alert</h2>
      <p>Le seguenti posizioni sono aperte da <strong>4 o piu' sessioni di mercato</strong>
         senza aver raggiunto il target price ne' lo stop loss.</p>
      <p>Valuta manualmente se chiudere o mantenere la posizione.</p>

      <table style="border-collapse:collapse;width:100%;margin-top:16px;">
        <thead>
          <tr style="background:#2c3e50;color:white;">
            <th style="padding:8px;border:1px solid #ddd;">Ticker</th>
            <th style="padding:8px;border:1px solid #ddd;">Direzione</th>
            <th style="padding:8px;border:1px solid #ddd;">Data Ingresso</th>
            <th style="padding:8px;border:1px solid #ddd;">Prezzo Ingresso</th>
            <th style="padding:8px;border:1px solid #ddd;">Target</th>
            <th style="padding:8px;border:1px solid #ddd;">Stop Loss</th>
            <th style="padding:8px;border:1px solid #ddd;">Sessioni Aperte</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>

      <p style="margin-top:24px;color:#7f8c8d;font-size:12px;">
        Questo alert viene inviato una sola volta per posizione.<br>
        Athanor Alpha &mdash; Monitor automatico posizioni
      </p>
    </body>
    </html>
    """
    return html


def _send_email(to_alert: list) -> bool:
    """
    Invia l'email di alert via SMTP.

    Returns:
        True se l'invio e' riuscito, False altrimenti.
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    recipient = os.getenv("ALERT_RECIPIENT")

    if not all([smtp_user, smtp_password, recipient]):
        print("[time_exit_agent] ERRORE: credenziali SMTP mancanti nel file .env")
        return False

    tickers = ", ".join(pos.get("ticker", "?") for pos in to_alert)
    subject = f"[Athanor Alpha] Time-Based Exit Alert: {tickers}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient

    html_body = _build_email_html(to_alert)
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, msg.as_string())
        print(f"[time_exit_agent] Email inviata per: {tickers}")
        return True
    except Exception as e:
        print(f"[time_exit_agent] ERRORE invio email: {e}")
        return False


def time_exit_agent(state: dict) -> dict:
    """
    Nodo LangGraph per la Time-Based Exit Rule.

    Viene chiamato dal grafo nel modo 'review'.
    Non modifica lo state LangGraph: opera direttamente su positions.json.

    Args:
        state: stato LangGraph (non modificato da questo agente)

    Returns:
        state invariato
    """
    print("[time_exit_agent] Avvio controllo time-based exit...")

    positions = _load_positions()

    if not positions:
        print("[time_exit_agent] Nessuna posizione aperta. Nulla da controllare.")
        return state

    to_alert = check_time_based_exits(positions)

    if not to_alert:
        print("[time_exit_agent] Nessuna posizione supera la soglia di 4 sessioni.")
        return state

    print(f"[time_exit_agent] {len(to_alert)} posizione/i da alertare.")

    email_sent = _send_email(to_alert)

    if email_sent:
        # Aggiorna alert_sent=True solo per le posizioni alertate
        alerted_tickers = {pos.get("ticker") for pos in to_alert}
        for pos in positions:
            if pos.get("ticker") in alerted_tickers:
                pos["alert_sent"] = True
        _save_positions(positions)
        print("[time_exit_agent] positions.json aggiornato con alert_sent=True.")

    return state
