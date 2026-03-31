"""
Athanor Alpha — Pipeline principale.
Esegue: prefetch → agenti → risk → portfolio → log predictions → email report.

Uso:
    python -m src.run_pipeline                      # tutti i ticker da config/tickers.yaml
    python -m src.run_pipeline AAPL MSFT NVDA       # dry run su 3 ticker
    python -m src.run_pipeline AAPL MSFT --no-email # senza email
"""

import argparse
import json
import logging
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("athanor.pipeline")

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "hedge_fund.db"
TICKERS_CONFIG = ROOT / "config" / "tickers.yaml"


# ---------------------------------------------------------------------------
# Helpers DB
# ---------------------------------------------------------------------------
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _record_run_start(conn, run_id, tickers):
    conn.execute(
        "INSERT INTO pipeline_runs (run_id, started_at, status, tickers) VALUES (?,?,?,?)",
        (run_id, datetime.now(timezone.utc).isoformat(), "running", json.dumps(tickers)),
    )
    conn.commit()


def _record_run_end(conn, run_id, status, error=None):
    conn.execute(
        "UPDATE pipeline_runs SET finished_at=?, status=?, error_msg=? WHERE run_id=?",
        (datetime.now(timezone.utc).isoformat(), status, error, run_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Carica tickers
# ---------------------------------------------------------------------------
def load_tickers(override=None):
    if override:
        log.info(f"Ticker override: {override}")
        return override
    with open(TICKERS_CONFIG, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, list):
        tickers = cfg
    elif isinstance(cfg, dict):
        tickers = cfg.get("tickers", [])
    else:
        raise ValueError(f"Formato tickers.yaml non riconosciuto: {type(cfg)}")
    log.info(f"Ticker da config: {tickers}")
    return tickers


# ---------------------------------------------------------------------------
# Costruisce lo stato iniziale nel formato corretto per il grafo
# ---------------------------------------------------------------------------
def _build_initial_state(tickers: list, run_id: str) -> dict:
    """
    Costruisce l'AgentState iniziale con tutti i campi che gli agenti si aspettano.

    Campi richiesti dagli agenti:
      state["data"]["tickers"]       — lista ticker (philosophy agents)
      state["data"]["end_date"]      — data fine analisi (philosophy agents)
      state["metadata"]["tickers"]   — lista ticker (data_prefetch_agent)
      state["metadata"]["run_id"]    — id run corrente
      state["metadata"]["show_reasoning"] — flag per print reasoning (False = silenzioso)
    """
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return {
        "messages": [],
        "data": {
            # Richiesti dai philosophy agents (ben_graham, buffett, ecc.)
            "tickers": tickers,
            "end_date": end_date,
            "start_date": (datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 1)).strftime("%Y-%m-%d"),
            # Richiesti da prefetch e agenti core
            "prefetched_data": {},
            "analyst_signals": {},
            "portfolio_output": {},
            "risk_output": {},
            # Feedback loop (iniettati da SQLite se disponibili)
            "feedback_history": {},
            "agent_weights": {},
        },
        "metadata": {
            # Richiesti da data_prefetch_agent
            "tickers": tickers,
            # Richiesti da agenti per reasoning display
            "show_reasoning": False,
            # Configurazione LLM — letta da get_agent_model_config come fallback
            "model_name": "claude-sonnet-4-5",
            "model_provider": "Anthropic",
            # Tracking
            "run_id": run_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# Costruisce digest per send_digest(subject, body_text, body_html)
# ---------------------------------------------------------------------------
def _build_digest(final_state, tickers, run_id):
    signals = final_state.get("data", {}).get("analyst_signals", {})
    portfolio_output = final_state.get("data", {}).get("portfolio_recommendations", {})
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    subject = f"Athanor Alpha — Daily Report {now_str}"

    # Aggrega voti per ticker
    ticker_votes = {t: {"BUY": 0, "SELL": 0, "HOLD": 0, "agents": []} for t in tickers}
    for agent_id, agent_data in signals.items():
        if not isinstance(agent_data, dict):
            continue
        for ticker in tickers:
            if ticker not in agent_data:
                continue
            sig = agent_data[ticker]
            # Gli agenti usano segnali diversi: bullish/bearish/neutral o BUY/SELL/HOLD
            raw_signal = str(sig.get("signal", "HOLD")).upper()
            direction = {"BULLISH": "BUY", "BEARISH": "SELL", "NEUTRAL": "HOLD"}.get(raw_signal, raw_signal)
            conf = float(sig.get("confidence", 0))
            if direction in ticker_votes[ticker]:
                ticker_votes[ticker][direction] += 1
            ticker_votes[ticker]["agents"].append(
                f"  {agent_id:35s} {direction:4s} conf={conf:.2f}"
            )

    # Plain text
    lines = [
        "ATHANOR ALPHA — DAILY REPORT",
        f"Run    : {run_id}",
        f"Data   : {now_str}",
        f"Ticker : {', '.join(tickers)}",
        "",
        "=" * 52,
        "SEGNALI AGENTI",
        "=" * 52,
    ]
    for ticker in tickers:
        v = ticker_votes[ticker]
        lines.append(f"\n{ticker}:  BUY={v['BUY']}  SELL={v['SELL']}  HOLD={v['HOLD']}")
        for a in v["agents"]:
            lines.append(a)

    lines += ["", "=" * 52, "RACCOMANDAZIONI PORTFOLIO MANAGER", "=" * 52]
    recs = []
    if isinstance(portfolio_output, dict):
        recs = portfolio_output.get("recommendations", [])
        summary = portfolio_output.get("portfolio_summary", "")
        risk_notes = portfolio_output.get("risk_notes", "")
        if summary:
            lines.append(f"Sintesi: {summary[:200]}")
        if risk_notes:
            lines.append(f"Risk:    {risk_notes[:200]}")
        lines.append("")
    if recs:
        for rec in recs:
            t = rec.get("ticker", "?")
            action = rec.get("action", "?")
            sizing = rec.get("sizing_pct", 0)
            conv = rec.get("conviction", 0)
            reason = str(rec.get("reasoning", ""))[:120]
            lines.append(f"{t}: {action} | size={sizing}% | conviction={conv:.2f}")
            lines.append(f"  {reason}")
    else:
        lines.append("(nessun output portfolio manager — controlla i log)")

    body_text = "\n".join(lines)

    # HTML
    html_rows = ""
    for ticker in tickers:
        v = ticker_votes[ticker]
        html_rows += (
            f"<tr><td><b>{ticker}</b></td>"
            f"<td style='color:green'>BUY {v['BUY']}</td>"
            f"<td style='color:red'>SELL {v['SELL']}</td>"
            f"<td style='color:gray'>HOLD {v['HOLD']}</td></tr>"
        )

    body_html = f"""<html><body style="font-family:monospace;font-size:14px">
<h2>Athanor Alpha — Daily Report</h2>
<p>Run: <code>{run_id}</code> &nbsp;|&nbsp; {now_str}</p>
<h3>Voti agenti per ticker</h3>
<table border="1" cellpadding="6" style="border-collapse:collapse">
  <tr><th>Ticker</th><th>BUY</th><th>SELL</th><th>HOLD</th></tr>
  {html_rows}
</table>
<h3>Dettaglio completo</h3>
<pre style="background:#f5f5f5;padding:12px">{body_text}</pre>
</body></html>"""

    return subject, body_text, body_html


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def run_pipeline(tickers, send_email=True):
    run_id = str(uuid.uuid4())
    log.info("=" * 60)
    log.info(f"ATHANOR ALPHA — RUN {run_id[:8]}")
    log.info(f"Ticker : {tickers}")
    log.info(f"Email  : {'SI' if send_email else 'NO'}")
    log.info("=" * 60)

    conn = _get_conn()
    _record_run_start(conn, run_id, tickers)

    try:
        # ------------------------------------------------------------------
        # 1. Import
        # ------------------------------------------------------------------
        log.info("[1/4] Import moduli...")
        from src.graph.graph import compiled_graph
        from src.feedback.weight_adjuster import adjust_weights
        log.info("[1/4] OK.")

        # ------------------------------------------------------------------
        # 2. Stato iniziale
        # ------------------------------------------------------------------
        log.info("[2/4] Costruzione AgentState...")
        initial_state = _build_initial_state(tickers, run_id)
        log.info(f"[2/4] end_date={initial_state['data']['end_date']}  show_reasoning=False")

        # ------------------------------------------------------------------
        # 3. Esegue il grafo
        # NOTA: il grafo include già PREDICTION_LOG_NODE come ultimo nodo,
        # quindi log_predictions viene chiamato automaticamente.
        # ------------------------------------------------------------------
        log.info("[3/4] Esecuzione grafo (2-5 minuti, attendere)...")
        log.info("      prefetch → agenti paralleli → risk → portfolio → log")
        final_state = compiled_graph.invoke(initial_state)

        signals = final_state.get("data", {}).get("analyst_signals", {})
        log.info(f"[3/4] Grafo completato. Agenti con segnali: {len(signals)}")

        # ------------------------------------------------------------------
        # 4. Aggiorna pesi + email
        # ------------------------------------------------------------------
        log.info("[4/4] Aggiornamento pesi agenti...")
        try:
            weights = adjust_weights()
            log.info(f"[4/4] Pesi aggiornati per {len(weights)} agenti.")
        except Exception as e:
            log.warning(f"[4/4] adjust_weights warning: {e}")

        if send_email:
            log.info("[4/4] Invio email digest...")
            try:
                from src.alerts.email_sender import send_digest
                subject, body_text, body_html = _build_digest(final_state, tickers, run_id)
                ok = send_digest(subject=subject, body_text=body_text, body_html=body_html)
                if ok:
                    log.info("[4/4] Email inviata.")
                else:
                    log.warning("[4/4] send_digest ha restituito False — controlla SMTP.")
            except Exception as e:
                log.warning(f"[4/4] Email fallita (non bloccante): {e}")
        else:
            log.info("[4/4] Email saltata.")

        # ------------------------------------------------------------------
        _record_run_end(conn, run_id, "completed")
        log.info("=" * 60)
        log.info(f"RUN COMPLETATO CON SUCCESSO — run_id={run_id[:8]}")
        log.info("=" * 60)
        conn.close()
        return True

    except Exception as e:
        log.error(f"Pipeline fallita: {e}", exc_info=True)
        _record_run_end(conn, run_id, "failed", str(e))
        conn.close()
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Athanor Alpha — Pipeline principale")
    parser.add_argument("tickers", nargs="*", help="Ticker opzionali (default: da config/tickers.yaml)")
    parser.add_argument("--no-email", action="store_true", help="Disabilita invio email")
    args = parser.parse_args()

    tickers = load_tickers(args.tickers if args.tickers else None)
    if not tickers:
        log.error("Nessun ticker trovato.")
        sys.exit(1)

    success = run_pipeline(tickers, send_email=not args.no_email)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
