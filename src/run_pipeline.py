"""
Athanor Alpha — Pipeline principale.
Esegue: prefetch → agenti → risk → portfolio → log predictions → time exit → email report.

Uso:
    python -m src.run_pipeline                           # tutti i ticker, mode=full
    python -m src.run_pipeline AAPL MSFT NVDA            # dry run su 3 ticker
    python -m src.run_pipeline --mode light              # solo technicals + sentiment
    python -m src.run_pipeline --mode review             # solo portfolio review + time exit
    python -m src.run_pipeline AAPL MSFT --no-email      # senza email
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
# Configurazione modalità
# ---------------------------------------------------------------------------
# Definisce quali nodi analyst sono attivi per ciascun mode.
# Il grafo leggerà state["metadata"]["run_mode"] e salterà i nodi assenti.
MODE_CONFIG = {
    "full": {
        "description": "Pre-Market — tutti gli agenti (13:00 UTC)",
        "analyst_nodes": [
            "warren_buffett", "ben_graham", "charlie_munger",
            "michael_burry", "bill_ackman", "cathie_wood",
            "technicals", "fundamentals", "sentiment", "breakout_momentum",
        ],
        "skip_devils_advocate": False,
        "skip_risk_manager":    False,
        "skip_prediction_log":  False,
        "skip_time_exit":       True,   # Fase 5: non attivo al mattino
    },
    "light": {
        "description": "Mid-Day — technicals + sentiment (17:00 UTC)",
        "analyst_nodes": [
            "technicals", "sentiment", "breakout_momentum",
        ],
        "skip_devils_advocate": False,
        "skip_risk_manager":    True,
        "skip_prediction_log":  True,
        "skip_time_exit":       True,   # Fase 5: non attivo a metà giornata
    },
    "review": {
        "description": "Market Close — solo portfolio review (21:00 UTC)",
        "analyst_nodes": [],
        "skip_devils_advocate": True,
        "skip_risk_manager":    True,
        "skip_prediction_log":  False,
        "skip_time_exit":       False,  # Fase 5: ATTIVO la sera
    },
}

VALID_MODES = list(MODE_CONFIG.keys())


# ---------------------------------------------------------------------------
# Helpers DB
# ---------------------------------------------------------------------------
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _record_run_start(conn, run_id, tickers, mode):
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
# Costruisce lo stato iniziale
# ---------------------------------------------------------------------------
def _load_agent_model_map() -> dict:
    """Carica la mappa agente->modello da agent_router."""
    try:
        from src.llm.agent_router import _load, _AGENT_MAP
        _load()
        return dict(_AGENT_MAP)
    except Exception as e:
        log.warning(f"agent_router non disponibile, uso fallback globale: {e}")
        return {}

def _build_initial_state(tickers: list, run_id: str, mode: str, live: bool = False) -> dict:
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mode_cfg = MODE_CONFIG[mode]

    return {
        "messages": [],
        "data": {
            "tickers": tickers,
            "end_date": end_date,
            "start_date": (datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 1)).strftime("%Y-%m-%d"),
            "prefetched_data": {},
            "analyst_signals": {},
            "portfolio_output": {},
            "risk_output": {},
            "feedback_history": {},
            "agent_weights": {},
        },
        "metadata": {
            "agent_model_map": _load_agent_model_map(),
            "tickers": tickers,
            "show_reasoning": False,
            "model_name": "claude-sonnet-4-6",
            "model_provider": "Anthropic",
            "run_id": run_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            # --- FASE 4: campi per la modalità ---
            "run_mode": mode,
            "active_analyst_nodes": mode_cfg["analyst_nodes"],
            "skip_devils_advocate": mode_cfg["skip_devils_advocate"],
            "skip_risk_manager":    mode_cfg["skip_risk_manager"],
            "skip_prediction_log":  mode_cfg["skip_prediction_log"],
            # --- FASE 5: time-based exit ---
            "skip_time_exit":       mode_cfg["skip_time_exit"],
            # --- FASE 2: Alpaca execution (default OFF — requires --live) ---
            "skip_execution":       not live,
        },
    }


# ---------------------------------------------------------------------------
# Costruisce digest per send_digest(subject, body_text, body_html)
# ---------------------------------------------------------------------------
def _build_digest(final_state, tickers, run_id, mode):
    signals = final_state.get("data", {}).get("analyst_signals", {})
    portfolio_output = final_state.get("data", {}).get("portfolio_recommendations", {})
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mode_label = MODE_CONFIG[mode]["description"]

    subject = f"Athanor Alpha — {mode.upper()} Report {now_str}"

    # Aggrega voti per ticker
    ticker_votes = {t: {"BUY": 0, "SELL": 0, "HOLD": 0, "agents": []} for t in tickers}
    for agent_id, agent_data in signals.items():
        if not isinstance(agent_data, dict):
            continue
        for ticker in tickers:
            if ticker not in agent_data:
                continue
            sig = agent_data[ticker]
            raw_signal = str(sig.get("direction", sig.get("signal", "HOLD"))).upper()
            direction = {"BULLISH": "BUY", "BEARISH": "SELL", "NEUTRAL": "HOLD", "LONG": "BUY", "SHORT": "SELL"}.get(raw_signal, raw_signal)
            conf = float(sig.get("confidence", 0))
            if direction in ticker_votes[ticker]:
                ticker_votes[ticker][direction] += 1
            ticker_votes[ticker]["agents"].append(
                f"  {agent_id:35s} {direction:4s} conf={conf:.2f}"
            )

    # Posizioni aperte (dal portfolio manager output)
    open_positions = portfolio_output.get("open_positions", [])

    # Plain text
    lines = [
        "ATHANOR ALPHA — REPORT",
        f"Modalità: {mode_label}",
        f"Run    : {run_id}",
        f"Data   : {now_str}",
        f"Ticker : {', '.join(tickers)}",
        "",
        "=" * 60,
        "POSIZIONI APERTE",
        "=" * 60,
    ]

    if open_positions:
        lines.append(f"  {'TICKER':<6}  {'DIR':<5}  {'ENTRY':>8}  {'SL':>8}  {'TP':>8}  {'SIZE $':>9}")
        lines.append(f"  {'-'*56}")
        for p in open_positions:
            lines.append(
                f"  {p['ticker']:<6}  {p['direction']:<5}  "
                f"{p['entry_price']:>8.2f}  {p['stop_loss']:>8.2f}  "
                f"{p['take_profit']:>8.2f}  {p['size_usd']:>9,.0f}"
            )
            if p.get("note"):
                lines.append(f"         Nota: {p['note']}")
    else:
        lines.append("  Nessuna posizione aperta — portafoglio in cash.")

    if mode != "review":
        lines += [
            "",
            "=" * 60,
            "SEGNALI AGENTI",
            "=" * 60,
        ]
        for ticker in tickers:
            v = ticker_votes[ticker]
            lines.append(f"\n{ticker}:  BUY={v['BUY']}  SELL={v['SELL']}  HOLD={v['HOLD']}")
            for a in v["agents"]:
                lines.append(a)

    lines += ["", "=" * 60, "RACCOMANDAZIONI PORTFOLIO MANAGER", "=" * 60]
    recs = []
    if isinstance(portfolio_output, dict):
        recs = portfolio_output.get("recommendations", [])
        summary = portfolio_output.get("portfolio_summary", "")
        risk_notes = portfolio_output.get("risk_notes", "")
        if summary:
            lines.append(f"\nSintesi:\n{summary}")
        if risk_notes:
            lines.append(f"\nRisk notes:\n{risk_notes}")
        lines.append("")

    if recs:
        for rec in recs:
            t      = rec.get("ticker", "?")
            action = rec.get("action", "?")
            sizing = rec.get("sizing_pct", 0)
            conv   = rec.get("conviction", 0)
            entry  = rec.get("entry_price")
            sl     = rec.get("stop_loss")
            tp     = rec.get("take_profit")
            size   = rec.get("size_usd")
            rr     = rec.get("rr_ratio")
            reason = str(rec.get("reasoning", ""))

            lines.append(f"\n{'─'*50}")
            lines.append(f"{t}: {action} | size={sizing}% | conviction={conv:.2f} | consensus={rec.get('consensus','?')}")
            if entry is not None:
                lines.append(f"  Entry={entry:.2f}  SL={sl:.2f}  TP={tp:.2f}  Size=${size:,.0f}  R/R=1:{rr:.1f}")
            lines.append(f"  Reasoning: {reason}")
    else:
        lines.append("(nessun output portfolio manager — controlla i log)")

    body_text = "\n".join(lines)

    # ── HTML ──────────────────────────────────────────────────────────────
    html_rows = ""
    for ticker in tickers:
        v = ticker_votes[ticker]
        html_rows += (
            f"<tr><td><b>{ticker}</b></td>"
            f"<td style='color:green'>BUY {v['BUY']}</td>"
            f"<td style='color:red'>SELL {v['SELL']}</td>"
            f"<td style='color:gray'>HOLD {v['HOLD']}</td></tr>"
        )

    if open_positions:
        pos_rows = ""
        for p in open_positions:
            note_str = f"<br><small style='color:#888'>{p['note']}</small>" if p.get("note") else ""
            dir_color = "#27ae60" if p["direction"] == "LONG" else "#c0392b"
            pos_rows += (
                f"<tr>"
                f"<td><b>{p['ticker']}</b></td>"
                f"<td style='color:{dir_color};font-weight:bold'>{p['direction']}</td>"
                f"<td>${p['entry_price']:.2f}</td>"
                f"<td style='color:#c0392b'>${p['stop_loss']:.2f}</td>"
                f"<td style='color:#27ae60'>${p['take_profit']:.2f}</td>"
                f"<td>${p['size_usd']:,.0f}</td>"
                f"<td>{p['opened_at'][:10]}{note_str}</td>"
                f"</tr>"
            )
        positions_html = f"""
<h3>📂 Posizioni Aperte ({len(open_positions)})</h3>
<table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">
  <tr style="background:#f0f0f0">
    <th>Ticker</th><th>Dir</th><th>Entry</th><th>SL</th><th>TP</th><th>Size $</th><th>Aperta il</th>
  </tr>
  {pos_rows}
</table>"""
    else:
        positions_html = "<h3>📂 Posizioni Aperte</h3><p style='color:#888'>Nessuna posizione aperta — portafoglio in cash.</p>"

    agents_section_html = ""
    if mode != "review":
        agents_section_html = f"""
    <h3>📈 Voti Agenti per Ticker</h3>
    <table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">
      <tr style="background:#f0f0f0"><th>Ticker</th><th>BUY</th><th>SELL</th><th>HOLD</th></tr>
      {html_rows}
    </table>"""

    recs_html = ""
    if recs:
        for rec in recs:
            action = rec.get("action", "HOLD")
            color  = {"BUY": "#27ae60", "SELL": "#c0392b"}.get(action, "#888")
            entry  = rec.get("entry_price")
            levels_html = ""
            if entry is not None:
                levels_html = (
                    f"<tr><td>Entry / SL / TP</td>"
                    f"<td>${entry:.2f} / ${rec.get('stop_loss'):.2f} / ${rec.get('take_profit'):.2f}</td></tr>"
                    f"<tr><td>Size / R&R</td>"
                    f"<td>${rec.get('size_usd',0):,.0f} &nbsp; 1:{rec.get('rr_ratio',0):.1f}</td></tr>"
                )
            recs_html += f"""
<div style="border:1px solid #ddd;border-left:4px solid {color};border-radius:4px;padding:12px;margin:10px 0">
  <div style="font-size:16px;font-weight:bold;color:#2c3e50">
    {rec.get('ticker')} &nbsp;
    <span style="color:{color}">{action}</span>
    &nbsp; <span style="font-size:13px;color:#888">{rec.get('consensus','')}</span>
  </div>
  <table style="margin-top:8px;font-size:13px;width:100%">
    <tr><td style="color:#888;width:35%">Conviction</td><td>{rec.get('conviction',0):.2f}</td></tr>
    <tr><td style="color:#888">Sizing</td><td>{rec.get('sizing_pct',0)}%</td></tr>
    {levels_html}
  </table>
  <p style="font-size:13px;margin:8px 0 0;color:#444">{rec.get('reasoning','')}</p>
</div>"""

    # --- DEVIL VETO BLOCK ---
    da_out = final_state.get("data", {}).get("devils_advocate_output", {})
    da_vetoed = da_out.get("vetoed_tickers", [])
    da_reasons = da_out.get("veto_reasons", {})
    da_vix = da_out.get("vix_level", 0.0)
    da_regime = da_out.get("vix_regime", "")

    veto_html = ""
    if da_vetoed:
        veto_cards = ""
        for vt in da_vetoed:
            veto_reason = da_reasons.get(vt, "Vetoed by Devil's Advocate")
            veto_rec = next((r for r in recs if r.get("ticker") == vt), None)
            levels_html_veto = ""
            if veto_rec and veto_rec.get("entry_price") is not None:
                levels_html_veto = (
                    f"<div style='margin-top:8px;font-size:12px;color:#555'>"
                    f"Entry: <b>${veto_rec['entry_price']:.2f}</b> &nbsp;|&nbsp; "
                    f"SL: <b style='color:#c0392b'>${veto_rec['stop_loss']:.2f}</b> &nbsp;|&nbsp; "
                    f"TP: <b style='color:#27ae60'>${veto_rec['take_profit']:.2f}</b>"
                    f"</div>"
                )
            veto_cards += (
                f"<div style='border:1px solid #e67e22;border-left:4px solid #e67e22;border-radius:4px;padding:12px 14px;margin:8px 0;background:#fef5ec;'>"
                f"<div style='font-size:15px;font-weight:bold;color:#d35400'>&#9888; {vt} "
                f"<span style='font-size:12px;font-weight:normal;color:#e67e22'>INGRESSO NON CONFERMATO DAL DEVIL AGENT</span></div>"
                f"<div style='font-size:12px;color:#7f6000;margin-top:6px'><b>Motivazione veto:</b> {veto_reason}</div>"
                f"{levels_html_veto}</div>"
            )
        veto_html = (
            f"<div style='margin:16px 0'><h3 style='color:#d35400;margin-bottom:8px'>&#9888; Devil's Advocate &mdash; Ticker Vetati "
            f"<span style='font-size:12px;font-weight:normal;color:#888'>(VIX={da_vix:.1f} &middot; {da_regime})</span></h3>"
            f"{veto_cards}</div>"
        )

    summary_html = portfolio_output.get("portfolio_summary", "") if isinstance(portfolio_output, dict) else ""
    risk_html    = portfolio_output.get("risk_notes", "")         if isinstance(portfolio_output, dict) else ""

    mode_badge_color = {"full": "#2c3e50", "light": "#2980b9", "review": "#8e44ad"}.get(mode, "#2c3e50")

    body_html = f"""<html><body style="font-family:'Segoe UI',Arial,sans-serif;font-size:14px;background:#f4f4f4;padding:20px">
<div style="max-width:700px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.1)">
  <div style="background:{mode_badge_color};color:#fff;padding:20px 28px">
    <h2 style="margin:0">📊 Athanor Alpha — {mode.upper()} Report</h2>
    <p style="margin:4px 0 0;opacity:.8">{mode_label}</p>
    <p style="margin:4px 0 0;opacity:.8">Run: <code>{run_id}</code> &nbsp;|&nbsp; {now_str}</p>
  </div>
  <div style="padding:24px 28px">

    {positions_html}

    {agents_section_html}

    {veto_html}
    <h3>💼 Raccomandazioni Portfolio Manager</h3>
    {f'<p style="background:#f8f9fa;padding:10px;border-radius:4px">{summary_html}</p>' if summary_html else ''}
    {recs_html}

  </div>
  <div style="background:#f8f9fa;padding:14px 28px;font-size:11px;color:#999;text-align:center">
    Athanor Alpha · {now_str} · Automated report. No trade has been executed.
  </div>
</div>
</body></html>"""

    return subject, body_text, body_html


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------
def run_pipeline(tickers, send_email=False, mode="full", live=False):
    run_id = str(uuid.uuid4())
    mode_label = MODE_CONFIG[mode]["description"]
    log.info("=" * 60)
    log.info(f"ATHANOR ALPHA — RUN {run_id[:8]}")
    log.info(f"Modalità: {mode.upper()} — {mode_label}")
    log.info(f"Ticker : {tickers}")
    log.info(f"Email  : {'SI' if send_email else 'NO'}")
    log.info(f"Live   : {'YES — orders will be submitted to Alpaca' if live else 'NO (paper run)'}")
    log.info("=" * 60)

    conn = _get_conn()
    _record_run_start(conn, run_id, tickers, mode)

    try:
        log.info("[1/4] Import moduli...")
        from src.graph.graph import compiled_graph
        from src.feedback.weight_adjuster import adjust_weights
        log.info("[1/4] OK.")

        log.info("[2/4] Costruzione AgentState...")
        initial_state = _build_initial_state(tickers, run_id, mode, live=live)
        log.info(f"[2/4] end_date={initial_state['data']['end_date']}  mode={mode}  live={live}")

        log.info("[3/4] Esecuzione grafo (2-5 minuti, attendere)...")
        log.info(f"      Nodi analyst attivi: {MODE_CONFIG[mode]['analyst_nodes'] or '(nessuno — solo review)'}")
        final_state = compiled_graph.invoke(initial_state)

        signals = final_state.get("data", {}).get("analyst_signals", {})
        log.info(f"[3/4] Grafo completato. Agenti con segnali: {len(signals)}")

        if mode == "full":
            log.info("[4/4] Aggiornamento pesi agenti...")
            try:
                weights = adjust_weights()
                log.info(f"[4/4] Pesi aggiornati per {len(weights)} agenti.")
            except Exception as e:
                log.warning(f"[4/4] adjust_weights warning: {e}")

        if send_email == "premarket":
            # Build pre-market HTML preview (no send) — useful for local testing
            log.info("[4/4] Building pre-market email preview (--email=premarket)...")
            try:
                from src.alerts.premarket_builder import build_context, build_html, build_text
                _tickers = final_state.get("data", {}).get("tickers", tickers)
                ctx  = build_context(tickers=_tickers)
                html = build_html(ctx)
                import pathlib, datetime as _dt
                out_path = pathlib.Path("logs") / f"premarket_preview_{_dt.date.today()}.html"
                out_path.write_text(html, encoding="utf-8")
                log.info("[4/4] Pre-market HTML written to %s", out_path)
            except Exception as e:
                log.warning("[4/4] Pre-market preview failed (non-blocking): %s", e)
        else:
            log.info("[4/4] Email: daily briefings handled by cron scripts (send_premarket.ps1 / send_postmarket.ps1).")

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
def _run_backtest_single_day(tickers: list, as_of: str) -> bool:
    """
    Single-day backtest debug helper. Reconstructs PIT state at as_of,
    runs the graph (no execution, no emails), prints portfolio_output recommendations.
    """
    from datetime import date as _date
    from src.backtesting.state_reconstructor import reconstruct_state_at
    from src.graph.graph import compiled_graph

    log.info("=" * 60)
    log.info(f"BACKTEST single-day @ {as_of} | tickers={tickers}")
    log.info("=" * 60)
    try:
        state = reconstruct_state_at(as_of, tickers)
        state["metadata"]["skip_execution"]      = True
        state["metadata"]["skip_prediction_log"] = True
        state["metadata"]["skip_time_exit"]      = True
        state["metadata"]["active_analyst_nodes"] = MODE_CONFIG["full"]["analyst_nodes"]
        state["metadata"]["skip_devils_advocate"] = MODE_CONFIG["full"]["skip_devils_advocate"]
        state["metadata"]["skip_risk_manager"]    = MODE_CONFIG["full"]["skip_risk_manager"]
        final = compiled_graph.invoke(state)
        recs = final.get("data", {}).get("portfolio_output", {})
        if isinstance(recs, dict):
            recs = recs.get("recommendations", [])
        log.info(f"BACKTEST output: {len(recs)} recommendations")
        for r in recs:
            log.info(
                "  %-6s %-5s sizing=%s%% conv=%.2f reasoning=%.80s",
                r.get("ticker"), r.get("action"), r.get("sizing_pct"),
                float(r.get("conviction") or 0.0), str(r.get("reasoning", "")),
            )
        return True
    except Exception as exc:
        log.error("Backtest single-day failed: %s", exc, exc_info=True)
        return False


def _health_check() -> dict:
    """
    Compact JSON health check for cron/deploy scripts.
    Returns dict; caller prints JSON and exits 0 (healthy) or 1.
    """
    import os
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db_ok": False,
        "predictions_total": 0,
        "predictions_today": 0,
        "pipeline_runs_24h": 0,
        "kill_switch_armed": False,
        "cb_flags": [],
        "errors": [],
    }
    try:
        conn = _get_conn()
        status["db_ok"] = True
        status["predictions_total"] = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        today = datetime.now(timezone.utc).date().isoformat()
        status["predictions_today"] = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE DATE(timestamp)=?", (today,)
        ).fetchone()[0]
        status["pipeline_runs_24h"] = conn.execute(
            "SELECT COUNT(*) FROM pipeline_runs WHERE started_at > datetime('now','-24 hours')"
        ).fetchone()[0]
        conn.close()
    except Exception as exc:
        status["errors"].append(f"db: {exc}")

    status["kill_switch_armed"] = os.path.exists(".athanor_kill")

    for cb_id in ("cb1", "cb3", "cb5"):
        from pathlib import Path
        flags = list(Path(".").glob(f".circuit_breaker_{cb_id}_*"))
        if flags:
            status["cb_flags"].append(cb_id)

    return status


def main():
    parser = argparse.ArgumentParser(description="Athanor Alpha — Pipeline principale")
    parser.add_argument("tickers", nargs="*", help="Ticker opzionali (default: da config/tickers.yaml)")
    parser.add_argument(
        "--email",
        choices=["premarket"],
        default=None,
        help="Build pre-market email preview (no send). Default: off. Daily emails sent by cron scripts.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Submit orders to Alpaca paper trading (default: OFF — signals only)",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="full",
        help="Modalità di esecuzione: full (default), light, review",
    )
    parser.add_argument(
        "--backtest-as-of",
        type=str,
        default=None,
        help="Single-day backtest debug. Format YYYY-MM-DD. Skips execution and email.",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        default=False,
        help="Print compact JSON health status and exit 0 (healthy) or 1 (issues).",
    )
    args = parser.parse_args()

    if args.health_check:
        hc = _health_check()
        print(json.dumps(hc, indent=2))
        unhealthy = not hc["db_ok"] or hc["kill_switch_armed"] or bool(hc["errors"])
        sys.exit(1 if unhealthy else 0)

    tickers = load_tickers(args.tickers if args.tickers else None)
    if not tickers:
        log.error("Nessun ticker trovato.")
        sys.exit(1)

    if args.backtest_as_of:
        ok = _run_backtest_single_day(tickers, args.backtest_as_of)
        sys.exit(0 if ok else 1)

    success = run_pipeline(tickers, send_email=args.email, mode=args.mode, live=args.live)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
