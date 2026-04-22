import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Athanor Alpha — Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "hedge_fund.db")

AGENT_LABELS = {
    "ben_graham_agent": "Graham",
    "bill_ackman_agent": "Ackman",
    "cathie_wood_agent": "Wood",
    "charlie_munger_agent": "Munger",
    "fundamentals_analyst_agent": "Fundamentals",
    "michael_burry_agent": "Burry",
    "sentiment_agent": "Sentiment",
    "technical_analyst_agent": "Technical",
    "warren_buffett_agent": "Buffett",
    "breakout_momentum": "Breakout",
    "aswath_damodaran_agent": "Damodaran",
}

SIGNAL_COLORS = {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#78909c"}

ACTION_BG = {
    "BUY":  "#d4f4e2",
    "SELL": "#fde0e0",
    "HOLD": "#f5f5f5",
}

# ── DB helper ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect(DB_PATH)

    predictions = pd.read_sql("""
        SELECT id, run_id, agent_id, ticker, signal, confidence,
               timestamp, entry_price, stop_loss, take_profit
        FROM predictions
        ORDER BY timestamp DESC
    """, conn)

    outcomes = pd.read_sql("""
        SELECT prediction_id, ticker, actual_return_1d,
               actual_return_5d, actual_return_20d, window, evaluated_at
        FROM outcomes
    """, conn)

    agent_weights = pd.read_sql("""
        SELECT agent_id, ticker, weight, updated_at
        FROM agent_weights
        ORDER BY updated_at ASC
    """, conn)

    pipeline_runs = pd.read_sql("""
        SELECT run_id, started_at, finished_at, status, tickers, error_msg
        FROM pipeline_runs
        ORDER BY started_at DESC
    """, conn)

    positions = pd.read_sql("""
        SELECT * FROM positions ORDER BY opened_at DESC
    """, conn)

    signal_cache = pd.read_sql("""
        SELECT agent_id, ticker, signal, confidence, created_at
        FROM signal_cache
        ORDER BY created_at DESC
    """, conn)

    conn.close()

    # ── Parsing timestamps ──
    for df, col in [(predictions, "timestamp"), (agent_weights, "updated_at"),
                    (pipeline_runs, "started_at"), (signal_cache, "created_at")]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    predictions["date"] = predictions["timestamp"].dt.date
    predictions["agent_label"] = predictions["agent_id"].map(AGENT_LABELS).fillna(
        predictions["agent_id"]
    )

    # ── Join predictions + outcomes ──
    out_1d = outcomes[outcomes["window"] == "1d"][["prediction_id", "actual_return_1d"]].copy()
    out_5d = outcomes[outcomes["window"] == "5d"][["prediction_id", "actual_return_5d"]].copy()
    out_20d = outcomes[outcomes["window"] == "20d"][["prediction_id", "actual_return_20d"]].copy()

    if out_1d.empty:
        out_1d = outcomes[["prediction_id", "actual_return_1d"]].dropna(
            subset=["actual_return_1d"]
        ).drop_duplicates("prediction_id")
    if out_5d.empty:
        out_5d = outcomes[["prediction_id", "actual_return_5d"]].dropna(
            subset=["actual_return_5d"]
        ).drop_duplicates("prediction_id")
    if out_20d.empty:
        out_20d = outcomes[["prediction_id", "actual_return_20d"]].dropna(
            subset=["actual_return_20d"]
        ).drop_duplicates("prediction_id")

    merged = predictions.merge(out_1d, left_on="id", right_on="prediction_id", how="left")
    merged = merged.merge(out_5d, left_on="id", right_on="prediction_id", how="left")
    merged = merged.merge(out_20d, left_on="id", right_on="prediction_id", how="left")

    return merged, agent_weights, pipeline_runs, positions, signal_cache


@st.cache_data(ttl=60)
def load_portfolio_decisions():
    """Carica la tabella portfolio_decisions dal DB."""
    conn = sqlite3.connect(DB_PATH)
    try:
        dec = pd.read_sql("""
            SELECT
                id, run_id, timestamp, ticker, action,
                net_score, avg_confidence, weighted_conviction, conviction,
                sizing_pct, consensus,
                entry_price, stop_loss, take_profit, size_usd, rr_ratio,
                info_entry, info_sl, info_tp, info_size_usd, info_rr_ratio, info_direction,
                devil_vetoed, reasoning
            FROM portfolio_decisions
            ORDER BY timestamp DESC
        """, conn)
    except Exception:
        dec = pd.DataFrame()
    conn.close()

    if not dec.empty:
        dec["timestamp"] = pd.to_datetime(dec["timestamp"], utc=True, errors="coerce")
        dec["date"] = dec["timestamp"].dt.date
        dec["devil_vetoed"] = dec["devil_vetoed"].astype(bool)

    return dec


# ── Leggi query params ────────────────────────────────────────────────────────
query_params = st.query_params
ticker_from_url = query_params.get("ticker", None)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/fire-element.png", width=60)
st.sidebar.title("Athanor Alpha")
st.sidebar.caption("Multi-Agent Trading Dashboard")

PAGES = [
    "🏠 Overview",
    "📋 Segnali recenti",
    "📊 Decisioni Portfolio",
    "📈 Performance",
    "⚖️ Pesi agenti",
    "🔁 Pipeline runs",
]

# Se arriva un ?ticker= dall'email, pre-seleziona "Segnali recenti"
default_page_idx = 0
if ticker_from_url:
    default_page_idx = PAGES.index("📋 Segnali recenti")

page = st.sidebar.radio("Pannello", PAGES, index=default_page_idx)

st.sidebar.divider()
st.sidebar.caption(f"DB: `{DB_PATH}`")
if st.sidebar.button("🔄 Refresh dati"):
    st.cache_data.clear()
    st.rerun()

# ── Load ─────────────────────────────────────────────────────────────────────
try:
    df, weights, runs, positions, cache = load_data()
    decisions = load_portfolio_decisions()
except Exception as e:
    st.error(f"Errore connessione DB: {e}")
    st.stop()

# ── Helper metrics ────────────────────────────────────────────────────────────
def win_rate(series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return (s > 0).sum() / len(s)

def pnl_cumulative(sub):
    s = sub["actual_return_5d"].dropna()
    return (s * 0.01).cumsum()


# ── Radar Chart ──────────────────────────────────────────────────────────────
def plot_dimension_radar(ticker_name: str, current_run_id: str):
    """
    Legge logs/runs.jsonl, cerca il run_id corrispondente e disegna un
    Radar Chart delle 4 dimensioni (FUNDAMENTALS, TECHNICAL, SENTIMENT, MACRO).

    Robustezze applicate:
    - Path calcolato con Path(__file__).resolve() → infallibile anche su VPS
    - Errori espliciti nella UI Streamlit (st.error / st.warning / st.info)
    - run_id normalizzato a str + strip() per evitare type mismatch
    - Dimensioni mancanti (es. MACRO) escluse dalla media invece di essere
      forzate a 0.0, per non distorcere colori e logica
    """
    # ── 1. Path infallibile basato sul file reale su disco ───────────────────
    log_path = Path(__file__).resolve().parent.parent / "logs" / "runs.jsonl"

    # ── 2. Validazione anticipata con feedback visibile nella UI ─────────────
    if not log_path.exists():
        st.error(
            f"⚠️ Radar Chart: file di log non trovato.\n"
            f"Percorso cercato: `{log_path}`\n"
            f"Assicurati che la pipeline scriva in `logs/runs.jsonl` "
            f"e che la dashboard venga avviata dalla root del progetto."
        )
        return go.Figure()

    # ── 3. Normalizza run_id: strip + cast a str per match sicuro ────────────
    target_run_id = str(current_run_id).strip()

    dim_scores: dict = {}
    parse_error: str | None = None

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:          # salta righe vuote
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                parse_error = f"Riga JSON malformata ignorata: {e}"
                continue

            if str(data.get("run_id", "")).strip() == target_run_id:
                dim_scores = (
                    data
                    .get("per_ticker", {})
                    .get(ticker_name, {})
                    .get("dim_scores", {})
                )
                break  # trovato: esci subito

    except PermissionError:
        st.error(f"❌ Radar Chart: permesso negato su `{log_path}`.")
        return go.Figure()
    except OSError as e:
        st.error(f"❌ Radar Chart: errore I/O — {e}")
        return go.Figure()

    if parse_error:
        st.warning(f"⚠️ Radar Chart (avviso parsing): {parse_error}")

    if not dim_scores:
        st.info(
            f"ℹ️ Nessun `dim_scores` trovato per **{ticker_name}** "
            f"nel run `{target_run_id[:8]}…`  "
            f"(il ticker potrebbe non essere presente in questo run, "
            f"o il run_id non corrisponde a nessuna riga del log)."
        )
        return go.Figure()

    # ── 4. Dimensioni: usa solo quelle presenti nel JSON ────────────────────
    #    MACRO può mancare → non la forziamo a 0, ma la escludiamo dalla media
    DIMS = ["FUNDAMENTALS", "TECHNICAL", "SENTIMENT", "MACRO"]
    labels_present = [d.capitalize() for d in DIMS if d in dim_scores]
    values_present = [float(dim_scores[d]) for d in DIMS if d in dim_scores]

    if not values_present:
        st.warning(f"⚠️ `dim_scores` trovato per {ticker_name} ma è vuoto.")
        return go.Figure()

    # Chiudi il poligono radar (primo punto ripetuto alla fine)
    theta = labels_present + [labels_present[0]]
    r     = values_present + [values_present[0]]

    # Media solo sulle dimensioni presenti (non sui missing)
    avg_score  = sum(values_present) / len(values_present)
    line_color = "#26a69a" if avg_score >= 0 else "#ef5350"
    fill_color = "rgba(38,166,154,0.3)" if avg_score >= 0 else "rgba(239,83,80,0.3)"

    missing  = [d for d in DIMS if d not in dim_scores]
    subtitle = (
        f"  <sub>({len(values_present)}/4 dim — mancanti: {', '.join(missing)})</sub>"
        if missing else ""
    )

    fig = go.Figure(data=go.Scatterpolar(
        r=r, theta=theta, fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        showlegend=False,
        margin=dict(l=30, r=30, t=50, b=30),
        height=300,
        title=dict(text=f"Forza Dimensioni: {ticker_name}{subtitle}", font=dict(size=14)),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🔥 Athanor Alpha — Overview")

    last_run = runs.iloc[0] if not runs.empty else None
    last_run_time = last_run["started_at"].strftime("%d %b %Y %H:%M UTC") if last_run is not None else "N/A"
    last_run_status = last_run["status"] if last_run is not None else "N/A"

    total_preds = len(df)
    evaluated = df["actual_return_5d"].notna().sum()
    wr = win_rate(df[df["signal"].isin(["BUY", "SELL"])]["actual_return_5d"])
    n_agents = df["agent_id"].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Predizioni totali", f"{total_preds:,}")
    c2.metric("Valutate (5d)", f"{evaluated:,}")
    c3.metric("Win rate (5d)", f"{wr*100:.1f}%" if wr else "N/A")
    c4.metric("Agenti attivi", n_agents)
    c5.metric(
        "Ultimo run",
        last_run_time,
        delta=last_run_status,
        delta_color="normal" if last_run_status == "success" else "inverse",
    )

    # === INIZIO NUOVA SEZIONE: BRIEFING E MACRO ===
    st.markdown("### 🌍 Contesto di Mercato (Macro)")
    # (Nota: qui imposto CAUTION come default, ma potrai collegarlo al tuo DB se salvi il regime)
    macro_regime_attuale = "CAUTION"

    if macro_regime_attuale == "RISK_OFF":
        st.error("🚨 **ALLARME MACRO (RISK OFF):** Il mercato è estremamente instabile (es. VIX alto). L'IA ha **drasticamente ridotto** la size delle posizioni al 5%. Massima prudenza.")
    elif macro_regime_attuale == "CAUTION":
        st.warning("⚠️ **MERCATO INCERTO (CAUTION):** Il contesto macroeconomico presenta incertezze. Limite conservativo del 12% per posizione.")
    elif macro_regime_attuale == "RISK_ON":
        st.success("🟢 **MERCATO FAVOREVOLE (RISK ON):** Condizioni macro stabili. Il sistema permette esposizioni fino al 20% per trade.")

    st.markdown("---")
    st.subheader("🎯 Cosa fare oggi (Daily Briefing)")

    if not decisions.empty:
        # Prende l'ID dell'ultimo run disponibile
        latest_run_id = decisions.iloc[0]["run_id"]
        latest_decisions = decisions[decisions["run_id"] == latest_run_id]
        # Prende i top 3 BUY
        df_buys = latest_decisions[latest_decisions['action'] == 'BUY'].head(3)

        if not df_buys.empty:
            cols = st.columns(len(df_buys))
            for i, row in enumerate(df_buys.itertuples()):
                with cols[i]:
                    st.metric(
                        label=f"🟢 {row.ticker} (BUY)",
                        value=f"${row.entry_price:,.2f}" if pd.notna(row.entry_price) else "N/A",
                        delta=f"TP: ${row.take_profit:,.2f}" if pd.notna(row.take_profit) else ""
                    )
                    st.caption(f"SL: ${row.stop_loss:,.2f}" if pd.notna(row.stop_loss) else "")
                    with st.expander("Perché l'IA lo consiglia?"):
                        st.write(row.reasoning)
        else:
            st.info("Nessun trade consigliato oggi (Tutti HOLD o SELL).")
    # === FINE NUOVA SEZIONE ===

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuzione segnali")
        sig_counts = df.groupby(["signal", "agent_label"]).size().reset_index(name="n")
        fig = px.bar(
            sig_counts, x="agent_label", y="n", color="signal",
            color_discrete_map=SIGNAL_COLORS,
            labels={"agent_label": "Agente", "n": "# Segnali", "signal": "Segnale"},
            barmode="stack",
        )
        fig.update_layout(margin=dict(t=20, b=40), height=320)
        st.plotly_chart(fig, use_container_width=True)

    # === RADAR CHART (Aggiornato per la Overview) ===
    st.markdown("### 🎯 Analisi Dimensionale dei Trade Consigliati")
    st.write("Visualizzazione dell'allineamento delle 4 dimensioni (Fundamentals, Technical, Sentiment, Macro) per i top trade di oggi:")

    # Usiamo i ticker dei top 3 BUY che abbiamo già calcolato nel Daily Briefing
    if 'df_buys' in locals() and not df_buys.empty and 'latest_run_id' in locals() and latest_run_id:
        tickers_da_mostrare = df_buys['ticker'].tolist()
        radar_cols = st.columns(len(tickers_da_mostrare))
        for idx, t in enumerate(tickers_da_mostrare):
            with radar_cols[idx]:
                fig_radar = plot_dimension_radar(t, latest_run_id)
                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{t}_{idx}")
    else:
        st.info("Nessun trade consigliato su cui mostrare l'analisi dimensionale.")

    with col2:
        st.subheader("Segnali per ticker")
        ticker_counts = df.groupby(["ticker", "signal"]).size().reset_index(name="n")
        fig2 = px.bar(
            ticker_counts, x="ticker", y="n", color="signal",
            color_discrete_map=SIGNAL_COLORS,
            labels={"ticker": "Ticker", "n": "# Segnali"},
            barmode="stack",
        )
        fig2.update_layout(margin=dict(t=20, b=40), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Conviction distribution per segnale")
    fig3 = px.box(
        df[df["signal"].isin(["BUY", "SELL", "HOLD"])],
        x="signal", y="confidence", color="signal",
        color_discrete_map=SIGNAL_COLORS,
        labels={"confidence": "Confidence", "signal": "Segnale"},
        points="outliers",
    )
    fig3.update_layout(margin=dict(t=20), height=280, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: SEGNALI RECENTI
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Segnali recenti":
    st.title("📋 Segnali recenti")

    # ── Filtri — pre-popola ticker da query param ?ticker= ──
    col1, col2, col3, col4 = st.columns(4)
    tickers_list = ["Tutti"] + sorted(df["ticker"].unique().tolist())
    signals = ["Tutti", "BUY", "SELL", "HOLD"]
    agents = ["Tutti"] + sorted(df["agent_label"].unique().tolist())

    # Pre-seleziona ticker se arriva dall'URL
    default_ticker_idx = 0
    if ticker_from_url and ticker_from_url in tickers_list:
        default_ticker_idx = tickers_list.index(ticker_from_url)
        st.info(f"🔗 Filtro applicato automaticamente: **{ticker_from_url}** (link dall'email)")

    sel_ticker = col1.selectbox("Ticker", tickers_list, index=default_ticker_idx)
    sel_signal = col2.selectbox("Segnale", signals)
    sel_agent = col3.selectbox("Agente", agents)
    min_conf = col4.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

    view = df.copy()
    if sel_ticker != "Tutti":
        view = view[view["ticker"] == sel_ticker]
    if sel_signal != "Tutti":
        view = view[view["signal"] == sel_signal]
    if sel_agent != "Tutti":
        view = view[view["agent_label"] == sel_agent]
    view = view[view["confidence"] >= min_conf]

    st.caption(f"{len(view):,} segnali filtrati")

    display_cols = ["date", "agent_label", "ticker", "signal", "confidence",
                    "entry_price", "take_profit", "stop_loss",
                    "actual_return_1d", "actual_return_5d", "actual_return_20d"]
    display_cols = [c for c in display_cols if c in view.columns]
    view_disp = view[display_cols].head(200).copy()

    def color_pnl(val):
        if pd.isna(val):
            return "color: #78909c"
        return "color: #26a69a; font-weight:600" if val > 0 else "color: #ef5350; font-weight:600"

    def color_signal(val):
        colors = {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#78909c"}
        c = colors.get(val, "")
        return f"color: {c}; font-weight:600"

    styled = view_disp.style\
        .map(color_pnl, subset=[c for c in ["actual_return_1d", "actual_return_5d", "actual_return_20d"] if c in view_disp.columns])\
        .map(color_signal, subset=["signal"])\
        .format({
            "confidence": "{:.2f}",
            "entry_price": "${:.2f}",
            "take_profit": "${:.2f}",
            "stop_loss": "${:.2f}",
            "actual_return_1d": "{:+.2%}",
            "actual_return_5d": "{:+.2%}",
            "actual_return_20d": "{:+.2%}",
        }, na_rep="—")

    st.dataframe(styled, use_container_width=True, height=520)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DECISIONI PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Decisioni Portfolio":
    st.title("📊 Decisioni Portfolio")

    if decisions.empty:
        st.info("Nessuna decisione salvata ancora. Esegui almeno un run completo della pipeline.")
        st.stop()

    # ── Selezione run ──
    run_ids = decisions["run_id"].unique().tolist()
    # Mostra data leggibile nel dropdown
    run_labels = {}
    for rid in run_ids:
        ts = decisions[decisions["run_id"] == rid]["timestamp"].max()
        run_labels[rid] = f"{ts.strftime('%Y-%m-%d %H:%M UTC') if pd.notna(ts) else rid}  ({rid[:8]}…)"

    selected_run = st.selectbox(
        "Seleziona run",
        options=run_ids,
        format_func=lambda x: run_labels.get(x, x),
    )

    dec_run = decisions[decisions["run_id"] == selected_run].copy()

    # ── KPI del run selezionato ──
    n_buy  = (dec_run["action"] == "BUY").sum()
    n_sell = (dec_run["action"] == "SELL").sum()
    n_hold = (dec_run["action"] == "HOLD").sum()
    n_veto = dec_run["devil_vetoed"].sum()
    total_long = dec_run[dec_run["action"] == "BUY"]["sizing_pct"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BUY", n_buy)
    c2.metric("SELL", n_sell)
    c3.metric("HOLD", n_hold)
    c4.metric("DA Vetoed", int(n_veto))
    c5.metric("Gross Long %", f"{total_long:.1f}%")

    st.divider()

    # ── Tabella principale del run ──
    st.subheader("Riepilogo decisioni")

    def _fmt_price(v):
        return f"${v:,.2f}" if pd.notna(v) else "—"

    def _fmt_size(v):
        return f"${v:,.0f}" if pd.notna(v) else "—"

    def _fmt_pct(v):
        return f"{v:.1f}%" if pd.notna(v) else "—"

    def _fmt_score(v):
        return f"{v:+.3f}" if pd.notna(v) else "—"

    def _fmt_rr(v):
        return f"1:{v:.0f}" if pd.notna(v) else "—"

    rows_html = ""
    for _, row in dec_run.sort_values(["action", "conviction"], ascending=[True, False]).iterrows():
        action  = row["action"]
        ticker  = row["ticker"]
        vetoed  = row["devil_vetoed"]

        # Colori riga
        if action == "BUY":
            row_bg = "#f0faf5"
            row_text = "#1a3a2a"
            action_style = "background:#d4f4e2;color:#1a7a4a;border:1px solid #1a7a4a;"
        elif action == "SELL":
            row_bg = "#fff5f5"
            row_text = "#3a1a1a"
            action_style = "background:#fde0e0;color:#b03030;border:1px solid #b03030;"
        elif vetoed:
            row_bg = "#fff8f0"
            row_text = "#3a2a1a"
            action_style = "background:#ffe8cc;color:#b05000;border:1px solid #b05000;"
        else:
            row_bg = "#fafafa"
            row_text = "#333333"
            action_style = "background:#f0f0f0;color:#555;border:1px solid #aaa;"

        # Livelli da mostrare: operativi se BUY/SELL, informativi se HOLD
        if action in ("BUY", "SELL") and pd.notna(row.get("entry_price")):
            e_str  = _fmt_price(row["entry_price"])
            sl_str = _fmt_price(row["stop_loss"])
            tp_str = _fmt_price(row["take_profit"])
            sz_str = _fmt_size(row["size_usd"])
            rr_str = _fmt_rr(row["rr_ratio"])
            levels_note = ""
        elif pd.notna(row.get("info_entry")):
            e_str  = _fmt_price(row["info_entry"]) + " ℹ"
            sl_str = _fmt_price(row["info_sl"])
            tp_str = _fmt_price(row["info_tp"])
            sz_str = _fmt_size(row["info_size_usd"])
            rr_str = _fmt_rr(row["info_rr_ratio"])
            levels_note = '<span style="font-size:10px;color:#888;">(informativi)</span>'
        else:
            e_str = sl_str = tp_str = sz_str = rr_str = "—"
            levels_note = ""

        veto_badge = ' <span style="font-size:10px;background:#ffe0cc;color:#b05000;padding:1px 5px;border-radius:3px;">DA veto</span>' if vetoed else ""

        rows_html += f"""
        <tr style="background:{row_bg};border-bottom:1px solid #e0e0e0;color:{row_text};">
          <td style="padding:8px 10px;font-weight:700;color:{row_text};">{ticker}{veto_badge}</td>
          <td style="padding:8px 8px;"><span style="{action_style}padding:2px 8px;border-radius:3px;font-size:12px;font-weight:700;">{action}</span></td>
          <td style="padding:8px 8px;font-size:12px;color:{row_text};">{_fmt_score(row.get('net_score'))}</td>
          <td style="padding:8px 8px;font-size:12px;color:{row_text};">{_fmt_score(row.get('weighted_conviction'))}</td>
          <td style="padding:8px 8px;font-size:12px;color:{row_text};">{_fmt_pct(row.get('conviction', 0) * 100 if pd.notna(row.get('conviction')) else None)}</td>
          <td style="padding:8px 8px;font-size:12px;color:{row_text};">{row.get('consensus', '—')}</td>
          <td style="padding:8px 8px;font-size:12px;color:{row_text};">{e_str} {levels_note}</td>
          <td style="padding:8px 8px;font-size:12px;color:#b03030;">{sl_str}</td>
          <td style="padding:8px 8px;font-size:12px;color:#1a7a4a;">{tp_str}</td>
          <td style="padding:8px 8px;font-size:12px;color:{row_text};">{sz_str}</td>
          <td style="padding:8px 8px;font-size:12px;color:{row_text};">{rr_str}</td>
          <td style="padding:8px 8px;font-size:11px;color:#444;max-width:220px;">{str(row.get('reasoning',''))[:80]}…</td>
        </tr>"""

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="background:#0d1b2a;color:#a0b4c8;text-align:left;">
          <th style="padding:8px 10px;">Ticker</th>
          <th style="padding:8px 8px;">Azione</th>
          <th style="padding:8px 8px;" title="Da -1.0 a +1.0. Oltre +0.25 è un forte segnale di acquisto (Bullish).">Net Score ℹ️</th>
          <th style="padding:8px 8px;" title="Weighted Conviction: La forza reale. Sopra 0.008 (0.8%) il sistema consiglia il trade.">WC ℹ️</th>
          <th style="padding:8px 8px;" title="Sicurezza media degli agenti che hanno votato per questa direzione.">Conviction ℹ️</th>
          <th style="padding:8px 8px;" title="Frazione di dimensioni (Es. Macro, Fundamentals) d'accordo con il trade.">Consensus ℹ️</th>
          <th style="padding:8px 8px;">Entry</th>
          <th style="padding:8px 8px;color:#e07070;">SL</th>
          <th style="padding:8px 8px;color:#70c070;">TP</th>
          <th style="padding:8px 8px;" title="Dimensione consigliata per il trade calcolata sul rischio (VaR) e volatilità.">Size $ ℹ️</th>
          <th style="padding:8px 8px;">R:R</th>
          <th style="padding:8px 8px;">Reasoning</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""

    st.markdown(table_html, unsafe_allow_html=True)

    st.divider()

    # ── Storico conviction per ticker nel tempo ──
    st.subheader("📈 Storico conviction per ticker")

    tickers_with_data = sorted(decisions["ticker"].unique().tolist())
    sel_tickers_hist = st.multiselect(
        "Seleziona ticker",
        options=tickers_with_data,
        default=tickers_with_data[:min(5, len(tickers_with_data))],
    )

    if sel_tickers_hist:
        hist_data = decisions[decisions["ticker"].isin(sel_tickers_hist)].copy()
        hist_data = hist_data.sort_values("timestamp")

        fig_conv = px.line(
            hist_data,
            x="timestamp",
            y="conviction",
            color="ticker",
            markers=True,
            labels={"timestamp": "Data", "conviction": "Conviction", "ticker": "Ticker"},
            title="Conviction nel tempo per ticker",
        )
        fig_conv.add_hline(y=0.30, line_dash="dot", line_color="orange",
                           annotation_text="MIN_CONVICTION_TO_TRADE=0.30")
        fig_conv.update_layout(height=350, margin=dict(t=40))
        st.plotly_chart(fig_conv, use_container_width=True)

        fig_ns = px.line(
            hist_data,
            x="timestamp",
            y="net_score",
            color="ticker",
            markers=True,
            labels={"timestamp": "Data", "net_score": "Net Score", "ticker": "Ticker"},
            title="Net Score nel tempo per ticker",
        )
        fig_ns.add_hline(y=0.25, line_dash="dot", line_color="#26a69a",
                         annotation_text="BUY threshold")
        fig_ns.add_hline(y=-0.25, line_dash="dot", line_color="#ef5350",
                         annotation_text="SELL threshold")
        fig_ns.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.2)
        fig_ns.update_layout(height=350, margin=dict(t=40))
        st.plotly_chart(fig_ns, use_container_width=True)

    st.divider()

    # ── Distribuzione azioni per run ──
    st.subheader("Distribuzione azioni storiche per run")
    action_hist = decisions.groupby(["date", "action"]).size().reset_index(name="n")
    fig_act = px.bar(
        action_hist, x="date", y="n", color="action",
        color_discrete_map={"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#78909c"},
        labels={"date": "Data", "n": "# Ticker", "action": "Azione"},
        barmode="stack",
    )
    fig_act.update_layout(height=300, margin=dict(t=20))
    st.plotly_chart(fig_act, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance":
    st.title("📈 Performance storica")

    active = df[df["signal"].isin(["BUY", "SELL"])].copy()

    wr_1d  = win_rate(active["actual_return_1d"])
    wr_5d  = win_rate(active["actual_return_5d"])
    wr_20d = win_rate(active["actual_return_20d"])
    n_eval = active["actual_return_5d"].notna().sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win rate 1d",  f"{wr_1d*100:.1f}%"  if wr_1d  else "N/A")
    c2.metric("Win rate 5d",  f"{wr_5d*100:.1f}%"  if wr_5d  else "N/A")
    c3.metric("Win rate 20d", f"{wr_20d*100:.1f}%" if wr_20d else "N/A")
    c4.metric("Segnali valutati (5d)", n_eval)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("P&L simulato cumulato (5d, size=1%)")
        pnl_df = active[["date", "actual_return_5d"]].dropna().sort_values("date").copy()
        pnl_df["pnl_cum"] = (pnl_df["actual_return_5d"] * 0.01).cumsum() * 100
        fig = px.area(
            pnl_df, x="date", y="pnl_cum",
            labels={"date": "Data", "pnl_cum": "P&L cumulato (%)"},
            color_discrete_sequence=["#26a69a"],
        )
        fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
        fig.update_layout(margin=dict(t=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Win rate per agente (5d)")
        wr_agent = (
            active.groupby("agent_label")["actual_return_5d"]
            .apply(lambda s: (s.dropna() > 0).sum() / max(s.dropna().count(), 1))
            .reset_index()
        )
        wr_agent.columns = ["Agente", "Win rate"]
        wr_agent = wr_agent.sort_values("Win rate", ascending=True)
        fig2 = px.bar(
            wr_agent, x="Win rate", y="Agente", orientation="h",
            color="Win rate", color_continuous_scale=["#ef5350", "#ffeb3b", "#26a69a"],
            range_color=[0, 1],
        )
        fig2.add_vline(x=0.5, line_dash="dot", line_color="white", opacity=0.5)
        fig2.update_layout(margin=dict(t=20), height=300, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Distribuzione ritorni per agente (5d)")
    fig3 = px.box(
        active.dropna(subset=["actual_return_5d"]),
        x="agent_label", y="actual_return_5d",
        color="signal", color_discrete_map=SIGNAL_COLORS,
        labels={"agent_label": "Agente", "actual_return_5d": "Ritorno 5d"},
        points="outliers",
    )
    fig3.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
    fig3.update_layout(margin=dict(t=20), height=320)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Heatmap: frequenza segnali per ticker × mese")
    hm = df.copy()
    hm["month"] = pd.to_datetime(hm["timestamp"]).dt.to_period("M").astype(str)
    heat = hm.groupby(["ticker", "month"]).size().reset_index(name="n")
    heat_pivot = heat.pivot(index="ticker", columns="month", values="n").fillna(0)
    fig4 = px.imshow(
        heat_pivot,
        color_continuous_scale="Blues",
        labels={"color": "# Segnali"},
        aspect="auto",
    )
    fig4.update_layout(margin=dict(t=20), height=280)
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PESI AGENTI
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Pesi agenti":
    st.title("⚖️ Weight Adjuster — Evoluzione pesi EWA")

    if weights.empty:
        st.info("Nessun dato pesi disponibile.")
        st.stop()

    weights["agent_label"] = weights["agent_id"].map(AGENT_LABELS).fillna(weights["agent_id"])

    latest = weights.sort_values("updated_at").groupby(["agent_id", "ticker"]).last().reset_index()
    latest["agent_label"] = latest["agent_id"].map(AGENT_LABELS).fillna(latest["agent_id"])

    st.subheader("Pesi correnti per agente (media su tutti i ticker)")
    avg_w = latest.groupby("agent_label")["weight"].mean().reset_index().sort_values("weight", ascending=False)
    top_agent = avg_w.iloc[0]["agent_label"]
    bot_agent = avg_w.iloc[-1]["agent_label"]

    c1, c2 = st.columns(2)
    c1.metric("🥇 Agente con peso più alto", top_agent, f"{avg_w.iloc[0]['weight']:.3f}")
    c2.metric("🔻 Agente con peso più basso", bot_agent, f"{avg_w.iloc[-1]['weight']:.3f}")

    fig = px.bar(
        avg_w, x="agent_label", y="weight",
        color="weight", color_continuous_scale=["#ef5350", "#ffeb3b", "#26a69a"],
        labels={"agent_label": "Agente", "weight": "Peso medio"},
    )
    fig.update_layout(margin=dict(t=20), height=300, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evoluzione pesi nel tempo")
    sel_ticker_w = st.selectbox("Filtra per ticker", ["Tutti"] + sorted(latest["ticker"].unique().tolist()))

    w_plot = weights.copy()
    if sel_ticker_w != "Tutti":
        w_plot = w_plot[w_plot["ticker"] == sel_ticker_w]

    w_line = w_plot.groupby(["updated_at", "agent_label"])["weight"].mean().reset_index()
    fig2 = px.line(
        w_line, x="updated_at", y="weight", color="agent_label",
        labels={"updated_at": "Data", "weight": "Peso", "agent_label": "Agente"},
        markers=True,
    )
    fig2.update_layout(margin=dict(t=20), height=350)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tabella pesi correnti")
    pivot_w = latest.pivot_table(index="agent_label", columns="ticker", values="weight").round(3)
    st.dataframe(pivot_w.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE RUNS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔁 Pipeline runs":
    st.title("🔁 Storico Pipeline Runs")

    if runs.empty:
        st.info("Nessun run registrato.")
        st.stop()

    success = runs["status"].str.lower().str.contains("success|completed", na=False).sum()
    failed  = (~runs["status"].str.lower().str.contains("success|completed", na=False)).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Run totali", len(runs))
    c2.metric("✅ Success", success)
    c3.metric("❌ Failed", failed)

    st.subheader("Run recenti")

    def color_status(val):
        if str(val).lower() in ("success", "completed"):
            return "color: #26a69a; font-weight:600"
        return "color: #ef5350; font-weight:600"

    disp = runs[["started_at", "status", "tickers", "error_msg"]].head(50).copy()
    disp["started_at"] = disp["started_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    styled_runs = disp.style.map(color_status, subset=["status"])
    st.dataframe(styled_runs, use_container_width=True, height=450)

    if failed > 0:
        st.subheader("❌ Errori recenti")
        errors = runs[runs["status"] != "success"][["started_at", "status", "error_msg"]].dropna(subset=["error_msg"])
        if not errors.empty:
            for _, row in errors.head(5).iterrows():
                with st.expander(f"{row['started_at']} — {row['status']}"):
                    st.code(row["error_msg"])
