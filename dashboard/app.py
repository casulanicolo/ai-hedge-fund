import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

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

# ── DB helper ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect(DB_PATH)

    predictions = pd.read_sql("""
        SELECT id, run_id, agent_id, ticker, signal, confidence,
               timestamp, expected_return, entry_price, stop_loss, take_profit
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

    # ── Join predictions + outcomes (1d pivot) ──
    out_1d = outcomes[outcomes["window"] == "1d"][["prediction_id", "actual_return_1d"]].copy()
    out_5d = outcomes[outcomes["window"] == "5d"][["prediction_id", "actual_return_5d"]].copy()
    out_20d = outcomes[outcomes["window"] == "20d"][["prediction_id", "actual_return_20d"]].copy()

    # If window column not populated, use unique prediction_id rows
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


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/fire-element.png", width=60)
st.sidebar.title("Athanor Alpha")
st.sidebar.caption("Multi-Agent Trading Dashboard")

page = st.sidebar.radio(
    "Pannello",
    ["🏠 Overview", "📋 Segnali recenti", "📈 Performance", "⚖️ Pesi agenti", "🔁 Pipeline runs"],
)

st.sidebar.divider()
st.sidebar.caption(f"DB: `{DB_PATH}`")
if st.sidebar.button("🔄 Refresh dati"):
    st.cache_data.clear()
    st.rerun()

# ── Load ─────────────────────────────────────────────────────────────────────
try:
    df, weights, runs, positions, cache = load_data()
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
    """Simulated cumulative P&L assuming 1% portfolio per signal."""
    s = sub["actual_return_5d"].dropna()
    return (s * 0.01).cumsum()


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

    # ── Filtri ──
    col1, col2, col3, col4 = st.columns(4)
    tickers = ["Tutti"] + sorted(df["ticker"].unique().tolist())
    signals = ["Tutti", "BUY", "SELL", "HOLD"]
    agents = ["Tutti"] + sorted(df["agent_label"].unique().tolist())

    sel_ticker = col1.selectbox("Ticker", tickers)
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

    # ── Colonne display ──
    display_cols = ["date", "agent_label", "ticker", "signal", "confidence",
                    "entry_price", "take_profit", "stop_loss",
                    "actual_return_1d", "actual_return_5d", "actual_return_20d"]
    display_cols = [c for c in display_cols if c in view.columns]
    view_disp = view[display_cols].head(200).copy()

    # ── Colorazione condizionale P&L ──
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
# PAGE: PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performance":
    st.title("📈 Performance storica")

    active = df[df["signal"].isin(["BUY", "SELL"])].copy()

    # ── KPI row ──
    wr_1d = win_rate(active["actual_return_1d"])
    wr_5d = win_rate(active["actual_return_5d"])
    wr_20d = win_rate(active["actual_return_20d"])
    n_eval = active["actual_return_5d"].notna().sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win rate 1d", f"{wr_1d*100:.1f}%" if wr_1d else "N/A")
    c2.metric("Win rate 5d", f"{wr_5d*100:.1f}%" if wr_5d else "N/A")
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

    # ── Pesi correnti (ultimo per agente+ticker) ──
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
    failed = (~runs["status"].str.lower().str.contains("success|completed", na=False)).sum()
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
