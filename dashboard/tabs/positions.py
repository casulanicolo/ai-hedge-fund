"""
dashboard/tabs/positions.py
───────────────────────────
Tab 2 — Positions, Open Orders, Trade History, Stats, Chat.

Composition:
  1. Summary strip               — AUM / cash / positions / buying power
  2. Open positions table        — broker positions enriched with SL/TP/days/tag
  3. Open orders table           — pending orders from broker
  4. Trade statistics            — win rate, profit factor, avg win/loss, streaks
  5. Trade history table         — closed trades (cards or table view)
  6. Portfolio Analyst chat      — LLM agent on portfolio context
"""

from __future__ import annotations

import json
import os

import streamlit as st


def _maybe_autorefresh(interval_ms: int = 60_000) -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_ms, key="positions_autorefresh")
    except Exception:
        pass


def _render_trade_stats(stats: dict, c) -> None:
    """Render the 📊 Trade Statistics section."""
    if not stats or stats.get("total_trades", 0) == 0:
        st.info("Nessun trade chiuso nel periodo selezionato per le statistiche.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wr = stats.get("win_rate")
        st.metric("Win Rate", c.fmt_pct(wr, 1) if wr is not None else "—")
    with col2:
        pf = stats.get("profit_factor")
        st.metric("Profit Factor", c.fmt_ratio(pf, 2) if pf is not None else "—")
    with col3:
        aw = stats.get("avg_win")
        st.metric("Avg Win", c.fmt_usd(aw, 2) if aw is not None else "—")
    with col4:
        al = stats.get("avg_loss")
        st.metric("Avg Loss", c.fmt_usd(al, 2) if al is not None else "—")

    # Streaks + summary line
    total  = stats.get("total_trades", 0)
    n_win  = stats.get("winning_trades", 0)
    n_loss = stats.get("losing_trades", 0)
    mws    = stats.get("max_win_streak", 0)
    mls    = stats.get("max_loss_streak", 0)
    tpl    = stats.get("total_pl", 0.0)
    best   = stats.get("best_trade")
    worst  = stats.get("worst_trade")

    def _no_dollar(s: str) -> str:
        return s.replace("$", "USD ")

    st.caption(
        f"{total} trades · {n_win}W / {n_loss}L  ·  "
        f"Win streak: {mws}  ·  Loss streak: {mls}  ·  "
        f"Total P&L: {_no_dollar(c.fmt_usd(tpl, 2))}"
        + (f"  ·  Best: {best['ticker']} {_no_dollar(c.fmt_usd(best['pl'], 2))}" if best else "")
        + (f"  ·  Worst: {worst['ticker']} {_no_dollar(c.fmt_usd(worst['pl'], 2))}" if worst else "")
    )

    # Top/bottom 3 tickers
    by_ticker = stats.get("stats_by_ticker") or {}
    if by_ticker:
        items = list(by_ticker.items())
        top3    = items[:3]
        bottom3 = items[-3:] if len(items) > 3 else []
        shown   = top3 + ([("…", None)] if bottom3 else []) + bottom3
        rows = []
        for tkr, s in shown:
            if s is None:
                rows.append({"Ticker": "—", "Trades": "—", "Win Rate": "—", "Total P&L": "—"})
            else:
                rows.append({
                    "Ticker":    tkr,
                    "Trades":    s["n"],
                    "Win Rate":  c.fmt_pct(s["win_rate"], 1) if s["win_rate"] is not None else "—",
                    "Total P&L": c.fmt_usd(s["total_pl"], 2),
                })
        import pandas as _pd
        st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_chat(ds, c) -> None:
    """Render the 🤖 Portfolio Analyst chat section."""
    import anthropic as _anthropic

    if "portfolio_chat_history" not in st.session_state:
        st.session_state["portfolio_chat_history"] = []

    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.markdown("**🤖 Portfolio Analyst**")
    with col_clear:
        if st.button("🗑️ Cancella", key="chat_clear"):
            st.session_state["portfolio_chat_history"] = []
            st.rerun()

    # Display history
    for msg in st.session_state["portfolio_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Chiedimi delle posizioni, dei trade, dei rischi...")
    if not prompt:
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("ANTHROPIC_API_KEY non trovata nel .env — chat non disponibile.")
        return

    # Build context
    try:
        positions   = ds.get_positions()
        history_raw = ds.get_trade_history(days=30)
        stats       = ds.get_trade_stats(history_raw)
        acct        = ds.get_account_snapshot()

        pos_records = positions.head(20).to_dict("records") if not positions.empty else []
        hist_records = (history_raw.drop(columns=["top_agents"], errors="ignore")
                        .head(10).to_dict("records")) if not history_raw.empty else []

        context = json.dumps({
            "account":       acct,
            "open_positions": pos_records,
            "trade_history_last10": hist_records,
            "trade_stats":   {k: v for k, v in stats.items() if k != "stats_by_ticker"},
        }, default=str, ensure_ascii=False)
    except Exception as exc:
        context = f"(dati non disponibili: {exc})"

    system_prompt = (
        "Sei un analista di portafoglio esperto. Hai accesso ai seguenti dati "
        "del portafoglio Athanor Alpha: " + context +
        "\n\nRispondi in italiano in modo conciso e professionale. "
        "Non inventare dati non presenti nel contesto."
    )

    # Append user message
    st.session_state["portfolio_chat_history"].append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API with streaming
    client = _anthropic.Anthropic(api_key=api_key)
    messages_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["portfolio_chat_history"]
    ]

    with st.chat_message("assistant"):
        try:
            full_response = ""
            placeholder   = st.empty()
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                system=system_prompt,
                messages=messages_payload,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            st.session_state["portfolio_chat_history"].append(
                {"role": "assistant", "content": full_response}
            )
        except Exception as exc:
            err = f"Errore API: {exc}"
            st.error(err)
            st.session_state["portfolio_chat_history"].append(
                {"role": "assistant", "content": err}
            )


def render(ds, c) -> None:
    _maybe_autorefresh(60_000)

    # ── Summary strip ──────────────────────────────────────────────────────
    summary = ds.get_positions_summary()
    c.positions_summary_strip(summary)

    acct_online = bool(ds.get_account_snapshot())
    if not acct_online:
        st.caption(
            "Alpaca offline — set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY / "
            "ALPACA_BASE_URL in `.env` to populate live positions and orders."
        )

    # ── Open positions ─────────────────────────────────────────────────────
    st.markdown("&nbsp;")
    st.markdown("**Open positions**")
    positions = ds.get_positions()
    c.positions_table(positions)

    # ── Open orders ────────────────────────────────────────────────────────
    st.markdown("&nbsp;")
    st.markdown("**Open orders**")
    orders = ds.get_open_orders()
    c.orders_table(orders)

    # ── Trade history window selector ──────────────────────────────────────
    st.markdown("&nbsp;")
    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown("**Trade history**")
    with head_right:
        window = st.selectbox(
            "Window",
            options=("7d", "30d", "90d", "YTD", "All"),
            index=1,
            key="trade_history_window",
            label_visibility="collapsed",
        )

    days_lookup = {"7d": 7, "30d": 30, "90d": 90, "YTD": 365, "All": 3650}
    days        = days_lookup.get(window, 30)
    history     = ds.get_trade_history(days=days)

    # ── Trade statistics ───────────────────────────────────────────────────
    st.markdown("**📊 Trade Statistics**")
    stats = ds.get_trade_stats(history)
    _render_trade_stats(stats, c)

    # ── Trade history table ────────────────────────────────────────────────
    st.markdown("&nbsp;")
    c.trade_history_table(history)

    # ── Portfolio Analyst chat ─────────────────────────────────────────────
    st.markdown("&nbsp;")
    st.markdown("---")
    _render_chat(ds, c)
