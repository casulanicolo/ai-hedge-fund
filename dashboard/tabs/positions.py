"""
dashboard/tabs/positions.py
───────────────────────────
Tab 2 — Positions, Open Orders, Trade History.

Composition:
  1. Summary strip               — counts (long/short) + total unrealized
  2. Open positions table        — broker positions enriched with SL/TP/days/tag
  3. Open orders table           — pending orders from broker
  4. Trade history table         — FIFO-paired closed trades (realized P&L)

Auto-refresh: 60s when broker is online (best-effort via streamlit-autorefresh).
All readers go through `ds`; all rendering through `c`. Tab is read-only.
"""

from __future__ import annotations

import streamlit as st


def _maybe_autorefresh(interval_ms: int = 60_000) -> None:
    """Tick page every `interval_ms` if streamlit-autorefresh is installed."""
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_ms, key="positions_autorefresh")
    except Exception:
        pass


def render(ds, c) -> None:
    # ── Auto-refresh (best effort) ─────────────────────────────────────────
    _maybe_autorefresh(60_000)

    # ── Summary strip (always rendered — zeroes when broker offline) ───────
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

    # ── Trade history ──────────────────────────────────────────────────────
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
    history = ds.get_trade_history(days=days_lookup.get(window, 30))
    c.trade_history_table(history)
