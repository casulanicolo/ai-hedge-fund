"""
dashboard/tabs/regime.py
────────────────────────
Tab 5 — Regime & Risk.

Composition (top → bottom):
  1. Macro strip               — VIX (sparkline) · Regime · UST 10Y · 3M+spread
  2. Correlation heatmap       — return-correlation across current positions
  3. VaR card | Stress table   — historical 1d-95% VaR | linear factor scenarios

All readers go through `ds`; all rendering through `c`. Tab is read-only.
Stress tests are a v1 linear approximation (per-position betas hard-coded
in data_sources). A real factor regression replaces this in a later round.
"""

from __future__ import annotations

import streamlit as st


def render(ds, c) -> None:
    # ── Macro strip ────────────────────────────────────────────────────────
    macro = ds.get_macro_snapshot()
    c.macro_strip(macro)

    # ── Correlation heatmap ────────────────────────────────────────────────
    st.markdown("&nbsp;")

    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown("**Correlation heatmap** · current positions, daily returns")
    with head_right:
        window = st.selectbox(
            "Window",
            options=("10d", "20d", "60d", "120d"),
            index=1,
            key="corr_window",
            label_visibility="collapsed",
        )
    days_lookup = {"10d": 10, "20d": 20, "60d": 60, "120d": 120}

    positions = ds.get_positions()
    if positions is None or positions.empty:
        c.empty_placeholder("No open positions — correlation matrix needs ≥2 tickers.")
    else:
        tickers = tuple(sorted(set(positions["ticker"].astype(str))))
        corr = ds.get_correlation_matrix(tickers, days=days_lookup.get(window, 20))
        c.correlation_heatmap(corr)

    # ── VaR | Stress ───────────────────────────────────────────────────────
    st.markdown("&nbsp;")
    left, right = st.columns([1, 2], gap="medium")
    with left:
        st.markdown("**Value at Risk**")
        c.var_card(ds.get_var_1d_95())
    with right:
        st.markdown("**Stress tests** · linear factor approximation (v1)")
        c.stress_table(ds.get_stress_tests())

    # ── Footer note ────────────────────────────────────────────────────────
    st.caption(
        "Stress betas: SPY=1.0 (all), VIX=−0.20, 10Y rate=−0.5 (−3.0 for "
        "TLT/IEF/IEI/TLH), Tech=1.0 for AAPL/NVDA/MSFT/GOOGL/META/AMZN/TSLA/AMD/QQQ/XLK/SOXX. "
        "v1 — replaces a real factor regression later."
    )
