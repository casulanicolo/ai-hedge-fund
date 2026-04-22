"""
dashboard/tabs/attribution.py
─────────────────────────────
Tab 3 — Agent Attribution.

Composition (top → bottom):
  1. Window selector              — 1d / 5d / 20d (forward-return horizon)
  2. Scorecard table              — accuracy, predictions, confidence, weight, trend
  3. Contribution heatmap | Struggling agents card
  4. Weight trend plot            — daily mean weight per agent over selected window

Attribution proxy:
    contribution_bps = signal_dir × weight × actual_return × 10_000
where signal_dir ∈ {+1 BUY, −1 SELL, 0 HOLD}. Basis-point proxy of P&L
impact, NOT realized dollar P&L (which lives in executed_orders).

All readers go through `ds`; all rendering through `c`. Tab is read-only.
"""

from __future__ import annotations

import streamlit as st


def render(ds, c) -> None:
    # ── Header + window selector ───────────────────────────────────────────
    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown(
            "**Agent Attribution** · "
            "<span style='color:#8a8f98;font-size:11px;'>"
            "signed-weighted contribution proxy (basis points)"
            "</span>",
            unsafe_allow_html=True,
        )
    with head_right:
        window = st.selectbox(
            "Forward window",
            options=("1d", "5d", "20d"),
            index=1,
            key="attribution_window",
            label_visibility="collapsed",
        )

    # ── Scorecard ──────────────────────────────────────────────────────────
    accuracy = ds.get_agent_accuracy(window=window)
    weights  = ds.get_agent_weights()
    st.markdown("&nbsp;")
    st.markdown(f"**Scorecard** · `{window}` forward returns")
    c.agent_scorecard(accuracy, weights)

    # ── Heatmap | Struggling card ──────────────────────────────────────────
    st.markdown("&nbsp;")
    left, right = st.columns([2, 1], gap="medium")
    with left:
        st.markdown("**Contribution heatmap** · agent × ticker (bps)")
        contrib = ds.get_attribution_pl_ytd(window=window)
        c.contribution_heatmap(contrib)
    with right:
        st.markdown("**Struggling agents**")
        struggling = ds.get_struggling_agents(window=window, min_predictions=5)
        c.struggling_agents_card(struggling)

    # ── Weight trend ───────────────────────────────────────────────────────
    st.markdown("&nbsp;")

    trend_left, trend_right = st.columns([3, 1])
    with trend_left:
        st.markdown("**Weight trend** · daily mean per agent")
    with trend_right:
        trend_window = st.selectbox(
            "Trend window",
            options=("30d", "90d", "180d", "1Y"),
            index=1,
            key="weight_trend_window",
            label_visibility="collapsed",
        )

    days_lookup = {"30d": 30, "90d": 90, "180d": 180, "1Y": 365}
    trend = ds.get_weight_trend(days=days_lookup.get(trend_window, 90))
    c.weight_trend_plot(trend, height=300)

    # ── Footer note ────────────────────────────────────────────────────────
    st.caption(
        "Contribution = `signal_dir × weight × actual_return × 10_000`. "
        "Basis-point proxy ranks directional impact per agent×ticker. "
        "Realized dollar P&L lives in the Positions tab."
    )
