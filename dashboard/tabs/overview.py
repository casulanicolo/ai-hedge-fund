"""
dashboard/tabs/overview.py
──────────────────────────
Tab 1 — Overview.

Composition (top → bottom):
  1. KPI row (6 cards)              — AUM, P&L YTD, Sharpe 1Y, MaxDD 1Y,
                                       Win Rate 20, Alpha vs SPY YTD
  2. Equity vs SPY     +  Exposure  — Plotly line + range pills | stacked gauge
  3. Drawdown                       — under-water plot, same X axis as equity

All readers go through `ds`; all rendering through `c`. Tab is read-only.
"""

from __future__ import annotations


def render(ds, c) -> None:
    # ── KPI row ────────────────────────────────────────────────────────────
    k = ds.get_overview_kpis()
    cols = __import__("streamlit").columns(6, gap="small")
    with cols[0]:
        c.kpi_card("AUM", c.fmt_usd(k.get("aum"), 0))
    with cols[1]:
        pl     = k.get("pl_ytd")
        pl_pct = k.get("pl_ytd_pct")
        c.kpi_card(
            "P&L YTD",
            c.fmt_usd(pl, 0),
            delta=c.fmt_pct(pl_pct, 2, signed=True) if pl_pct is not None else None,
            delta_color="normal" if (pl_pct or 0) >= 0 else "inverse",
        )
    with cols[2]:
        c.kpi_card("Sharpe 1Y", c.fmt_ratio(k.get("sharpe_1y"), 2))
    with cols[3]:
        c.kpi_card("Max DD 1Y", c.fmt_pct(k.get("max_dd_1y"), 2, signed=True))
    with cols[4]:
        wr = k.get("win_rate_20")
        c.kpi_card("Win Rate", c.fmt_pct(wr, 1) if wr is not None else "—",
                   help_text="Realized closed trades from executed_orders (FIFO pairing).")
    with cols[5]:
        a = k.get("alpha_vs_spy_ytd")
        c.kpi_card(
            "Alpha vs SPY",
            c.fmt_pct(a, 2, signed=True) if a is not None else "—",
            delta_color="normal" if (a or 0) >= 0 else "inverse",
            help_text="Portfolio YTD return − SPY YTD return.",
        )

    # ── Range pills + Equity / Exposure (2 cols 2:1) ───────────────────────
    st = __import__("streamlit")
    st.markdown("&nbsp;")
    range_pill = c.range_pills(key="overview_range", default="1Y")

    days_lookup = {"1Y": 365, "6M": 182, "3M": 92, "1M": 31, "WTD": 14}
    eq_full = ds.get_equity_curve(days=days_lookup.get(range_pill, 365))
    eq      = ds._filter_range(eq_full, range_pill)

    bm_full = ds.get_benchmark("SPY", days=days_lookup.get(range_pill, 365))
    bm      = ds._filter_range(bm_full, range_pill)

    left, right = st.columns([2, 1], gap="medium")
    with left:
        st.markdown(f"**Equity vs SPY** · `{range_pill}`")
        c.equity_plot(eq, bm, height=320)
    with right:
        st.markdown("**Exposure**")
        ex = ds.get_exposure_breakdown()
        c.exposure_gauge(
            long_pct=ex["long_pct"], short_pct=ex["short_pct"],
            cash_pct=ex["cash_pct"], net_pct=ex["net_pct"],
            gross_pct=ex["gross_pct"], beta_proxy=ex["beta_proxy"],
        )

    # ── Drawdown ───────────────────────────────────────────────────────────
    st.markdown("**Drawdown**")
    c.drawdown_plot(eq, height=180)

    # ── Footer hint when broker offline ───────────────────────────────────
    acct = ds.get_account_snapshot()
    if not acct:
        st.caption(
            "Alpaca offline — set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY / "
            "ALPACA_BASE_URL in .env to populate live data."
        )
