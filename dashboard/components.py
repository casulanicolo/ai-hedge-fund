"""
dashboard/components.py
───────────────────────
Reusable UI primitives. Tabs compose these; no tab is allowed to call
plotly / pandas styler directly — every visual lives here.

Conventions
-----------
- All renderers take primitive Python types or DataFrames, never raw
  Alpaca/SQLite handles. Reading is `data_sources.py`'s job.
- Every renderer must handle "empty" gracefully — show a soft placeholder
  via `empty_placeholder()` instead of crashing.
- Numeric formatting goes through `fmt_*` helpers so the styling stays
  consistent across tabs.

Tab-1 primitives (kpi_card, equity_plot, drawdown_plot, exposure_gauge,
range_pills, header_strip, empty_placeholder) are implemented fully.
Tab-2…6 primitives are stubs that render a placeholder banner — they
will be filled when each tab lands.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Brand palette (mirrors static/athanor.css)
# ─────────────────────────────────────────────────────────────────────────────

COLOR_BG          = "#0e1117"
COLOR_FG          = "#e6e6e6"
COLOR_MUTED       = "#8a8f98"
COLOR_ACCENT      = "#3ec1d3"   # cyan — portfolio
COLOR_BENCHMARK   = "#8a8f98"   # gray dotted — SPY
COLOR_POSITIVE    = "#27ae60"
COLOR_NEGATIVE    = "#c0392b"
COLOR_WARN        = "#e8a83a"
COLOR_KILL        = "#ff3b3b"


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_usd(v: Optional[float], decimals: int = 0) -> str:
    if v is None:
        return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.{decimals}f}"


def fmt_pct(v: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    if v is None:
        return "—"
    fmt = f"{{:+.{decimals}f}}%" if signed else f"{{:.{decimals}f}}%"
    return fmt.format(v)


def fmt_ratio(v: Optional[float], decimals: int = 2) -> str:
    return "—" if v is None else f"{v:.{decimals}f}"


def fmt_int(v: Optional[float]) -> str:
    return "—" if v is None else f"{int(v):,}"


# ─────────────────────────────────────────────────────────────────────────────
# Empty-state placeholder
# ─────────────────────────────────────────────────────────────────────────────

def empty_placeholder(msg: str = "No data yet.", *, icon: str = "ℹ") -> None:
    st.markdown(
        f"<div style='padding:14px;border:1px dashed #444;border-radius:6px;"
        f"color:{COLOR_MUTED};text-align:center;font-family:monospace;'>"
        f"{icon}&nbsp;&nbsp;{msg}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header strip (top of every page)
# ─────────────────────────────────────────────────────────────────────────────

def header_strip(*, mode: str, last_sync: str, kill_armed: bool) -> None:
    badge_kill = (
        f"<span style='background:{COLOR_KILL};color:#fff;padding:2px 8px;"
        f"border-radius:4px;font-weight:bold;'>KILL ARMED</span>"
        if kill_armed else ""
    )
    badge_mode = (
        f"<span style='background:#222;color:{COLOR_ACCENT};padding:2px 8px;"
        f"border-radius:4px;font-family:monospace;'>{mode}</span>"
    )
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;padding:6px 0;border-bottom:1px solid #222;"
        f"margin-bottom:14px;font-family:monospace;color:{COLOR_MUTED};'>"
        f"<div><b style='color:{COLOR_FG};'>ATHANOR ALPHA</b> · v2 · "
        f"{badge_mode} · last sync {last_sync}</div>"
        f"<div>{badge_kill}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KPI card (Tab 1)
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(
    label:    str,
    value:    str,
    *,
    delta:    Optional[str]  = None,
    delta_color: str         = "off",     # "normal" | "inverse" | "off"
    help_text: Optional[str] = None,
) -> None:
    """Thin wrapper over st.metric with monospace value + brand colors."""
    st.metric(label=label, value=value, delta=delta,
              delta_color=delta_color, help=help_text)


# ─────────────────────────────────────────────────────────────────────────────
# Equity curve + benchmark overlay (Tab 1)
# ─────────────────────────────────────────────────────────────────────────────

RANGE_OPTIONS: tuple[str, ...] = ("1Y", "6M", "3M", "1M", "WTD")


def range_pills(key: str, *, default: str = "1Y") -> str:
    """Range selector pills used by Tab 1 equity chart."""
    if hasattr(st, "segmented_control"):
        return st.segmented_control(
            "Range", options=list(RANGE_OPTIONS),
            default=default, key=key, label_visibility="collapsed",
        ) or default
    return st.radio(
        "Range", RANGE_OPTIONS,
        index=RANGE_OPTIONS.index(default), key=key,
        horizontal=True, label_visibility="collapsed",
    )


def equity_plot(
    equity:    pd.DataFrame,         # cols: date, equity
    benchmark: pd.DataFrame,         # cols: date, close
    *,
    title:     str = "",
    height:    int = 320,
) -> None:
    """Portfolio equity (cyan solid) vs benchmark (gray dotted, rebased)."""
    if equity.empty:
        empty_placeholder("Equity curve unavailable — no trade history yet.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity["date"], y=equity["equity"],
        mode="lines", name="Portfolio",
        line=dict(color=COLOR_ACCENT, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ))

    if not benchmark.empty:
        # rebase benchmark to portfolio start
        start_eq = float(equity["equity"].iloc[0])
        bm = benchmark.copy()
        if not bm.empty and float(bm["close"].iloc[0]) > 0:
            bm["rebased"] = bm["close"] / float(bm["close"].iloc[0]) * start_eq
            fig.add_trace(go.Scatter(
                x=bm["date"], y=bm["rebased"],
                mode="lines", name="SPY (rebased)",
                line=dict(color=COLOR_BENCHMARK, width=1.4, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>SPY</extra>",
            ))

    fig.update_layout(
        title=title or None,
        height=height,
        margin=dict(l=10, r=10, t=30 if title else 10, b=10),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_FG, family="JetBrains Mono, monospace", size=11),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
        xaxis=dict(gridcolor="#222"),
        yaxis=dict(gridcolor="#222", tickformat="$,.0f"),
    )
    st.plotly_chart(fig, use_container_width=True)


def drawdown_plot(equity: pd.DataFrame, *, height: int = 180) -> None:
    """Under-water plot. Computed inline — no SQL."""
    if equity.empty:
        empty_placeholder("Drawdown unavailable.")
        return
    eq = equity["equity"].astype(float)
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity["date"], y=dd,
        mode="lines", name="Drawdown",
        line=dict(color=COLOR_NEGATIVE, width=1.4),
        fill="tozeroy", fillcolor="rgba(192,57,43,0.18)",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_FG, family="JetBrains Mono, monospace", size=11),
        showlegend=False, hovermode="x unified",
        xaxis=dict(gridcolor="#222"),
        yaxis=dict(gridcolor="#222", ticksuffix="%", zeroline=True,
                   zerolinecolor="#444"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Exposure gauge (Tab 1)
# ─────────────────────────────────────────────────────────────────────────────

def exposure_gauge(
    *,
    long_pct:   float,
    short_pct:  float,
    cash_pct:   float,
    net_pct:    float,
    gross_pct:  float,
    beta_proxy: Optional[float] = None,
) -> None:
    """Stacked horizontal bar (long/short/cash) + footer with net/gross/beta."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[long_pct],  y=["exposure"], orientation="h",
        name="Long",  marker_color=COLOR_POSITIVE,
        hovertemplate="Long %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[short_pct], y=["exposure"], orientation="h",
        name="Short", marker_color=COLOR_NEGATIVE,
        hovertemplate="Short %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[cash_pct],  y=["exposure"], orientation="h",
        name="Cash",  marker_color="#444",
        hovertemplate="Cash %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        height=110,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_FG, family="JetBrains Mono, monospace", size=11),
        legend=dict(orientation="h", y=-0.4),
        xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#222"),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    beta_str = "—" if beta_proxy is None else f"{beta_proxy:.2f}"
    st.markdown(
        f"<div style='font-family:monospace;color:{COLOR_MUTED};font-size:12px;'>"
        f"net {net_pct:+.0f}% · gross {gross_pct:.0f}% · β {beta_str}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STUBS for Tab 2…6 (replaced when each tab lands)
# ─────────────────────────────────────────────────────────────────────────────

def _stub(name: str) -> None:
    st.info(f"[stub] {name} — implementato quando atterra la sua tab.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Positions / Orders / Trade history
# ─────────────────────────────────────────────────────────────────────────────

def _color_pl(v: Any) -> str:
    """Cell-level color rule for signed numbers."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if x > 0:  return f"color:{COLOR_POSITIVE};"
    if x < 0:  return f"color:{COLOR_NEGATIVE};"
    return f"color:{COLOR_MUTED};"


def positions_table(df: pd.DataFrame) -> None:
    """Open positions enriched with SL/TP/days/tag."""
    if df is None or df.empty:
        empty_placeholder("No open positions.")
        return

    view = df.copy()
    # Pretty-format columns the styler will render as strings
    if "qty"            in view: view["qty"]            = view["qty"].map(lambda v: fmt_int(v))
    if "entry"          in view: view["entry"]          = view["entry"].map(lambda v: fmt_usd(v, 2))
    if "mark"           in view: view["mark"]           = view["mark"].map(lambda v: fmt_usd(v, 2))
    if "unrealized_pl"  in view: view["unrealized_pl"]  = view["unrealized_pl"].map(lambda v: fmt_usd(v, 2))
    if "unrealized_pct" in view: view["unrealized_pct"] = view["unrealized_pct"].map(lambda v: fmt_pct(v, 2, signed=True))
    if "dist_sl"        in view: view["dist_sl"]        = view["dist_sl"].map(lambda v: fmt_pct(v, 2, signed=True) if v is not None else "—")
    if "dist_tp"        in view: view["dist_tp"]        = view["dist_tp"].map(lambda v: fmt_pct(v, 2, signed=True) if v is not None else "—")
    if "days_held"      in view: view["days_held"]      = view["days_held"].map(lambda v: fmt_int(v) if v is not None else "—")
    if "tag"            in view: view["tag"]            = view["tag"].fillna("—")

    # Original numeric series for color rules
    pl_raw  = pd.to_numeric(df.get("unrealized_pl",  pd.Series()), errors="coerce")
    pct_raw = pd.to_numeric(df.get("unrealized_pct", pd.Series()), errors="coerce")

    def _style_row(row):
        styles = [""] * len(row)
        if "unrealized_pl" in row.index:
            i = list(row.index).index("unrealized_pl")
            styles[i] = _color_pl(pl_raw.get(row.name))
        if "unrealized_pct" in row.index:
            i = list(row.index).index("unrealized_pct")
            styles[i] = _color_pl(pct_raw.get(row.name))
        return styles

    styled = view.style.apply(_style_row, axis=1).set_properties(
        **{"font-family": "JetBrains Mono, monospace", "font-size": "12px"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def orders_table(df: pd.DataFrame) -> None:
    """Open / pending broker orders."""
    if df is None or df.empty:
        empty_placeholder("No open orders.")
        return

    view = df.copy()
    if "qty"        in view: view["qty"]        = view["qty"].map(lambda v: fmt_int(v))
    if "limit_price" in view: view["limit_price"] = view["limit_price"].map(lambda v: fmt_usd(v, 2) if v else "—")
    if "stop_price"  in view: view["stop_price"]  = view["stop_price"].map(lambda v: fmt_usd(v, 2) if v else "—")
    if "submitted_at" in view: view["submitted_at"] = view["submitted_at"].astype(str).str.slice(0, 19)

    st.dataframe(
        view.style.set_properties(**{"font-family": "JetBrains Mono, monospace", "font-size": "12px"}),
        use_container_width=True, hide_index=True,
    )


def trade_history_table(df: pd.DataFrame) -> None:
    """Closed trades — FIFO-paired realized P&L."""
    if df is None or df.empty:
        empty_placeholder("No closed trades in window.")
        return

    view = df.copy()
    pl_raw  = pd.to_numeric(view.get("realized_pl",  pd.Series()), errors="coerce")
    pct_raw = pd.to_numeric(view.get("realized_pct", pd.Series()), errors="coerce")

    if "qty"          in view: view["qty"]          = view["qty"].map(lambda v: fmt_int(v))
    if "entry_price"  in view: view["entry_price"]  = view["entry_price"].map(lambda v: fmt_usd(v, 2))
    if "exit_price"   in view: view["exit_price"]   = view["exit_price"].map(lambda v: fmt_usd(v, 2))
    if "realized_pl"  in view: view["realized_pl"]  = view["realized_pl"].map(lambda v: fmt_usd(v, 2))
    if "realized_pct" in view: view["realized_pct"] = view["realized_pct"].map(lambda v: fmt_pct(v, 2, signed=True))
    if "opened_at"    in view: view["opened_at"]    = view["opened_at"].astype(str).str.slice(0, 19)
    if "closed_at"    in view: view["closed_at"]    = view["closed_at"].astype(str).str.slice(0, 19)
    if "tag"          in view: view["tag"]          = view["tag"].fillna("—")

    def _style_row(row):
        styles = [""] * len(row)
        if "realized_pl" in row.index:
            i = list(row.index).index("realized_pl")
            styles[i] = _color_pl(pl_raw.get(row.name))
        if "realized_pct" in row.index:
            i = list(row.index).index("realized_pct")
            styles[i] = _color_pl(pct_raw.get(row.name))
        return styles

    styled = view.style.apply(_style_row, axis=1).set_properties(
        **{"font-family": "JetBrains Mono, monospace", "font-size": "12px"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def positions_summary_strip(summary: dict) -> None:
    """Compact header strip above the positions table.

    Renders 4 KPI cells: AUM, Cash, Open Positions, Buying Power.
    Expects the dict returned by ds.get_positions_summary() — always
    populated (zeroes when broker offline / flat), never empty.
    """
    summary = summary or {}
    aum    = summary.get("aum",          0.0) or 0.0
    cash   = summary.get("cash",         0.0) or 0.0
    bp     = summary.get("buying_power", 0.0) or 0.0
    n_pos  = int(summary.get("n_positions", 0) or 0)

    cells = (
        ("AUM",            fmt_usd(aum,  0)),
        ("Cash",           fmt_usd(cash, 0)),
        ("Open Positions", fmt_int(n_pos)),
        ("Buying Power",   fmt_usd(bp,   0)),
    )
    cols = st.columns(len(cells), gap="small")
    for col, (label, value) in zip(cols, cells):
        with col:
            st.markdown(
                f"<div style='border:1px solid {COLOR_MUTED}22;"
                f"border-radius:6px;padding:8px 12px;background:#161a22;"
                f"font-family:JetBrains Mono, monospace;'>"
                f"<div style='color:{COLOR_MUTED};font-size:10px;"
                f"text-transform:uppercase;letter-spacing:0.6px;'>{label}</div>"
                f"<div style='color:{COLOR_FG};font-size:18px;font-weight:700;"
                f"margin-top:2px;'>{value}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Agent Attribution
# ─────────────────────────────────────────────────────────────────────────────

def agent_scorecard(accuracy: pd.DataFrame, weights: pd.DataFrame) -> None:
    """Per-agent scorecard: accuracy, n_predictions, avg_confidence, weight."""
    if accuracy is None or accuracy.empty:
        empty_placeholder("No agent accuracy data — predictions/outcomes empty.")
        return

    df = accuracy.copy()
    if weights is not None and not weights.empty:
        wlu = weights.set_index("agent_id")[["weight_avg","weight_trend"]]
        df = df.merge(wlu, left_on="agent_id", right_index=True, how="left")
    else:
        df["weight_avg"]   = None
        df["weight_trend"] = None

    pretty = pd.DataFrame({
        "Agent":       df["agent_name"],
        "Accuracy":    df["accuracy"].map(lambda v: fmt_pct(v, 1)),
        "Predictions": df["n_predictions"].map(fmt_int),
        "Confidence":  df["avg_confidence"].map(lambda v: fmt_pct(v, 1)),
        "Weight":      df["weight_avg"].map(lambda v: fmt_ratio(v, 3) if v is not None else "—"),
        "Trend 30d":   df["weight_trend"].map(
            lambda v: f"{v:+.3f}" if v is not None and not pd.isna(v) else "—"),
    })

    acc_raw   = pd.to_numeric(df["accuracy"],     errors="coerce")
    trend_raw = pd.to_numeric(df["weight_trend"], errors="coerce")

    def _style(row):
        styles = [""] * len(row)
        cols = list(row.index)
        if "Accuracy"  in cols: styles[cols.index("Accuracy")]  = (
            f"color:{COLOR_POSITIVE};" if (acc_raw.iloc[row.name] or 0) >= 50
            else f"color:{COLOR_NEGATIVE};"
        )
        if "Trend 30d" in cols:
            t = trend_raw.iloc[row.name]
            if pd.notna(t):
                styles[cols.index("Trend 30d")] = (
                    f"color:{COLOR_POSITIVE};" if t > 0
                    else f"color:{COLOR_NEGATIVE};" if t < 0
                    else f"color:{COLOR_MUTED};"
                )
        return styles

    styled = pretty.style.apply(_style, axis=1).set_properties(
        **{"font-family": "JetBrains Mono, monospace", "font-size": "12px"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def contribution_heatmap(df: pd.DataFrame) -> None:
    """Agent × Ticker heatmap of signed weighted contribution (basis points)."""
    if df is None or df.empty:
        empty_placeholder("No attribution data — outcomes table empty.")
        return

    pivot = (df.pivot_table(index="agent_name", columns="ticker",
                            values="contribution_bps", aggfunc="sum")
               .fillna(0.0))
    if pivot.empty:
        empty_placeholder("No attribution data after pivot.")
        return

    pivot = pivot.loc[pivot.abs().sum(axis=1).sort_values(ascending=True).index]
    abs_max = float(pivot.abs().to_numpy().max() or 1.0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale=[
            [0.0, COLOR_NEGATIVE],
            [0.5, "#1a1d23"],
            [1.0, COLOR_POSITIVE],
        ],
        zmin=-abs_max, zmax=abs_max, zmid=0.0,
        colorbar=dict(title=dict(text="bps", font=dict(color=COLOR_MUTED, size=10)),
                      tickfont=dict(color=COLOR_MUTED, size=10)),
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z:+.1f} bps<extra></extra>",
    ))
    fig.update_layout(
        height=max(260, 26 * len(pivot.index)),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_FG, family="JetBrains Mono, monospace", size=11),
        xaxis=dict(side="top", tickfont=dict(color=COLOR_FG)),
        yaxis=dict(tickfont=dict(color=COLOR_FG), automargin=True),
    )
    st.plotly_chart(fig, use_container_width=True)


def weight_trend_plot(df: pd.DataFrame, *, height: int = 300) -> None:
    """One line per agent — daily mean weight over time."""
    if df is None or df.empty:
        empty_placeholder("No weight history.")
        return

    fig = go.Figure()
    palette = (COLOR_ACCENT, "#e8a83a", COLOR_POSITIVE, COLOR_NEGATIVE,
               "#9b59b6", "#3498db", "#1abc9c", "#e67e22",
               "#95a5a6", "#f1c40f", "#16a085", "#d35400")

    for i, (agent_name, sub) in enumerate(df.groupby("agent_name")):
        sub = sub.sort_values("date")
        fig.add_trace(go.Scatter(
            x=sub["date"], y=sub["weight"], mode="lines",
            name=agent_name, line=dict(color=palette[i % len(palette)], width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>" + agent_name + "<br>w=%{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_FG, family="JetBrains Mono, monospace", size=11),
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
        hovermode="x unified",
        xaxis=dict(gridcolor="#222", type="date"),
        yaxis=dict(gridcolor="#222", tickformat=".2f", title=dict(text="weight",
                                                                  font=dict(color=COLOR_MUTED))),
    )
    fig.update_xaxes(tickformat="%b %d", dtick="D1")
    st.plotly_chart(fig, use_container_width=True)


def struggling_agents_card(df: pd.DataFrame) -> None:
    """Highlight agents with accuracy < 50% and ≥ N predictions."""
    if df is None or df.empty:
        st.markdown(
            f"<div style='padding:10px 14px;border-left:3px solid {COLOR_POSITIVE};"
            f"background:#161a22;font-family:JetBrains Mono,monospace;font-size:12px;"
            f"color:{COLOR_FG};'>✓ No struggling agents — all above 50% accuracy."
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"<tr><td style='padding:4px 8px;color:{COLOR_FG};'>{r['agent_name']}</td>"
            f"<td style='padding:4px 8px;color:{COLOR_NEGATIVE};text-align:right;'>"
            f"{fmt_pct(r['accuracy'], 1)}</td>"
            f"<td style='padding:4px 8px;color:{COLOR_MUTED};text-align:right;'>"
            f"{fmt_int(r['n_predictions'])}</td>"
            f"<td style='padding:4px 8px;color:{COLOR_MUTED};text-align:right;'>"
            f"{fmt_ratio(r['weight_avg'], 3)}</td></tr>"
        )
    st.markdown(
        f"<div style='border-left:3px solid {COLOR_NEGATIVE};background:#161a22;"
        f"padding:8px 14px;font-family:JetBrains Mono,monospace;'>"
        f"<div style='color:{COLOR_NEGATIVE};font-size:11px;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin-bottom:6px;'>⚠ Struggling agents</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
        f"<thead><tr style='color:{COLOR_MUTED};border-bottom:1px solid #222;'>"
        f"<th style='padding:4px 8px;text-align:left;'>Agent</th>"
        f"<th style='padding:4px 8px;text-align:right;'>Accuracy</th>"
        f"<th style='padding:4px 8px;text-align:right;'>N</th>"
        f"<th style='padding:4px 8px;text-align:right;'>Weight</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Backtest
# ─────────────────────────────────────────────────────────────────────────────

import datetime as _dt


def backtest_form(*, default_label: str = "run") -> dict:
    """Render the backtest config form.

    Returns a dict with the user's selections (label/start/end/tickers/...).
    The "submitted" key is True iff the user clicked Launch this rerun.
    """
    today = _dt.date.today()

    with st.form("backtest_form", clear_on_submit=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            label    = st.text_input("Label", value=default_label,
                                     help="Used for output filenames + log path.")
            tickers  = st.text_input("Tickers", value="all",
                                     help="Comma-separated symbols, or 'all' for config/tickers.yaml.")
            capital  = st.number_input("Capital ($)", min_value=1_000.0,
                                       value=100_000.0, step=10_000.0, format="%.0f")
            no_cache = st.checkbox("Disable state cache (--no-cache)", value=False)
        with c2:
            walk_forward = st.checkbox("Walk-forward (IS + OOS)", value=False,
                                       help="Splits the window into in-sample / out-of-sample halves.")
            if walk_forward:
                is_start  = st.date_input("IS start",  value=today.replace(year=today.year - 4),
                                          key="bt_is_start")
                is_end    = st.date_input("IS end",    value=today.replace(year=today.year - 2),
                                          key="bt_is_end")
                oos_start = st.date_input("OOS start", value=today.replace(year=today.year - 2),
                                          key="bt_oos_start")
                oos_end   = st.date_input("OOS end",   value=today, key="bt_oos_end")
                start = end = None
            else:
                start = st.date_input("Start", value=today.replace(year=today.year - 1),
                                      key="bt_start")
                end   = st.date_input("End",   value=today, key="bt_end")
                is_start = is_end = oos_start = oos_end = None

        submitted = st.form_submit_button("▶  Launch backtest",
                                          use_container_width=True, type="primary")

    return {
        "submitted":    bool(submitted),
        "label":        label,
        "tickers":      tickers,
        "capital":      float(capital),
        "no_cache":     bool(no_cache),
        "walk_forward": bool(walk_forward),
        "start":        start.isoformat()      if start      else None,
        "end":          end.isoformat()        if end        else None,
        "is_start":     is_start.isoformat()   if is_start   else None,
        "is_end":       is_end.isoformat()     if is_end     else None,
        "oos_start":    oos_start.isoformat()  if oos_start  else None,
        "oos_end":      oos_end.isoformat()    if oos_end    else None,
    }


_STATUS_COLORS = {
    "running":   COLOR_ACCENT,
    "done":      COLOR_POSITIVE,
    "error":     COLOR_NEGATIVE,
    "cancelled": COLOR_WARN,
    "stale":     COLOR_MUTED,
}


def backtest_progress_bar(progress: dict, log_lines: Optional[list[str]] = None) -> None:
    """Top: status badge + meta. Bottom: live tail of the subprocess log."""
    if not progress:
        return
    status = str(progress.get("status") or "—")
    color  = _STATUS_COLORS.get(status, COLOR_MUTED)
    pid    = progress.get("pid") or "—"
    label  = progress.get("label") or "—"
    started = (progress.get("started_at") or "")[:19].replace("T", " ")

    st.markdown(
        f"<div style='border:1px solid #222;border-radius:6px;padding:10px 14px;"
        f"background:#161a22;font-family:JetBrains Mono,monospace;font-size:12px;'>"
        f"<span style='background:{color};color:#0e1117;padding:2px 8px;border-radius:4px;"
        f"font-weight:700;text-transform:uppercase;letter-spacing:0.6px;'>"
        f"{status}</span>"
        f"<span style='color:{COLOR_MUTED};margin-left:14px;'>label</span> "
        f"<span style='color:{COLOR_FG};'>{label}</span>"
        f"<span style='color:{COLOR_MUTED};margin-left:14px;'>pid</span> "
        f"<span style='color:{COLOR_FG};'>{pid}</span>"
        f"<span style='color:{COLOR_MUTED};margin-left:14px;'>started</span> "
        f"<span style='color:{COLOR_FG};'>{started}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if status == "running":
        st.progress(0.0)        # indeterminate-style — engine emits no %

    if log_lines:
        body = "\n".join(log_lines[-80:])
        st.code(body, language="log")
    else:
        st.caption("No log output yet.")


def _metric_row(metrics: dict) -> None:
    """Compact metric strip used by the results panel + saved-runs detail."""
    if not metrics:
        empty_placeholder("Metrics block missing in report.")
        return

    pairs = [
        ("Total return",  metrics.get("total_return"),  "pct"),
        ("CAGR",          metrics.get("cagr"),          "pct"),
        ("Sharpe",        metrics.get("sharpe"),        "ratio"),
        ("Sortino",       metrics.get("sortino"),       "ratio"),
        ("Max DD",        metrics.get("max_drawdown"),  "pct"),
        ("Win rate",      metrics.get("win_rate"),      "pct"),
        ("Profit factor", metrics.get("profit_factor"), "ratio"),
        ("Trades",        metrics.get("n_trades"),      "int"),
    ]
    cols = st.columns(len(pairs), gap="small")
    for col, (label, value, kind) in zip(cols, pairs):
        with col:
            if value is None:
                txt = "—"
            elif kind == "pct":
                txt = fmt_pct(float(value) * 100.0 if abs(float(value)) <= 1.0 else float(value),
                              2, signed=True)
            elif kind == "ratio":
                txt = fmt_ratio(value, 2)
            else:
                txt = fmt_int(value)
            st.metric(label, txt)


def backtest_results_panel(report: dict, *, on_save=None) -> None:
    """Render a finished backtest report.

    `on_save` is an optional callable taking no args; when provided, a
    "Save run" button is shown that invokes it.
    """
    if not report:
        empty_placeholder("Open a saved run from the list to inspect its metrics.")
        return

    head = (f"**{report.get('label') or '—'}** · "
            f"{report.get('start') or '?'} → {report.get('end') or '?'}")
    st.markdown(head)

    metrics = (report.get("metrics") or report.get("performance") or report)
    _metric_row(metrics)

    # Walk-forward dual block
    if "is" in report and "oos" in report:
        st.markdown("&nbsp;")
        c_is, c_oos = st.columns(2)
        with c_is:
            st.markdown("**In-sample**")
            _metric_row(report["is"].get("metrics") or report["is"])
        with c_oos:
            st.markdown("**Out-of-sample**")
            _metric_row(report["oos"].get("metrics") or report["oos"])
        if report.get("sharpe_decay") is not None:
            st.caption(f"Sharpe decay (IS − OOS): {fmt_ratio(report.get('sharpe_decay'), 3)}  "
                       f"·  Return retention: {fmt_ratio(report.get('return_retention'), 3)}")

    if on_save is not None:
        st.markdown("&nbsp;")
        if st.button("💾  Save run to backtests/runs/", key="bt_save_run"):
            on_save()


def saved_runs_list(runs: list[dict]) -> Optional[dict]:
    """Render the runs index. Returns the selected run dict, or None."""
    if not runs:
        empty_placeholder("No saved backtests yet — launch one above.")
        return None

    options = [f"[{r['source']}] {r['name']}  ·  {r['modified']}  ·  {r['size_kb']} kB"
               for r in runs]
    idx = st.radio("Runs", options=options, key="bt_runs_radio",
                   label_visibility="collapsed")
    return runs[options.index(idx)] if idx else None

# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Regime & Risk
# ─────────────────────────────────────────────────────────────────────────────

def macro_strip(snapshot: dict) -> None:
    """4-cell macro header: VIX (+ sparkline), Regime, 10Y, 3M·spread."""
    snapshot = snapshot or {}
    vix      = snapshot.get("vix")
    trend    = snapshot.get("vix_trend") or []
    regime   = snapshot.get("regime")        or "UNKNOWN"
    color    = snapshot.get("regime_color")  or COLOR_MUTED
    ust_10y  = snapshot.get("ust_10y")
    ust_3m   = snapshot.get("ust_3m")
    spread   = snapshot.get("curve_spread")
    label    = snapshot.get("curve_label")   or "—"

    cols = st.columns(4, gap="small")
    with cols[0]:
        st.metric("VIX", fmt_ratio(vix, 2))
        if trend:
            sp = go.Figure(go.Scatter(x=list(range(len(trend))), y=trend,
                                       mode="lines", line=dict(color=COLOR_ACCENT, width=1.4)))
            sp.update_layout(
                height=60, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, showlegend=False,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
            )
            st.plotly_chart(sp, use_container_width=True)
    with cols[1]:
        st.markdown(
            f"<div style='border:1px solid #222;border-radius:6px;padding:12px 14px;"
            f"background:#161a22;font-family:JetBrains Mono,monospace;'>"
            f"<div style='color:{COLOR_MUTED};font-size:11px;text-transform:uppercase;"
            f"letter-spacing:0.6px;'>Regime</div>"
            f"<div style='color:{color};font-size:22px;font-weight:700;margin-top:4px;'>"
            f"{regime}</div></div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.metric("UST 10Y", fmt_pct(ust_10y, 2) if ust_10y is not None else "—")
    with cols[3]:
        st.metric("3M · Spread",
                  f"{fmt_pct(ust_3m, 2)}  ·  {label}" if ust_3m is not None else "—",
                  delta=f"{spread:+.0f} bps" if spread is not None else None,
                  delta_color="normal" if (spread or 0) >= 0 else "inverse")


def correlation_heatmap(df: pd.DataFrame) -> None:
    """Symmetric return-correlation heatmap. Empty → soft placeholder."""
    if df is None or df.empty or df.shape[0] < 2:
        empty_placeholder("Correlation unavailable — need ≥2 positions with price history.")
        return

    fig = go.Figure(data=go.Heatmap(
        z=df.values, x=list(df.columns), y=list(df.index),
        colorscale=[
            [0.0, COLOR_NEGATIVE],
            [0.5, "#1a1d23"],
            [1.0, COLOR_POSITIVE],
        ],
        zmin=-1.0, zmax=1.0, zmid=0.0,
        colorbar=dict(title=dict(text="ρ", font=dict(color=COLOR_MUTED, size=10)),
                      tickfont=dict(color=COLOR_MUTED, size=10)),
        hovertemplate="%{y} · %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(260, 30 * len(df.index)),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_FG, family="JetBrains Mono, monospace", size=11),
        xaxis=dict(side="top"), yaxis=dict(automargin=True),
    )
    st.plotly_chart(fig, use_container_width=True)


def var_card(d: dict) -> None:
    """Single-card VaR display. Expects {var_usd, var_pct, method, lookback_days}."""
    d = d or {}
    st.markdown(
        f"<div style='border:1px solid #222;border-radius:6px;padding:14px 16px;"
        f"background:#161a22;font-family:JetBrains Mono,monospace;'>"
        f"<div style='color:{COLOR_MUTED};font-size:11px;text-transform:uppercase;"
        f"letter-spacing:0.6px;'>1-day VaR · 95% · {d.get('method','—')}</div>"
        f"<div style='color:{COLOR_NEGATIVE};font-size:26px;font-weight:700;"
        f"margin-top:4px;'>−{fmt_usd(d.get('var_usd'), 0).lstrip('-')}</div>"
        f"<div style='color:{COLOR_MUTED};font-size:12px;margin-top:2px;'>"
        f"−{fmt_pct(d.get('var_pct'), 2)} · "
        f"{d.get('n_returns', 0)} returns / {d.get('lookback_days', 0)}d lookback</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def stress_table(df: pd.DataFrame) -> None:
    """Static linear stress scenarios — table with worst-3 positions per row."""
    if df is None or df.empty:
        empty_placeholder("No stress data — open positions required.")
        return

    rows_html = []
    for _, r in df.iterrows():
        d_usd = r.get("delta_equity_usd") or 0.0
        d_pct = r.get("delta_equity_pct") or 0.0
        color = COLOR_POSITIVE if d_usd >= 0 else COLOR_NEGATIVE
        worst = r.get("worst_positions") or []
        worst_str = "  ·  ".join(
            f"{w['ticker']} {fmt_usd(w['delta_usd'], 0)}" for w in worst[:3]
        ) or "—"
        rows_html.append(
            f"<tr>"
            f"<td style='padding:6px 10px;color:{COLOR_FG};'>{r['scenario']}</td>"
            f"<td style='padding:6px 10px;text-align:right;color:{color};'>"
            f"{fmt_usd(d_usd, 0)}</td>"
            f"<td style='padding:6px 10px;text-align:right;color:{color};'>"
            f"{fmt_pct(d_pct, 2, signed=True)}</td>"
            f"<td style='padding:6px 10px;color:{COLOR_MUTED};font-size:11px;'>"
            f"{worst_str}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<div style='border:1px solid #222;border-radius:6px;background:#161a22;"
        f"font-family:JetBrains Mono,monospace;'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
        f"<thead><tr style='color:{COLOR_MUTED};border-bottom:1px solid #222;'>"
        f"<th style='padding:6px 10px;text-align:left;'>Scenario</th>"
        f"<th style='padding:6px 10px;text-align:right;'>Δ USD</th>"
        f"<th style='padding:6px 10px;text-align:right;'>Δ %</th>"
        f"<th style='padding:6px 10px;text-align:left;'>Worst 3 positions</th>"
        f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Health & Audit
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_runs_table(df: pd.DataFrame) -> None:
    """Last-24h pipeline runs."""
    if df is None or df.empty:
        empty_placeholder("No pipeline runs in the last 24 hours.")
        return

    view = df.copy()
    if "started_at"  in view: view["started_at"]  = view["started_at"].astype(str).str.slice(0, 19)
    if "finished_at" in view: view["finished_at"] = view["finished_at"].astype(str).str.slice(0, 19)
    if "duration_s"  in view: view["duration_s"]  = view["duration_s"].map(
        lambda v: f"{v:.1f}s" if pd.notna(v) else "—")
    if "run_id"      in view: view["run_id"]      = view["run_id"].astype(str).str.slice(0, 8)

    status_raw = view.get("status", pd.Series(dtype=str)).astype(str)
    def _style_row(row):
        styles = [""] * len(row)
        s = str(status_raw.iloc[row.name]).upper()
        cols = list(row.index)
        if "status" in cols:
            i = cols.index("status")
            styles[i] = (f"color:{COLOR_POSITIVE};" if s == "OK"
                         else f"color:{COLOR_NEGATIVE};" if s in ("ERROR","FAILED")
                         else f"color:{COLOR_MUTED};")
        return styles

    styled = view.style.apply(_style_row, axis=1).set_properties(
        **{"font-family": "JetBrains Mono, monospace", "font-size": "12px"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def email_status_card(d: dict) -> None:
    d = d or {}
    pre  = int(d.get("premarket_sent",  0) or 0)
    post = int(d.get("postmarket_sent", 0) or 0)
    crit = int(d.get("critical_sent",   0) or 0)
    def _dot(n, want=1, color_good=COLOR_POSITIVE, color_bad=COLOR_WARN):
        if n >= want: return f"<span style='color:{color_good};'>● sent</span> ({n})"
        return f"<span style='color:{color_bad};'>○ none</span>"
    st.markdown(
        f"<div style='border:1px solid #222;border-radius:6px;padding:12px 16px;"
        f"background:#161a22;font-family:JetBrains Mono,monospace;font-size:12px;'>"
        f"<div style='color:{COLOR_MUTED};font-size:11px;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin-bottom:8px;'>Email · today {d.get('date','')}</div>"
        f"<div>Premarket   &nbsp;{_dot(pre)}</div>"
        f"<div>Postmarket  &nbsp;{_dot(post)}</div>"
        f"<div>Critical    &nbsp;{_dot(crit, want=0, color_good=COLOR_POSITIVE if crit==0 else COLOR_NEGATIVE)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def alpaca_metrics_card(d: dict) -> None:
    d = d or {}
    p50 = d.get("p50_ms"); p95 = d.get("p95_ms")
    err = int(d.get("errors") or 0); req = int(d.get("requests") or 0)
    last = d.get("last_call_at") or "—"
    err_color = COLOR_NEGATIVE if err > 0 else COLOR_POSITIVE
    st.markdown(
        f"<div style='border:1px solid #222;border-radius:6px;padding:12px 16px;"
        f"background:#161a22;font-family:JetBrains Mono,monospace;font-size:12px;'>"
        f"<div style='color:{COLOR_MUTED};font-size:11px;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin-bottom:8px;'>Alpaca · 24h</div>"
        f"<div>requests &nbsp;<span style='color:{COLOR_FG};'>{req}</span> &nbsp;·&nbsp; "
        f"errors <span style='color:{err_color};'>{err}</span></div>"
        f"<div>p50 <span style='color:{COLOR_FG};'>"
        f"{fmt_ratio(p50, 0) + ' ms' if p50 is not None else '—'}</span> &nbsp;·&nbsp; "
        f"p95 <span style='color:{COLOR_FG};'>"
        f"{fmt_ratio(p95, 0) + ' ms' if p95 is not None else '—'}</span></div>"
        f"<div style='color:{COLOR_MUTED};margin-top:4px;'>last call {last}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def llm_cost_card(d: dict) -> None:
    d = d or {}
    a = d.get("anthropic_usd"); o = d.get("openai_usd"); proj = d.get("projected_usd")
    tin = int(d.get("tokens_in")  or 0); tout = int(d.get("tokens_out") or 0)
    calls = int(d.get("calls") or 0)
    st.markdown(
        f"<div style='border:1px solid #222;border-radius:6px;padding:12px 16px;"
        f"background:#161a22;font-family:JetBrains Mono,monospace;font-size:12px;'>"
        f"<div style='color:{COLOR_MUTED};font-size:11px;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin-bottom:8px;'>LLM cost · MTD</div>"
        f"<div>Anthropic <span style='color:{COLOR_FG};'>"
        f"{fmt_usd(a, 2) if a is not None else '—'}</span> &nbsp;·&nbsp; "
        f"OpenAI <span style='color:{COLOR_FG};'>"
        f"{fmt_usd(o, 2) if o is not None else '—'}</span></div>"
        f"<div>Projected EOM <span style='color:{COLOR_ACCENT};'>"
        f"{fmt_usd(proj, 2) if proj is not None else '—'}</span></div>"
        f"<div style='color:{COLOR_MUTED};margin-top:4px;'>"
        f"{calls} calls · {tin:,} in / {tout:,} out tokens</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def audit_table(df: pd.DataFrame) -> None:
    """Composed event stream (pipeline + orders + monitor)."""
    if df is None or df.empty:
        empty_placeholder("Audit trail empty.")
        return

    view = df.copy()
    if "ts" in view: view["ts"] = view["ts"].astype(str).str.slice(0, 19)
    st.dataframe(
        view.style.set_properties(**{"font-family": "JetBrains Mono, monospace",
                                     "font-size": "12px"}),
        use_container_width=True, hide_index=True,
    )


def kill_switch_button(*, armed: bool) -> Optional[bool]:
    """Tab-6 kill button. Returns True if user clicked ARM, False if DISARM,
       None if no action. Wired live: caller persists via data_sources."""
    label = "🛑  EMERGENCY KILL  🛑" if not armed else "↻  DISARM KILL SWITCH"
    bg    = COLOR_KILL if not armed else "#444"
    st.markdown(
        f"<style>div[data-testid='stButton'] button:has(div:contains('{label}'))"
        f"{{background-color:{bg};color:#fff;font-weight:700;height:64px;"
        f"border:2px solid #fff;font-size:18px;}}</style>",
        unsafe_allow_html=True,
    )
    if st.button(label, key="kill_switch_btn", use_container_width=True):
        return not armed   # True = arming now, False = disarming now
    return None
