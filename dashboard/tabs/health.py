"""
dashboard/tabs/health.py
────────────────────────
Tab 6 — Health & Audit + Emergency Kill switch.

Composition (top → bottom):
  1. Status row             — pipeline runs (24h) | email status | Alpaca | LLM cost
  2. Pipeline runs table    — full row-level view
  3. Audit trail            — composed events (pipeline + orders + monitor)
  4. EMERGENCY KILL panel   — armed-state badge + arm/disarm button + reason input

Writes are explicit and confirmed:
  - arm  → writes `.athanor_kill` JSON  (ds.arm_kill_switch)
  - disarm → unlinks `.athanor_kill`     (ds.disarm_kill_switch)
"""

from __future__ import annotations

import streamlit as st


def render(ds, c) -> None:
    # ── Status row ─────────────────────────────────────────────────────────
    st.markdown("**Operational status**")

    runs   = ds.get_pipeline_runs_24h()
    n_runs = 0 if runs is None or runs.empty else int(len(runs))
    n_err  = 0 if runs is None or runs.empty else int(
        (runs["status"].astype(str).str.upper().isin(["ERROR","FAILED"])).sum()
    )

    cols = st.columns(4, gap="small")
    with cols[0]:
        st.metric("Pipeline runs · 24h", str(n_runs),
                  delta=f"{n_err} errors" if n_err else None,
                  delta_color="inverse" if n_err else "off")
    with cols[1]:
        c.email_status_card(ds.get_email_status_today())
    with cols[2]:
        c.alpaca_metrics_card(ds.get_alpaca_metrics_24h())
    with cols[3]:
        c.llm_cost_card(ds.get_llm_cost_month())

    # ── Pipeline runs table ────────────────────────────────────────────────
    st.markdown("&nbsp;")
    st.markdown("**Pipeline runs · last 24h**")
    c.pipeline_runs_table(runs)

    # ── Audit trail ────────────────────────────────────────────────────────
    st.markdown("&nbsp;")

    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown("**Audit trail** · pipeline + orders + monitor (latest first)")
    with head_right:
        limit = st.selectbox(
            "Limit",
            options=(25, 50, 100, 200),
            index=1,
            key="audit_limit",
            label_visibility="collapsed",
        )
    c.audit_table(ds.get_audit_trail(limit=int(limit)))

    # ── Circuit Breaker status ─────────────────────────────────────────────
    st.markdown("&nbsp;")
    st.markdown("**Circuit Breakers · current status**")

    cb_statuses = ds.get_circuit_breaker_status()
    if cb_statuses:
        cb_cols = st.columns(5, gap="small")
        cb_meta = {
            "CB1": ("Daily P&L < -3%",    "Halts new OPEN orders"),
            "CB2": ("Position loss > 8%",  "Force-closes that position"),
            "CB3": ("VIX > 35",            "Halts new OPEN orders"),
            "CB4": ("Reject rate > 50%",   "Soft alert only"),
            "CB5": ("Equity drawdown >15%","Halts ALL new orders"),
        }
        for col, s in zip(cb_cols, cb_statuses):
            with col:
                triggered = getattr(s, "triggered", False)
                cb_id     = getattr(s, "cb_id", "?")
                reason    = getattr(s, "reason", "")
                desc, remedy = cb_meta.get(cb_id, ("", ""))
                color  = "#ff3b3b" if triggered else "#27ae60"
                label  = "TRIGGERED" if triggered else "OK"
                st.markdown(
                    f"<div style='border:1px solid #333;border-radius:6px;padding:10px;"
                    f"background:#161a22;font-family:JetBrains Mono,monospace;font-size:11px;'>"
                    f"<b style='color:#ccc;'>{cb_id}</b><br>"
                    f"<span style='background:{color};color:#0e1117;padding:1px 8px;"
                    f"border-radius:3px;font-weight:700;font-size:10px;'>{label}</span><br>"
                    f"<span style='color:#888;font-size:10px;'>{desc}</span><br>"
                    f"<span style='color:#666;font-size:9px;'>{reason[:60]}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        any_triggered = any(getattr(s, "triggered", False) for s in cb_statuses)
        if any_triggered:
            st.warning(
                "One or more circuit breakers are TRIGGERED. "
                "CB1/CB3/CB5 → OPEN orders halted. CB2 → position auto-closed. "
                "CB4 → review broker connectivity. Reset via: `python -c \"from src.risk.circuit_breakers import reset_cb; reset_cb('cb1')\"`"
            )
    else:
        st.info("Circuit breaker data unavailable (adapter not connected).")

    # ── EMERGENCY KILL panel ───────────────────────────────────────────────
    st.markdown("&nbsp;")
    st.markdown("**Emergency kill switch**")

    armed = ds.is_kill_switch_active()
    badge_color = "#ff3b3b" if armed else "#27ae60"
    badge_text  = "ARMED" if armed else "DISARMED"
    st.markdown(
        f"<div style='border:1px solid #222;border-radius:6px;padding:14px 16px;"
        f"background:#161a22;font-family:JetBrains Mono,monospace;font-size:12px;'>"
        f"<span style='background:{badge_color};color:#0e1117;padding:2px 10px;"
        f"border-radius:4px;font-weight:700;letter-spacing:0.6px;'>{badge_text}</span>"
        f"&nbsp;&nbsp;Live execution is "
        f"<b style='color:{badge_color};'>"
        f"{'BLOCKED' if armed else 'allowed'}</b> while this badge is shown."
        f"</div>",
        unsafe_allow_html=True,
    )

    if not armed:
        with st.form("kill_arm_form", clear_on_submit=True):
            reason = st.text_input(
                "Reason (recorded in `.athanor_kill`)",
                value="armed via dashboard",
                max_chars=200,
            )
            confirm = st.checkbox(
                "I understand this BLOCKS all live execution until disarmed.",
                value=False,
            )
            submitted = st.form_submit_button("🛑  ARM KILL SWITCH",
                                              type="primary",
                                              use_container_width=True)
        if submitted:
            if not confirm:
                st.error("Confirm the checkbox before arming.")
            elif ds.arm_kill_switch(reason=reason or "armed via dashboard"):
                st.success(".athanor_kill written — kill switch ARMED.")
                st.rerun()
            else:
                st.error("Failed to write .athanor_kill (see logs).")
    else:
        if st.button("↻  DISARM KILL SWITCH", use_container_width=True):
            if ds.disarm_kill_switch():
                st.success(".athanor_kill removed — kill switch DISARMED.")
                st.rerun()
            else:
                st.error("Failed to remove .athanor_kill (see logs).")
