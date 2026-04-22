"""
dashboard/tabs/backtest.py
──────────────────────────
Tab 4 — Backtest launcher + results browser.

Composition (top → bottom):
  1. Config form                — date range / tickers / capital / walk-forward
  2. Active run panel           — status badge + log tail (auto-refresh while live)
  3. Saved runs list | Report   — pick a run on the left, inspect on the right
  4. Save Run button            — copy a `results/` JSON into `backtests/runs/`

All subprocess control lives in `ds`. Tab is read-mostly; the only writes
are: launch_backtest(), cancel_backtest(), save_backtest_run().
"""

from __future__ import annotations

import streamlit as st


def _maybe_autorefresh(interval_ms: int = 2_000) -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_ms, key="backtest_autorefresh")
    except Exception:
        pass


def render(ds, c) -> None:
    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        "**Backtest** · "
        "<span style='color:#8a8f98;font-size:11px;'>"
        "spawns `python -m src.backtesting.cli` as a subprocess; reports "
        "saved under `results/`</span>",
        unsafe_allow_html=True,
    )

    # ── Config form ────────────────────────────────────────────────────────
    st.markdown("&nbsp;")
    st.markdown("**Configure run**")
    form = c.backtest_form(default_label=st.session_state.get("bt_active_label", "run"))

    if form["submitted"]:
        try:
            progress = ds.launch_backtest(form)
            st.session_state["bt_active_label"] = progress["label"]
            st.success(f"Launched backtest `{progress['label']}` (pid {progress['pid']}).")
            st.rerun()
        except Exception as exc:
            st.error(f"Launch failed: {exc}")

    # ── Active run panel ───────────────────────────────────────────────────
    active_label = st.session_state.get("bt_active_label")
    if active_label:
        st.markdown("&nbsp;")
        st.markdown(f"**Active run** · `{active_label}`")

        progress = ds.get_backtest_progress(active_label)

        # If the subprocess died but progress still says 'running', stamp it.
        if progress.get("status") == "stale":
            progress = ds.reconcile_finished_backtest(active_label)

        log_lines = ds.tail_backtest_log(active_label, n=80)
        c.backtest_progress_bar(progress, log_lines)

        ctrl_left, ctrl_right = st.columns([1, 1])
        with ctrl_left:
            if progress.get("status") == "running":
                if st.button("⏹  Cancel run", key="bt_cancel"):
                    if ds.cancel_backtest(active_label):
                        st.warning(f"Sent SIGTERM to pid {progress.get('pid')}.")
                        st.rerun()
                    else:
                        st.error("Cancel failed — process already gone?")
        with ctrl_right:
            if st.button("Clear active run", key="bt_clear_active"):
                st.session_state.pop("bt_active_label", None)
                st.rerun()

        # Auto-refresh only while the run is live.
        if progress.get("status") == "running":
            _maybe_autorefresh(2_000)

    # ── Saved runs ─────────────────────────────────────────────────────────
    st.markdown("&nbsp;")
    st.markdown("**Saved runs**")
    runs = ds.list_saved_backtests()

    left, right = st.columns([1, 2], gap="medium")
    with left:
        selected = c.saved_runs_list(runs)
    with right:
        if selected is None:
            c.empty_placeholder("Pick a run on the left to view its report.")
        else:
            report = ds.load_backtest_report(selected["path"])

            def _save():
                try:
                    dst = ds.save_backtest_run(selected["path"])
                    st.success(f"Saved → `{dst}`")
                except Exception as exc:
                    st.error(f"Save failed: {exc}")

            on_save = _save if selected.get("source") == "live" else None
            c.backtest_results_panel(report, on_save=on_save)

    # ── Footer note ────────────────────────────────────────────────────────
    st.caption(
        "Reports auto-saved by the CLI live in `results/`. Click "
        "**Save run** on a live report to copy it into `backtests/runs/` "
        "(survives `results/` cleanup)."
    )
