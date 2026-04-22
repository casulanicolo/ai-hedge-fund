"""
dashboard/app.py
────────────────
Thin router. No business logic — only:
  1. page config + CSS injection
  2. header strip + sidebar (mode, refresh, kill-armed banner)
  3. st.tabs(...) → delegates each tab body to dashboard/tabs/<name>.render(ds, c)

Every tab module exposes ONE entry point:
    def render(ds, c) -> None
where `ds` is the dashboard.data_sources module and `c` is dashboard.components.

Missing tab modules degrade gracefully (placeholder banner) so the router
keeps loading even mid-build.

Legacy v1 of this file is preserved at dashboard/app_v1_legacy.py until
Phase 6 ships and parity has been verified.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import streamlit as st

# Make project root importable when launched as `streamlit run dashboard/app.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import components as c          # noqa: E402
from dashboard import data_sources as ds       # noqa: E402

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Athanor Alpha",
    page_icon="🜂",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Theme: inject athanor.css once
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    """Standard Streamlit CSS injection: read athanor.css and inject as
       a <style> block via st.markdown(unsafe_allow_html=True). Must be
       called at the top of the script, before st.tabs()."""
    css_path = Path(__file__).parent / "static" / "athanor.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        log.warning("athanor.css missing at %s — using Streamlit defaults",
                    css_path)


_inject_css()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — refresh + light/dark + nav mirror
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🜂 Athanor Alpha")
    st.caption("Multi-Agent Trading · v2")
    st.divider()

    if st.button("🔄  Refresh data", use_container_width=True):
        ds.clear_all_caches()
        st.rerun()

    theme = st.radio(
        "Theme",
        options=("Dark", "Light"),
        horizontal=True,
        index=0,
        help="Light mode is best-effort — some Plotly charts stay dark.",
    )
    if theme == "Light":
        st.markdown(
            "<style>"
            ":root{--athanor-bg:#fafafa;--athanor-bg-2:#fff;"
            "--athanor-fg:#111;--athanor-muted:#666;--athanor-border:#e2e2e2;}"
            "html,body,[data-testid='stAppViewContainer']"
            "{background:#fafafa!important;color:#111!important;}"
            "</style>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(f"DB · `{ds.DB_PATH.name}`")
    st.caption(f"Runs · `{ds.RUNS_LOG_PATH.name}`")


# ─────────────────────────────────────────────────────────────────────────────
# Header strip (mode + last sync + kill-armed badge)
# ─────────────────────────────────────────────────────────────────────────────

_hdr = ds.get_header_info()
c.header_strip(mode=_hdr.mode, last_sync=_hdr.last_sync,
               kill_armed=_hdr.kill_armed)

if _hdr.kill_armed:
    st.error(
        "🛑  KILL SWITCH ARMED — `.athanor_kill` present. "
        "Live execution is blocked until the file is removed (Tab 6).",
        icon="⚠",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab router
# ─────────────────────────────────────────────────────────────────────────────

TABS: tuple[tuple[str, str], ...] = (
    ("Overview",     "overview"),
    ("Positions",    "positions"),
    ("Attribution",  "attribution"),
    ("Backtest",     "backtest"),
    ("Regime",       "regime"),
    ("Health",       "health"),
)

_tab_objects = st.tabs([label for label, _ in TABS])

for (label, module_name), tab_obj in zip(TABS, _tab_objects):
    with tab_obj:
        try:
            mod = importlib.import_module(f"dashboard.tabs.{module_name}")
        except ModuleNotFoundError:
            c.empty_placeholder(
                f"Tab '{label}' non ancora implementata — arriverà in un prossimo round.",
                icon="🚧",
            )
            continue
        except Exception as exc:
            log.exception("import dashboard.tabs.%s failed", module_name)
            st.error(f"Tab '{label}' import failed: {exc}")
            continue

        render = getattr(mod, "render", None)
        if render is None:
            st.error(f"dashboard.tabs.{module_name} has no render(ds, c)")
            continue

        try:
            render(ds, c)
        except Exception as exc:
            log.exception("render dashboard.tabs.%s failed", module_name)
            st.error(f"Tab '{label}' render failed: {exc}")
