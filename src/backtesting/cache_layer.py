"""
src/backtesting/cache_layer.py
──────────────────────────────
Two-tier cache for forward backtest:

1. State cache  : pickled AgentState (post-prefetch) per as_of date.
   Path: cache/backtest_states/YYYY-MM-DD.pkl
   Keyed by (as_of, sorted_tickers_hash).

2. Signal cache : per-agent JSON output per ticker per date.
   Path: cache/backtest_signals/{ticker}/{date}/{agent}.json

Rationale
---------
LLM calls dominate backtest cost. Caching analyst output by (date, ticker,
agent) lets re-runs (different portfolio sizing, different risk weights,
different IS/OOS splits) skip the LLM phase entirely.

State cache is coarser (full prefetched data) and used to skip
PointInTimeDataProvider re-fetches when iterating across daily bars.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


STATES_DIR  = Path("cache/backtest_states")
SIGNALS_DIR = Path("cache/backtest_signals")


# ── helpers ───────────────────────────────────────────────────────────────
def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _date_str(d: date | str) -> str:
    if isinstance(d, str):
        return d[:10]
    return d.isoformat()


def _ticker_hash(tickers: list[str]) -> str:
    joined = ",".join(sorted(t.upper() for t in tickers))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


# ── state cache ───────────────────────────────────────────────────────────
def state_cache_path(as_of: date | str, tickers: list[str]) -> Path:
    _ensure(STATES_DIR)
    return STATES_DIR / f"{_date_str(as_of)}_{_ticker_hash(tickers)}.pkl"


def load_state(as_of: date | str, tickers: list[str]) -> Optional[Any]:
    path = state_cache_path(as_of, tickers)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("state cache read failed %s: %s", path.name, exc)
        return None


def save_state(as_of: date | str, tickers: list[str], state: Any) -> None:
    path = state_cache_path(as_of, tickers)
    try:
        with path.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.warning("state cache write failed %s: %s", path.name, exc)


# ── signal cache ──────────────────────────────────────────────────────────
def signal_cache_path(ticker: str, as_of: date | str, agent: str) -> Path:
    d = _date_str(as_of)
    folder = _ensure(SIGNALS_DIR / ticker.upper() / d)
    return folder / f"{agent}.json"


def load_signal(ticker: str, as_of: date | str, agent: str) -> Optional[dict]:
    path = signal_cache_path(ticker, as_of, agent)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("signal cache read failed %s: %s", path, exc)
        return None


def save_signal(ticker: str, as_of: date | str, agent: str, signal: dict) -> None:
    path = signal_cache_path(ticker, as_of, agent)
    try:
        path.write_text(json.dumps(signal, default=str, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("signal cache write failed %s: %s", path, exc)


def load_all_signals(ticker: str, as_of: date | str) -> dict[str, dict]:
    """Return {agent_id: signal_dict} for every cached agent at (ticker, as_of)."""
    folder = SIGNALS_DIR / ticker.upper() / _date_str(as_of)
    if not folder.exists():
        return {}
    out: dict[str, dict] = {}
    for fp in folder.glob("*.json"):
        try:
            out[fp.stem] = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


def clear_cache(scope: str = "all") -> int:
    """Delete cached files. Returns count removed. scope: state | signals | all."""
    removed = 0
    if scope in ("state", "all") and STATES_DIR.exists():
        for fp in STATES_DIR.glob("*.pkl"):
            fp.unlink(); removed += 1
    if scope in ("signals", "all") and SIGNALS_DIR.exists():
        for fp in SIGNALS_DIR.rglob("*.json"):
            fp.unlink(); removed += 1
    return removed
