"""
src/data/ttl_cache.py
──────────────────────
Simple in-memory TTL cache for Athanor Alpha's DataPrefetcher.

TTL defaults (overridable per-key):
  - prices / intraday : 60 seconds   (refreshed every monitor cycle)
  - financials        : 86400 seconds (24h — quarterly data)
  - sec_filings       : 604800 seconds (7 days — filings don't change often)
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Optional


# Default TTLs in seconds
TTL_PRICES     = 60
TTL_FINANCIALS = 86_400
TTL_SEC        = 604_800


class TTLCache:
    """
    Thread-safe in-memory key-value cache with per-entry TTL.

    Usage
    -----
    cache = TTLCache()
    cache.set("AAPL:ohlcv", data, ttl=TTL_FINANCIALS)
    value = cache.get("AAPL:ohlcv")   # None if expired or missing
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value, or None if missing / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: float = TTL_PRICES) -> None:
        """Store *value* under *key* for *ttl* seconds."""
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton — shared across all DataPrefetcher calls
_cache = TTLCache()


def get_ttl_cache() -> TTLCache:
    """Return the global TTL cache instance."""
    return _cache
