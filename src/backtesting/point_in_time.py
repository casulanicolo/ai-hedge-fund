"""
src/backtesting/point_in_time.py
────────────────────────────────
Point-in-time data provider for forward-only backtesting.

Guarantees Ω(τ) = { x : pub_time(x) ≤ τ }: no future leakage.

Sources
-------
- OHLCV   : yfinance.history(end=as_of+1d), sliced to dates ≤ as_of
- Filings : SEC EDGAR submissions (filingDate ≤ as_of)
- XBRL    : SEC EDGAR companyfacts (entry.filed ≤ as_of)
- Macro   : yfinance ^VIX/^TNX/^IRX/^FVX history sliced ≤ as_of
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


CACHE_DIR = Path("cache/backtest_pit")
SEC_BASE = "https://data.sec.gov"
SEC_TICKER_CIK = "https://www.sec.gov/files/company_tickers.json"
SEC_HEADERS = {
    "User-Agent": "AthanorAlpha research@athanor-alpha.com",
    "Accept-Encoding": "gzip, deflate",
}
SEC_RATE_DELAY = 0.11  # ≤ 10 req/sec


def _to_date(d: Any) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    raise TypeError(f"Cannot coerce {type(d)} to date")


class PointInTimeDataProvider:
    """
    Stateless reader (with on-disk cache) returning only data
    that was *publicly available* at or before `as_of`.

    No execution, no signals — purely a data accessor.
    """

    def __init__(self, cache_dir: Path | str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cik_map: dict[str, str] = {}
        self._session = requests.Session()
        self._session.headers.update(SEC_HEADERS)
        self._last_sec_call: float = 0.0

    # ── OHLCV ─────────────────────────────────────────────────────────────
    def get_ohlcv(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        as_of: date | str,
    ) -> pd.DataFrame:
        """
        Return OHLCV bars for [start, end] but truncated to dates ≤ as_of.

        yfinance returns split/dividend-adjusted Close when auto_adjust=True.
        Adjustments are not point-in-time perfect (future splits restate
        history) — acceptable for academic backtest, not for tick-precise sim.
        """
        start_d = _to_date(start)
        end_d   = _to_date(end)
        as_of_d = _to_date(as_of)

        effective_end = min(end_d, as_of_d)
        if effective_end < start_d:
            return pd.DataFrame()

        try:
            df = yf.Ticker(ticker).history(
                start=start_d.isoformat(),
                end=(effective_end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                actions=False,
            )
        except Exception as exc:
            logger.warning("get_ohlcv(%s) failed: %s", ticker, exc)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # Defensive: drop tz, then truncate to ≤ as_of
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df.index.date <= as_of_d]
        return df

    # ── SEC filings (10-K, 10-Q, 8-K, Form 4) ─────────────────────────────
    def get_filings(self, ticker: str, as_of: date | str) -> dict:
        """
        Return SEC filings with filingDate ≤ as_of.
        Same shape as src/data/sec_edgar.py SECFetcher.get_filings.
        """
        as_of_d = _to_date(as_of)
        cik = self._get_cik(ticker)
        if cik is None:
            return {"ticker": ticker, "error": "CIK not found"}

        subs = self._fetch_submissions(cik)
        recent = subs.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        dates      = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        documents  = recent.get("primaryDocument", [])

        result: dict[str, Any] = {
            "ticker": ticker, "cik": cik, "as_of": as_of_d.isoformat(),
            "10-Q": [], "10-K": [], "form4": [], "8-K": [],
        }

        for i, form in enumerate(forms):
            filed_str = dates[i] if i < len(dates) else None
            if not filed_str:
                continue
            try:
                filed_d = datetime.strptime(filed_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if filed_d > as_of_d:
                continue

            entry = {
                "form": form,
                "date": filed_str,
                "accession": accessions[i] if i < len(accessions) else None,
                "document": documents[i] if i < len(documents) else None,
            }
            if form == "10-Q" and len(result["10-Q"]) < 4:
                result["10-Q"].append(entry)
            elif form == "10-K" and len(result["10-K"]) < 1:
                result["10-K"].append(entry)
            elif form == "4"   and len(result["form4"]) < 10:
                result["form4"].append(entry)
            elif form == "8-K" and len(result["8-K"]) < 5:
                result["8-K"].append(entry)
        return result

    # ── XBRL fundamentals (point-in-time via `filed`) ─────────────────────
    def get_fundamentals(self, ticker: str, as_of: date | str) -> dict:
        """
        Return latest XBRL metrics whose entry.filed ≤ as_of.

        Critical: SEC publishes 10-K reports months after period end. A
        FY2019 10-K filed 2020-10-30 must NOT appear when as_of=2020-03-01.
        """
        as_of_d = _to_date(as_of)
        cik = self._get_cik(ticker)
        if cik is None:
            return {"ticker": ticker, "error": "CIK not found"}

        facts = self._fetch_companyfacts(cik)
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        def latest_pit(concept: str) -> Optional[float]:
            data = us_gaap.get(concept, {})
            units = data.get("units", {})
            entries = units.get("USD", units.get("USD/shares", units.get("pure", []))) or []
            valid: list[dict] = []
            for e in entries:
                filed = e.get("filed")
                if not filed:
                    continue
                try:
                    if datetime.strptime(filed, "%Y-%m-%d").date() <= as_of_d:
                        valid.append(e)
                except ValueError:
                    continue
            if not valid:
                return None
            # Prefer 10-K > 10-Q; tie-break by `end` desc, then `filed` desc
            valid.sort(
                key=lambda x: (
                    0 if x.get("form") == "10-K" else 1,
                    x.get("end") or "",
                    x.get("filed") or "",
                ),
                reverse=True,
            )
            # After reverse: 10-K rank 0 becomes last → fix: explicit sort
            ten_k = [v for v in valid if v.get("form") == "10-K"]
            ten_q = [v for v in valid if v.get("form") == "10-Q"]
            pool = ten_k or ten_q or valid
            pool.sort(key=lambda x: (x.get("end") or "", x.get("filed") or ""), reverse=True)
            return pool[0].get("val")

        return {
            "ticker": ticker,
            "cik": cik,
            "as_of": as_of_d.isoformat(),
            "revenue":      latest_pit("Revenues"),
            "net_income":   latest_pit("NetIncomeLoss"),
            "eps_basic":    latest_pit("EarningsPerShareBasic"),
            "total_assets": latest_pit("Assets"),
            "total_debt":   latest_pit("LongTermDebt"),
            "operating_cf": latest_pit("NetCashProvidedByUsedInOperatingActivities"),
        }

    # ── Macro context as-of ───────────────────────────────────────────────
    def get_macro(self, as_of: date | str) -> dict:
        """
        Return VIX + yield curve as observed at as_of.
        """
        as_of_d = _to_date(as_of)

        def last_close(sym: str, lookback_days: int = 30) -> Optional[float]:
            try:
                df = yf.Ticker(sym).history(
                    start=(as_of_d - timedelta(days=lookback_days)).isoformat(),
                    end=(as_of_d + timedelta(days=1)).isoformat(),
                    auto_adjust=True, actions=False,
                )
                if df is None or df.empty:
                    return None
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[df.index.date <= as_of_d]
                if df.empty:
                    return None
                return round(float(df["Close"].iloc[-1]), 3)
            except Exception as exc:
                logger.warning("macro %s: %s", sym, exc)
                return None

        vix       = last_close("^VIX")
        yield_10y = last_close("^TNX")
        yield_3m  = last_close("^IRX")
        yield_5y  = last_close("^FVX")
        spread = (
            round(yield_10y - yield_3m, 3)
            if yield_10y is not None and yield_3m is not None else None
        )
        return {
            "as_of":              as_of_d.isoformat(),
            "vix":                vix,
            "yield_10y":          yield_10y,
            "yield_3m":           yield_3m,
            "yield_5y":           yield_5y,
            "yield_spread_10y3m": spread,
        }

    # ── SEC plumbing ──────────────────────────────────────────────────────
    def _sec_get(self, url: str) -> dict:
        elapsed = time.time() - self._last_sec_call
        if elapsed < SEC_RATE_DELAY:
            time.sleep(SEC_RATE_DELAY - elapsed)
        resp = self._session.get(url, timeout=20)
        self._last_sec_call = time.time()
        resp.raise_for_status()
        return resp.json()

    def _get_cik(self, ticker: str) -> Optional[str]:
        ticker = ticker.upper()
        if ticker.endswith("-USD"):
            return None  # crypto, no SEC registration
        if ticker in self._cik_map:
            return self._cik_map[ticker]

        cache_path = self.cache_dir / "cik_map.json"
        if cache_path.exists():
            try:
                self._cik_map = json.loads(cache_path.read_text(encoding="utf-8"))
                if ticker in self._cik_map:
                    return self._cik_map[ticker]
            except Exception:
                pass

        raw = self._sec_get(SEC_TICKER_CIK)
        self._cik_map = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in raw.values()
        }
        cache_path.write_text(json.dumps(self._cik_map), encoding="utf-8")
        return self._cik_map.get(ticker)

    def _fetch_submissions(self, cik: str) -> dict:
        path = self.cache_dir / f"submissions_{cik}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        data = self._sec_get(f"{SEC_BASE}/submissions/CIK{cik}.json")
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def _fetch_companyfacts(self, cik: str) -> dict:
        path = self.cache_dir / f"companyfacts_{cik}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        data = self._sec_get(f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
