"""
SEC EDGAR REST API fetcher for Athanor Alpha.
Public API — no API key required.
Rate limit: 10 req/sec per SEC policy.
Disk cache with 7-day TTL.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL          = "https://data.sec.gov"
SUBMISSIONS_URL   = "https://data.sec.gov/submissions"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts"
TICKER_CIK_URL    = "https://www.sec.gov/files/company_tickers.json"

RATE_LIMIT_DELAY = 0.11          # seconds between requests (≤ 10 req/sec)
CACHE_TTL_DAYS   = 7
CACHE_DIR        = Path("cache/sec_edgar")

# SEC requires a descriptive User-Agent to avoid being blocked
HEADERS = {
    "User-Agent": "AthanorAlpha research@athanor-alpha.com",
    "Accept-Encoding": "gzip, deflate"
}


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_path(key: str) -> Path:
    """Return the file path for a given cache key."""
    safe_key = key.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe_key}.json"


def _cache_read(key: str) -> Optional[dict]:
    """Return cached data if it exists and is within TTL, else None."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if datetime.utcnow() - cached_at > timedelta(days=CACHE_TTL_DAYS):
            logger.debug(f"Cache expired for {key}")
            return None
        return entry["data"]
    except Exception as e:
        logger.warning(f"Cache read error for {key}: {e}")
        return None


def _cache_write(key: str, data: dict) -> None:
    """Write data to disk cache with current timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    entry = {
        "cached_at": datetime.utcnow().isoformat(),
        "data": data,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
    except Exception as e:
        logger.warning(f"Cache write error for {key}: {e}")


# ── Main class ────────────────────────────────────────────────────────────────
class SECFetcher:
    """
    Fetches SEC EDGAR filings for a list of tickers.
    All results are cached to disk for 7 days.
    """

    def __init__(self):
        self.session = requests.Session()
        # No persistent Host header — let requests set it per-request
        self.session.headers.update({
            "User-Agent": "AthanorAlpha research@athanor-alpha.com",
            "Accept-Encoding": "gzip, deflate",
        })
        self._cik_map: dict[str, str] = {}
        self._last_request_time: float = 0.0

    # ── Rate limiter ──────────────────────────────────────────────────────────
    def _get(self, url: str, params: dict = None) -> dict:
        """HTTP GET with rate limiting. Raises on HTTP error."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)

        logger.debug(f"GET {url}")
        response = self.session.get(url, params=params, timeout=15)
        self._last_request_time = time.time()
        response.raise_for_status()
        return response.json()

    # ── Placeholder methods (implemented in next steps) ───────────────────────
    def _get_cik(self, ticker: str) -> Optional[str]:
        """Resolve ticker → zero-padded 10-digit CIK."""
        ticker = ticker.upper()

        # Return from in-memory map if already resolved
        if ticker in self._cik_map:
            return self._cik_map[ticker]

        # Try disk cache first
        cache_key = f"cik_map"
        cik_map = _cache_read(cache_key)

        if cik_map is None:
            # Download the full ticker→CIK map from SEC (one-time fetch)
            logger.info("Downloading CIK map from SEC EDGAR...")
            raw = self._get(TICKER_CIK_URL)
            # SEC returns {0: {cik_str, ticker, title}, 1: {...}, ...}
            cik_map = {
                v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                for v in raw.values()
            }
            _cache_write(cache_key, cik_map)

        self._cik_map = cik_map
        result = cik_map.get(ticker)
        if result is None:
            logger.warning(f"CIK not found for ticker: {ticker}")
        return result

    def _get_submissions(self, cik: str) -> dict:
        """Download submission history for a CIK (cached)."""
        cache_key = f"submissions_{cik}"
        cached = _cache_read(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: submissions for CIK {cik}")
            return cached

        url = f"{SUBMISSIONS_URL}/CIK{cik}.json"
        # Host header must change for this endpoint
        self.session.headers.update({
            "User-Agent": "AthanorAlpha research@athanor-alpha.com",
            "Accept-Encoding": "gzip, deflate",
        })
        data = self._get(url)
        _cache_write(cache_key, data)
        return data

    def get_filings(self, ticker: str) -> dict:
        """
        Return SEC filings for a ticker:
        - Last 4 10-Q
        - Last 10-K
        - Last 10 Form 4 (insider transactions)
        - Last 5 8-K (material events)
        Also fetches key XBRL metrics from companyfacts.
        """
        cik = self._get_cik(ticker)
        if cik is None:
            return {"error": f"CIK not found for {ticker}"}

        cache_key = f"filings_{ticker}"
        cached = _cache_read(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: filings for {ticker}")
            return cached

        result = {
            "ticker": ticker,
            "cik": cik,
            "10-Q": [],
            "10-K": [],
            "form4": [],
            "8-K": [],
            "xbrl_metrics": {},
        }

        # ── Submissions (filing history) ──────────────────────────────────
        try:
            subs = self._get_submissions(cik)
            filings = subs.get("filings", {}).get("recent", {})

            forms      = filings.get("form", [])
            dates      = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            documents  = filings.get("primaryDocument", [])

            for i, form in enumerate(forms):
                entry = {
                    "form": form,
                    "date": dates[i] if i < len(dates) else None,
                    "accession": accessions[i] if i < len(accessions) else None,
                    "document": documents[i] if i < len(documents) else None,
                }
                if form == "10-Q" and len(result["10-Q"]) < 4:
                    result["10-Q"].append(entry)
                elif form == "10-K" and len(result["10-K"]) < 1:
                    result["10-K"].append(entry)
                elif form == "4" and len(result["form4"]) < 10:
                    result["form4"].append(entry)
                elif form == "8-K" and len(result["8-K"]) < 5:
                    result["8-K"].append(entry)
        except Exception as e:
            logger.warning(f"Submissions fetch failed for {ticker}: {e}")

        # ── XBRL company facts (key metrics) ─────────────────────────────
        try:
            facts_url = f"{COMPANY_FACTS_URL}/CIK{cik}.json"
            facts = self._get(facts_url)
            us_gaap = facts.get("facts", {}).get("us-gaap", {})

            def latest_value(concept: str) -> float | None:
                """Extract the most recent annual value for a GAAP concept."""
                data = us_gaap.get(concept, {})
                units = data.get("units", {})
                # Try USD first, then pure numbers
                entries = units.get("USD", units.get("pure", []))
                # Keep only 10-K annual filings
                annual = [e for e in entries if e.get("form") == "10-K"]
                if not annual:
                    return None
                annual.sort(key=lambda x: x.get("end", ""), reverse=True)
                return annual[0].get("val")

            result["xbrl_metrics"] = {
                "revenue":         latest_value("Revenues"),
                "net_income":      latest_value("NetIncomeLoss"),
                "eps_basic":       latest_value("EarningsPerShareBasic"),
                "total_assets":    latest_value("Assets"),
                "total_debt":      latest_value("LongTermDebt"),
                "operating_cf":    latest_value("NetCashProvidedByUsedInOperatingActivities"),
            }
        except Exception as e:
            logger.warning(f"XBRL fetch failed for {ticker}: {e}")

        _cache_write(cache_key, result)
        return result

    def fetch_all(self, tickers: list[str]) -> dict[str, dict]:
        """
        Fetch SEC filings for all tickers.
        Returns dict: ticker → filings dict.
        """
        results = {}
        for ticker in tickers:
            logger.info(f"SECFetcher: fetching {ticker}...")
            try:
                results[ticker] = self.get_filings(ticker)
            except Exception as e:
                logger.error(f"SECFetcher: failed for {ticker}: {e}")
                results[ticker] = {"error": str(e)}
        return results