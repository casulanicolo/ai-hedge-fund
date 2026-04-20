"""
tests/backtesting/test_point_in_time.py
───────────────────────────────────────
Acceptance: PointInTimeDataProvider must NOT leak future data.

Critical asymmetry exercised here: a 10-K with period end 2019-12-31
filed on 2020-10-30 must be invisible at as_of=2020-03-01.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.backtesting.point_in_time import PointInTimeDataProvider, _to_date


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def provider(tmp_path: Path) -> PointInTimeDataProvider:
    return PointInTimeDataProvider(cache_dir=tmp_path)


@pytest.fixture
def fake_companyfacts() -> dict:
    """
    Synthetic SEC companyfacts for AAPL with two FY 10-K reports:
      - FY2018 10-K filed 2018-11-05  (val=10)
      - FY2019 10-K filed 2020-10-30  (val=20)  ← future-leak candidate
    Plus one 10-Q filed 2019-08-01 (val=15).
    """
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"val": 10, "form": "10-K", "end": "2018-09-30",
                             "filed": "2018-11-05", "fy": 2018, "fp": "FY"},
                            {"val": 15, "form": "10-Q", "end": "2019-06-30",
                             "filed": "2019-08-01", "fy": 2019, "fp": "Q3"},
                            {"val": 20, "form": "10-K", "end": "2019-09-30",
                             "filed": "2020-10-30", "fy": 2019, "fp": "FY"},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"val": 5, "form": "10-K", "end": "2018-09-30",
                             "filed": "2018-11-05", "fy": 2018, "fp": "FY"},
                        ]
                    }
                },
            }
        }
    }


@pytest.fixture
def fake_submissions() -> dict:
    return {
        "filings": {
            "recent": {
                "form":            ["10-K",       "10-Q",       "10-K",       "8-K"],
                "filingDate":      ["2018-11-05", "2019-08-01", "2020-10-30", "2020-04-15"],
                "accessionNumber": ["a1",         "a2",         "a3",         "a4"],
                "primaryDocument": ["d1.htm",     "d2.htm",     "d3.htm",     "d4.htm"],
            }
        }
    }


# ── Tests: helpers ────────────────────────────────────────────────────────
def test_to_date_accepts_iso_string():
    assert _to_date("2024-01-15") == date(2024, 1, 15)


def test_to_date_accepts_date_object():
    d = date(2024, 1, 15)
    assert _to_date(d) == d


# ── Tests: get_fundamentals (THE critical leakage check) ──────────────────
def test_get_fundamentals_excludes_future_filed_10k(provider, fake_companyfacts):
    """
    AAPL FY2019 10-K was filed 2020-10-30. At as_of=2020-03-01, only the
    FY2018 10-K (filed 2018-11-05, val=10) must be returned for Revenues.
    """
    with patch.object(provider, "_get_cik", return_value="0000320193"), \
         patch.object(provider, "_fetch_companyfacts", return_value=fake_companyfacts):
        out = provider.get_fundamentals("AAPL", as_of="2020-03-01")
    assert out["revenue"] == 10, "FY2019 10-K (filed 2020-10-30) must be invisible at 2020-03-01"


def test_get_fundamentals_includes_10k_after_filing(provider, fake_companyfacts):
    with patch.object(provider, "_get_cik", return_value="0000320193"), \
         patch.object(provider, "_fetch_companyfacts", return_value=fake_companyfacts):
        out = provider.get_fundamentals("AAPL", as_of="2020-12-01")
    assert out["revenue"] == 20, "After 2020-10-30 the FY2019 10-K becomes visible"


def test_get_fundamentals_returns_none_when_no_filing_yet(provider, fake_companyfacts):
    with patch.object(provider, "_get_cik", return_value="0000320193"), \
         patch.object(provider, "_fetch_companyfacts", return_value=fake_companyfacts):
        out = provider.get_fundamentals("AAPL", as_of="2018-01-01")
    assert out["revenue"] is None
    assert out["net_income"] is None


def test_get_fundamentals_unknown_cik_returns_error(provider):
    with patch.object(provider, "_get_cik", return_value=None):
        out = provider.get_fundamentals("ZZZZ", as_of="2024-01-01")
    assert out.get("error") == "CIK not found"


# ── Tests: get_filings ────────────────────────────────────────────────────
def test_get_filings_excludes_future_filings(provider, fake_submissions):
    with patch.object(provider, "_get_cik", return_value="0000320193"), \
         patch.object(provider, "_fetch_submissions", return_value=fake_submissions):
        out = provider.get_filings("AAPL", as_of="2020-03-01")
    # 10-K filed 2018-11-05 OK; 10-Q filed 2019-08-01 OK; 10-K 2020-10-30 NOT OK; 8-K 2020-04-15 NOT OK
    assert len(out["10-K"]) == 1 and out["10-K"][0]["date"] == "2018-11-05"
    assert len(out["10-Q"]) == 1 and out["10-Q"][0]["date"] == "2019-08-01"
    assert out["8-K"] == []


def test_get_filings_includes_after_filing_date(provider, fake_submissions):
    with patch.object(provider, "_get_cik", return_value="0000320193"), \
         patch.object(provider, "_fetch_submissions", return_value=fake_submissions):
        out = provider.get_filings("AAPL", as_of="2020-12-31")
    assert len(out["10-K"]) == 1  # only one 10-K slot, but the latest 2020-10-30 wins iteration order
    assert any(e["date"] == "2020-04-15" for e in out["8-K"])


# ── Tests: get_ohlcv truncation ───────────────────────────────────────────
def test_get_ohlcv_truncates_to_as_of(provider):
    """
    Mock yfinance to return 2 weeks of bars; verify rows after as_of dropped.
    """
    idx = pd.date_range("2024-01-02", "2024-01-15", freq="B")
    fake_df = pd.DataFrame({
        "Open":   range(len(idx)), "High": range(len(idx)),
        "Low":    range(len(idx)), "Close": range(len(idx)),
        "Volume": [100] * len(idx),
    }, index=idx)

    with patch("src.backtesting.point_in_time.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = fake_df
        out = provider.get_ohlcv("AAPL",
                                 start="2024-01-02",
                                 end="2024-01-15",
                                 as_of="2024-01-08")

    assert not out.empty
    assert out.index.max().date() <= date(2024, 1, 8)


def test_get_ohlcv_returns_empty_when_as_of_before_start(provider):
    out = provider.get_ohlcv("AAPL",
                             start="2024-06-01",
                             end="2024-06-30",
                             as_of="2024-01-01")
    assert out.empty


def test_get_cik_skips_crypto(provider):
    assert provider._get_cik("BTC-USD") is None
    assert provider._get_cik("ETH-USD") is None
