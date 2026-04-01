"""
src/data/state_reader.py
------------------------
Athanor Alpha - typed helper functions to read prefetched data from AgentState.

NO agent ever imports yfinance or calls SEC EDGAR directly.
Every agent reads data exclusively through these functions.

Available keys in prefetched_data[ticker]:
  ohlcv_daily     pd.DataFrame  OHLCV daily (1 year)
  ohlcv_5m        pd.DataFrame  OHLCV intraday 5m
  info            dict          market cap, P/E, sector, etc.
  income_stmt     pd.DataFrame  annual income statement
  income_stmt_q   pd.DataFrame  quarterly income statement
  balance_sheet   pd.DataFrame  annual balance sheet
  balance_sheet_q pd.DataFrame  quarterly balance sheet
  cash_flow       pd.DataFrame  annual cash flow
  cash_flow_q     pd.DataFrame  quarterly cash flow
  holders         pd.DataFrame  institutional holders
  sec_filings     dict          SEC EDGAR data (10-Q, 10-K, Form4, 8-K, xbrl_metrics)
  fetched_at      str           ISO-8601 UTC timestamp
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_ticker_payload(state: dict, ticker: str) -> Optional[dict]:
    """Return the full prefetched payload for a ticker, or None if missing."""
    prefetched: dict = state.get("data", {}).get("prefetched_data", {})
    payload = prefetched.get(ticker)
    if payload is None:
        logger.warning("state_reader: no prefetched data for ticker '%s'", ticker)
    return payload


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def get_ohlcv_daily(state: dict, ticker: str) -> Optional[pd.DataFrame]:
    """Daily OHLCV DataFrame (1 year). Index = Date."""
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return None
    df = payload.get("ohlcv_daily")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.warning("state_reader: ohlcv_daily empty for '%s'", ticker)
        return None
    return df


def get_ohlcv_5m(state: dict, ticker: str) -> Optional[pd.DataFrame]:
    """Intraday 5-minute OHLCV DataFrame (last 5 trading days)."""
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return None
    df = payload.get("ohlcv_5m")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


def get_ohlcv_weekly(state: dict, ticker: str) -> Optional[pd.DataFrame]:
    """Weekly OHLCV DataFrame (2 years). Index = Date."""
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return None
    df = payload.get("ohlcv_weekly")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


def get_ohlcv_4h(state: dict, ticker: str) -> Optional[pd.DataFrame]:
    """4-hour OHLCV DataFrame (60 days). Index = Datetime."""
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return None
    df = payload.get("ohlcv_4h")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


# ---------------------------------------------------------------------------
# Company info
# ---------------------------------------------------------------------------

def get_info(state: dict, ticker: str) -> dict:
    """
    Company info dict from yfinance (market cap, P/E, sector, etc.).
    Returns empty dict if unavailable.
    """
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return {}
    return payload.get("info") or {}


def get_market_cap(state: dict, ticker: str) -> Optional[float]:
    """Market cap in USD, or None."""
    info = get_info(state, ticker)
    return info.get("marketCap")


def get_pe_ratio(state: dict, ticker: str) -> Optional[float]:
    """Trailing P/E ratio, or None."""
    info = get_info(state, ticker)
    return info.get("trailingPE")


def get_sector(state: dict, ticker: str) -> Optional[str]:
    """Sector string (e.g. 'Technology'), or None."""
    info = get_info(state, ticker)
    return info.get("sector")


# ---------------------------------------------------------------------------
# Financial statements
# ---------------------------------------------------------------------------

def get_income_stmt(state: dict, ticker: str, quarterly: bool = False) -> Optional[pd.DataFrame]:
    """
    Income statement DataFrame.
    quarterly=False -> annual (default)
    quarterly=True  -> quarterly
    Columns = fiscal period dates, rows = line items.
    """
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return None
    key = "income_stmt_q" if quarterly else "income_stmt"
    df = payload.get(key)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


def get_balance_sheet(state: dict, ticker: str, quarterly: bool = False) -> Optional[pd.DataFrame]:
    """Balance sheet DataFrame. quarterly flag same as get_income_stmt."""
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return None
    key = "balance_sheet_q" if quarterly else "balance_sheet"
    df = payload.get(key)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


def get_cash_flow(state: dict, ticker: str, quarterly: bool = False) -> Optional[pd.DataFrame]:
    """Cash flow statement DataFrame. quarterly flag same as get_income_stmt."""
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return None
    key = "cash_flow_q" if quarterly else "cash_flow"
    df = payload.get(key)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


def get_holders(state: dict, ticker: str) -> Optional[pd.DataFrame]:
    """Institutional holders DataFrame."""
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return None
    df = payload.get("holders")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


# ---------------------------------------------------------------------------
# Convenience: extract a single scalar from a financial statement
# ---------------------------------------------------------------------------

def _extract_scalar(df: Optional[pd.DataFrame], row_name: str) -> Optional[float]:
    """
    Given a financial statement DataFrame (rows = line items, cols = periods),
    return the most recent value for `row_name`, or None.
    Handles partial string match (case-insensitive).
    """
    if df is None or df.empty:
        return None
    # Try exact match first
    if row_name in df.index:
        row = df.loc[row_name]
    else:
        # Partial case-insensitive match
        matches = [i for i in df.index if row_name.lower() in str(i).lower()]
        if not matches:
            return None
        row = df.loc[matches[0]]
    # Most recent period = first column
    try:
        val = row.iloc[0]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def get_revenue(state: dict, ticker: str, quarterly: bool = False) -> Optional[float]:
    """Most recent total revenue (annual or quarterly)."""
    df = get_income_stmt(state, ticker, quarterly=quarterly)
    return _extract_scalar(df, "Total Revenue")


def get_net_income(state: dict, ticker: str, quarterly: bool = False) -> Optional[float]:
    """Most recent net income."""
    df = get_income_stmt(state, ticker, quarterly=quarterly)
    return _extract_scalar(df, "Net Income")


def get_operating_income(state: dict, ticker: str, quarterly: bool = False) -> Optional[float]:
    """Most recent operating income."""
    df = get_income_stmt(state, ticker, quarterly=quarterly)
    return _extract_scalar(df, "Operating Income")


def get_total_assets(state: dict, ticker: str) -> Optional[float]:
    """Most recent total assets (annual)."""
    df = get_balance_sheet(state, ticker)
    return _extract_scalar(df, "Total Assets")


def get_total_debt(state: dict, ticker: str) -> Optional[float]:
    """Most recent total debt (annual)."""
    df = get_balance_sheet(state, ticker)
    val = _extract_scalar(df, "Total Debt")
    if val is None:
        val = _extract_scalar(df, "Long Term Debt")
    return val


def get_stockholders_equity(state: dict, ticker: str) -> Optional[float]:
    """Most recent stockholders equity (annual)."""
    df = get_balance_sheet(state, ticker)
    return _extract_scalar(df, "Stockholders Equity")


def get_free_cash_flow(state: dict, ticker: str, quarterly: bool = False) -> Optional[float]:
    """Most recent free cash flow."""
    df = get_cash_flow(state, ticker, quarterly=quarterly)
    return _extract_scalar(df, "Free Cash Flow")


def get_operating_cash_flow(state: dict, ticker: str, quarterly: bool = False) -> Optional[float]:
    """Most recent operating cash flow."""
    df = get_cash_flow(state, ticker, quarterly=quarterly)
    return _extract_scalar(df, "Operating Cash Flow")


# ---------------------------------------------------------------------------
# Derived ratios (computed locally, no LLM)
# ---------------------------------------------------------------------------

def get_debt_to_equity(state: dict, ticker: str) -> Optional[float]:
    """Total debt / stockholders equity. None if either is unavailable."""
    debt = get_total_debt(state, ticker)
    equity = get_stockholders_equity(state, ticker)
    if debt is None or equity is None or equity == 0:
        return None
    return debt / equity


def get_return_on_equity(state: dict, ticker: str) -> Optional[float]:
    """Net income / stockholders equity (annual)."""
    net_income = get_net_income(state, ticker)
    equity = get_stockholders_equity(state, ticker)
    if net_income is None or equity is None or equity == 0:
        return None
    return net_income / equity


def get_net_margin(state: dict, ticker: str) -> Optional[float]:
    """Net income / revenue (annual)."""
    net_income = get_net_income(state, ticker)
    revenue = get_revenue(state, ticker)
    if net_income is None or revenue is None or revenue == 0:
        return None
    return net_income / revenue


def get_operating_margin(state: dict, ticker: str) -> Optional[float]:
    """Operating income / revenue (annual)."""
    op_income = get_operating_income(state, ticker)
    revenue = get_revenue(state, ticker)
    if op_income is None or revenue is None or revenue == 0:
        return None
    return op_income / revenue


# ---------------------------------------------------------------------------
# SEC filings
# ---------------------------------------------------------------------------

def get_sec_filings(state: dict, ticker: str) -> dict:
    """
    Full SEC EDGAR payload for ticker.
    Keys: filings_10q, filings_10k, filings_form4, filings_8k, xbrl_metrics, fetched_at
    Returns empty dict if unavailable.
    """
    payload = _get_ticker_payload(state, ticker)
    if payload is None:
        return {}
    return payload.get("sec_filings") or {}


def get_xbrl_metrics(state: dict, ticker: str) -> dict:
    """
    Parsed XBRL financial metrics from SEC EDGAR.
    Keys: revenue, net_income, eps, total_assets, total_debt, operating_cf
    Each value is a list of {"val": ..., "end": "YYYY-MM-DD", ...} dicts.
    Returns empty dict if unavailable.
    """
    sec = get_sec_filings(state, ticker)
    return sec.get("xbrl_metrics") or {}


def get_insider_transactions(state: dict, ticker: str) -> list:
    """
    List of recent Form 4 insider transactions.
    Each item is a dict with SEC filing metadata.
    Returns empty list if unavailable.
    """
    sec = get_sec_filings(state, ticker)
    return sec.get("filings_form4") or []


def get_recent_8k(state: dict, ticker: str) -> list:
    """
    List of recent 8-K material event filings.
    Returns empty list if unavailable.
    """
    sec = get_sec_filings(state, ticker)
    return sec.get("filings_8k") or []


# ---------------------------------------------------------------------------
# Financials summary dict (for agents that want everything in one call)
# ---------------------------------------------------------------------------

def get_financials_summary(state: dict, ticker: str) -> dict:
    """
    Returns a flat dict of the most commonly used financial metrics,
    computed from prefetched data. Useful for agents that need to pass
    a summary to an LLM prompt.

    Keys:
      market_cap, pe_ratio, sector,
      revenue, net_income, operating_income,
      net_margin, operating_margin, return_on_equity,
      total_assets, total_debt, stockholders_equity, debt_to_equity,
      free_cash_flow, operating_cash_flow
    """
    return {
        "market_cap": get_market_cap(state, ticker),
        "pe_ratio": get_pe_ratio(state, ticker),
        "sector": get_sector(state, ticker),
        "revenue": get_revenue(state, ticker),
        "net_income": get_net_income(state, ticker),
        "operating_income": get_operating_income(state, ticker),
        "net_margin": get_net_margin(state, ticker),
        "operating_margin": get_operating_margin(state, ticker),
        "return_on_equity": get_return_on_equity(state, ticker),
        "total_assets": get_total_assets(state, ticker),
        "total_debt": get_total_debt(state, ticker),
        "stockholders_equity": get_stockholders_equity(state, ticker),
        "debt_to_equity": get_debt_to_equity(state, ticker),
        "free_cash_flow": get_free_cash_flow(state, ticker),
        "operating_cash_flow": get_operating_cash_flow(state, ticker),
    }
