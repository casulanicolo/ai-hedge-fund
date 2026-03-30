"""
src/tools/api_shim.py
---------------------
Athanor Alpha - drop-in replacements for src/tools/api functions.

These functions have IDENTICAL signatures to the originals but read
from state["data"]["prefetched_data"] instead of calling external APIs.

USAGE IN AGENTS:
  1. Change import line:
       from src.tools.api_shim import get_financial_metrics, get_market_cap, search_line_items
  2. Register state at top of agent function:
       from src.tools.api_shim import register_state
       register_state(state)
  3. All subsequent calls to get_financial_metrics() etc. work automatically.
"""
from __future__ import annotations

import logging
import threading
import pandas as pd
from typing import Any, Optional

from src.data.models import FinancialMetrics, LineItem
from src.data.state_reader import (
    get_info,
    get_market_cap as _get_market_cap,
    get_ohlcv_daily,
    get_income_stmt,
    get_cash_flow,
    get_balance_sheet,
    _extract_scalar,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local state registry
# Each agent thread registers its state once; all shim functions read from it.
# ---------------------------------------------------------------------------
_local = threading.local()

def register_state(state: dict) -> None:
    """Call this once at the top of each agent function before any shim calls."""
    _local.state = state

def _get_state() -> Optional[dict]:
    return getattr(_local, "state", None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None and pd.notna(val) else None
    except Exception:
        return None

def _growth(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)

def _col0(df, row: str) -> Optional[float]:
    return _extract_scalar(df, row)

def _col1(df, row: str) -> Optional[float]:
    """Second most recent period."""
    if df is None or df.empty or df.shape[1] < 2:
        return None
    matches = [i for i in df.index if row.lower() in str(i).lower()]
    if not matches:
        return None
    try:
        return _safe(df.loc[matches[0]].iloc[1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FinancialMetrics builder
# ---------------------------------------------------------------------------

def _build_financial_metrics(state: dict, ticker: str) -> FinancialMetrics:
    info  = get_info(state, ticker)
    inc_a = get_income_stmt(state, ticker, quarterly=False)
    cf_a  = get_cash_flow(state, ticker, quarterly=False)
    bs_a  = get_balance_sheet(state, ticker, quarterly=False)

    market_cap = _safe(info.get("marketCap"))
    shares     = _safe(info.get("sharesOutstanding"))

    rev0  = _col0(inc_a, "Total Revenue")
    rev1  = _col1(inc_a, "Total Revenue")
    ni0   = _col0(inc_a, "Net Income")
    ni1   = _col1(inc_a, "Net Income")
    fcf0  = _col0(cf_a,  "Free Cash Flow")
    fcf1  = _col1(cf_a,  "Free Cash Flow")
    oi0   = _col0(inc_a, "Operating Income")
    oi1   = _col1(inc_a, "Operating Income")
    bv0   = _col0(bs_a,  "Stockholders Equity")
    bv1   = _col1(bs_a,  "Stockholders Equity")
    total_assets = _col0(bs_a, "Total Assets")
    total_debt   = _col0(bs_a, "Total Debt") or _col0(bs_a, "Long Term Debt")

    debt_to_eq = _safe(info.get("debtToEquity"))
    if debt_to_eq is not None:
        debt_to_eq = debt_to_eq / 100  # yfinance gives as percentage

    return FinancialMetrics(
        ticker=ticker,
        report_period="latest",
        period="ttm",
        currency="USD",
        market_cap=market_cap,
        enterprise_value=_safe(info.get("enterpriseValue")),
        price_to_earnings_ratio=_safe(info.get("trailingPE")),
        price_to_book_ratio=_safe(info.get("priceToBook")),
        price_to_sales_ratio=_safe(info.get("priceToSalesTrailing12Months")),
        enterprise_value_to_ebitda_ratio=_safe(info.get("enterpriseToEbitda")),
        enterprise_value_to_revenue_ratio=_safe(info.get("enterpriseToRevenue")),
        free_cash_flow_yield=(fcf0 / market_cap) if (fcf0 and market_cap and market_cap != 0) else None,
        peg_ratio=_safe(info.get("pegRatio")),
        gross_margin=_safe(info.get("grossMargins")),
        operating_margin=_safe(info.get("operatingMargins")),
        net_margin=_safe(info.get("profitMargins")),
        return_on_equity=_safe(info.get("returnOnEquity")),
        return_on_assets=_safe(info.get("returnOnAssets")),
        return_on_invested_capital=None,
        asset_turnover=None,
        inventory_turnover=None,
        receivables_turnover=None,
        days_sales_outstanding=None,
        operating_cycle=None,
        working_capital_turnover=None,
        current_ratio=_safe(info.get("currentRatio")),
        quick_ratio=_safe(info.get("quickRatio")),
        cash_ratio=None,
        operating_cash_flow_ratio=None,
        debt_to_equity=debt_to_eq,
        debt_to_assets=(total_debt / total_assets) if (total_debt and total_assets and total_assets != 0) else None,
        interest_coverage=None,
        revenue_growth=_growth(rev0, rev1),
        earnings_growth=_growth(ni0, ni1),
        book_value_growth=_growth(bv0, bv1),
        earnings_per_share_growth=None,
        free_cash_flow_growth=_growth(fcf0, fcf1),
        operating_income_growth=_growth(oi0, oi1),
        ebitda_growth=None,
        payout_ratio=_safe(info.get("payoutRatio")),
        earnings_per_share=_safe(info.get("trailingEps")),
        book_value_per_share=_safe(info.get("bookValue")),
        free_cash_flow_per_share=(fcf0 / shares) if (fcf0 and shares and shares != 0) else None,
    )


# ---------------------------------------------------------------------------
# LineItem builder
# ---------------------------------------------------------------------------

def _build_line_item(state: dict, ticker: str, field_names: list[str]) -> LineItem:
    info  = get_info(state, ticker)
    inc_a = get_income_stmt(state, ticker, quarterly=False)
    cf_a  = get_cash_flow(state, ticker, quarterly=False)
    bs_a  = get_balance_sheet(state, ticker, quarterly=False)

    shares = _safe(info.get("sharesOutstanding"))

    mapping = {
        "revenue":                               _col0(inc_a, "Total Revenue"),
        "net_income":                            _col0(inc_a, "Net Income"),
        "operating_income":                      _col0(inc_a, "Operating Income"),
        "gross_profit":                          _col0(inc_a, "Gross Profit"),
        "capital_expenditure":                   _col0(cf_a,  "Capital Expenditure"),
        "depreciation_and_amortization":         _col0(cf_a,  "Depreciation"),
        "free_cash_flow":                        _col0(cf_a,  "Free Cash Flow"),
        "operating_cash_flow":                   _col0(cf_a,  "Operating Cash Flow"),
        "total_assets":                          _col0(bs_a,  "Total Assets"),
        "total_liabilities":                     _col0(bs_a,  "Total Liabilities Net Minority Interest"),
        "shareholders_equity":                   _col0(bs_a,  "Stockholders Equity"),
        "total_debt":                            _col0(bs_a,  "Total Debt"),
        "cash_and_equivalents":                  _col0(bs_a,  "Cash And Cash Equivalents"),
        "outstanding_shares":                    shares,
        "issuance_or_purchase_of_equity_shares": _col0(cf_a,  "Repurchase Of Capital Stock"),
        "dividends_and_other_cash_distributions":_col0(cf_a,  "Cash Dividends Paid"),
        "earnings_per_share":                    _safe(info.get("trailingEps")),
        "book_value_per_share":                  _safe(info.get("bookValue")),
        "price_to_earnings_ratio":               _safe(info.get("trailingPE")),
        "price_to_book_ratio":                   _safe(info.get("priceToBook")),
        "return_on_equity":                      _safe(info.get("returnOnEquity")),
        "return_on_assets":                      _safe(info.get("returnOnAssets")),
        "debt_to_equity":                        _safe(info.get("debtToEquity")),
        "current_ratio":                         _safe(info.get("currentRatio")),
        "market_cap":                            _safe(info.get("marketCap")),
    }

    data = {"ticker": ticker, "report_period": "latest", "period": "ttm", "currency": "USD"}
    for field in field_names:
        data[field] = mapping.get(field)
    return LineItem(**data)


# ---------------------------------------------------------------------------
# Public shim functions — identical signatures to src/tools/api
# ---------------------------------------------------------------------------

def get_financial_metrics(
    ticker: str,
    end_date: str = None,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    state = _get_state()
    if state is None:
        logger.warning("api_shim: state not registered for ticker '%s'", ticker)
        return []
    return [_build_financial_metrics(state, ticker)]


def get_market_cap(
    ticker: str,
    end_date: str = None,
    api_key: str = None,
) -> Optional[float]:
    state = _get_state()
    if state is None:
        return None
    return _get_market_cap(state, ticker)


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str = None,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    state = _get_state()
    if state is None:
        logger.warning("api_shim: state not registered for ticker '%s'", ticker)
        return []
    return [_build_line_item(state, ticker, line_items)]


def get_prices(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    api_key: str = None,
) -> Optional[pd.DataFrame]:
    state = _get_state()
    if state is None:
        return None
    return get_ohlcv_daily(state, ticker)


def prices_to_df(prices) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        return prices
    return pd.DataFrame()


def get_insider_trades(ticker: str, end_date: str = None, start_date: str = None,
                       limit: int = 1000, api_key: str = None) -> list:
    return []


def get_company_news(ticker: str, end_date: str = None, start_date: str = None,
                     limit: int = 100, api_key: str = None) -> list:
    return []
