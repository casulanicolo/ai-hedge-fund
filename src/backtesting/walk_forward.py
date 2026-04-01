"""Walk-forward in-sample / out-of-sample backtesting analysis.

Usage:
    analyzer = WalkForwardAnalyzer(
        agent=run_hedge_fund,
        tickers=["AAPL", "MSFT"],
        is_start="2023-01-01", is_end="2023-12-31",
        oos_start="2024-01-01", oos_end="2024-12-31",
        initial_capital=100_000,
        model_name="gpt-4o",
        model_provider="OpenAI",
        selected_analysts=None,
    )
    result = analyzer.run()
    analyzer.print_report(result)
    analyzer.save_results(result)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from .engine import BacktestEngine
from .benchmarks import BenchmarkCalculator
from .types import PerformanceMetrics, PortfolioValuePoint
from src.tools.api import get_price_data


ANNUAL_RF_RATE = 0.0434
ANNUAL_TRADING_DAYS = 252


@dataclass
class PeriodResult:
    """Performance metrics for a single backtest window (IS or OOS)."""

    label: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    volatility_ann_pct: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown_pct: float | None
    max_drawdown_date: str | None
    calmar_ratio: float | None
    positive_days_pct: float
    benchmark_return_pct: float | None
    benchmark_ann_return_pct: float | None
    alpha_ann_pct: float | None
    beta: float | None
    n_trading_days: int


@dataclass
class WalkForwardResult:
    """Combined IS and OOS results with overfitting diagnostics."""

    is_result: PeriodResult
    oos_result: PeriodResult
    # IS Sharpe - OOS Sharpe: ideal < 0.3, concerning > 1.0
    sharpe_decay: float | None
    # OOS / IS total return ratio: ideal > 0.5
    return_retention: float | None


class WalkForwardAnalyzer:
    """Runs two independent BacktestEngine runs (IS + OOS) and compares them.

    The two periods are intentionally isolated: the OOS period starts with a
    fresh portfolio at `initial_capital`, giving an unbiased performance read.
    """

    def __init__(
        self,
        *,
        agent,
        tickers: list[str],
        is_start: str,
        is_end: str,
        oos_start: str,
        oos_end: str,
        initial_capital: float = 100_000.0,
        model_name: str,
        model_provider: str,
        selected_analysts: list[str] | None,
        margin_requirement: float = 0.0,
    ) -> None:
        self._agent = agent
        self._tickers = tickers
        self._is_start = is_start
        self._is_end = is_end
        self._oos_start = oos_start
        self._oos_end = oos_end
        self._initial_capital = initial_capital
        self._model_name = model_name
        self._model_provider = model_provider
        self._selected_analysts = selected_analysts
        self._margin_requirement = margin_requirement
        self._benchmark = BenchmarkCalculator()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_engine(self, start: str, end: str) -> BacktestEngine:
        return BacktestEngine(
            agent=self._agent,
            tickers=self._tickers,
            start_date=start,
            end_date=end,
            initial_capital=self._initial_capital,
            model_name=self._model_name,
            model_provider=self._model_provider,
            selected_analysts=self._selected_analysts,
            initial_margin_requirement=self._margin_requirement,
        )

    def _run_period(
        self, start: str, end: str, label: str
    ) -> tuple[PerformanceMetrics, list[PortfolioValuePoint]]:
        print(f"\n{'=' * 60}")
        print(f"  {label}: {start}  ->  {end}")
        print(f"  Tickers : {', '.join(self._tickers)}")
        print(f"  Model   : {self._model_provider}/{self._model_name}")
        print(f"{'=' * 60}\n")

        engine = self._build_engine(start, end)
        metrics = engine.run_backtest()
        portfolio_values = list(engine.get_portfolio_values())
        return metrics, portfolio_values

    def _annualize_return(self, total_return_pct: float, n_days: int) -> float:
        if n_days <= 0:
            return 0.0
        return ((1 + total_return_pct / 100) ** (ANNUAL_TRADING_DAYS / n_days) - 1) * 100

    def _compute_alpha_beta(
        self, strategy_returns: pd.Series, start: str, end: str
    ) -> tuple[float | None, float | None]:
        """Jensen's alpha (annualized %) and market beta vs SPY."""
        try:
            spy_df = get_price_data("SPY", start, end)
            if spy_df.empty:
                return None, None
            spy_returns = spy_df["close"].pct_change().dropna()
            spy_returns.index = pd.to_datetime(spy_returns.index)

            strat = strategy_returns.copy()
            strat.index = pd.to_datetime(strat.index)

            aligned = pd.concat(
                [strat.rename("strat"), spy_returns.rename("spy")], axis=1
            ).dropna()
            if len(aligned) < 5:
                return None, None

            cov_mat = aligned.cov()
            spy_var = cov_mat.loc["spy", "spy"]
            if spy_var < 1e-15:
                return None, None

            beta = float(cov_mat.loc["strat", "spy"] / spy_var)
            daily_rf = ANNUAL_RF_RATE / ANNUAL_TRADING_DAYS
            alpha_daily = (aligned["strat"] - daily_rf).mean() - beta * (
                aligned["spy"] - daily_rf
            ).mean()
            alpha_ann = float(alpha_daily * ANNUAL_TRADING_DAYS * 100)
            return alpha_ann, beta
        except Exception:
            return None, None

    def _compute_period_result(
        self,
        label: str,
        start: str,
        end: str,
        engine_metrics: PerformanceMetrics,
        portfolio_values: list[PortfolioValuePoint],
    ) -> PeriodResult:
        if not portfolio_values:
            return PeriodResult(
                label=label,
                start_date=start,
                end_date=end,
                initial_capital=self._initial_capital,
                final_value=self._initial_capital,
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                volatility_ann_pct=0.0,
                sharpe_ratio=None,
                sortino_ratio=None,
                max_drawdown_pct=None,
                max_drawdown_date=None,
                calmar_ratio=None,
                positive_days_pct=0.0,
                benchmark_return_pct=None,
                benchmark_ann_return_pct=None,
                alpha_ann_pct=None,
                beta=None,
                n_trading_days=0,
            )

        df = pd.DataFrame(portfolio_values).set_index("Date")
        df["daily_return"] = df["Portfolio Value"].pct_change()
        clean = df["daily_return"].dropna()

        initial_value = float(portfolio_values[0]["Portfolio Value"])
        final_value = float(portfolio_values[-1]["Portfolio Value"])
        total_return = (final_value / initial_value - 1.0) * 100.0
        n_days = len(clean)

        ann_return = self._annualize_return(total_return, n_days)
        volatility = float(clean.std() * np.sqrt(ANNUAL_TRADING_DAYS) * 100) if n_days > 1 else 0.0
        positive_days = float((clean > 0).mean() * 100) if n_days > 0 else 0.0

        max_dd = engine_metrics.get("max_drawdown")
        calmar = None
        if max_dd is not None and max_dd < 0:
            calmar = ann_return / abs(max_dd)

        # Benchmark
        bench_total = self._benchmark.get_return_pct("SPY", start, end)
        bench_ann = self._annualize_return(bench_total, n_days) if bench_total is not None else None

        # Alpha / Beta
        alpha, beta = self._compute_alpha_beta(clean, start, end)

        return PeriodResult(
            label=label,
            start_date=start,
            end_date=end,
            initial_capital=self._initial_capital,
            final_value=final_value,
            total_return_pct=total_return,
            annualized_return_pct=ann_return,
            volatility_ann_pct=volatility,
            sharpe_ratio=engine_metrics.get("sharpe_ratio"),
            sortino_ratio=engine_metrics.get("sortino_ratio"),
            max_drawdown_pct=max_dd,
            max_drawdown_date=engine_metrics.get("max_drawdown_date"),
            calmar_ratio=calmar,
            positive_days_pct=positive_days,
            benchmark_return_pct=bench_total,
            benchmark_ann_return_pct=bench_ann,
            alpha_ann_pct=alpha,
            beta=beta,
            n_trading_days=n_days,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> WalkForwardResult:
        """Execute IS and OOS backtests and return combined result."""
        is_metrics, is_values = self._run_period(self._is_start, self._is_end, "IN-SAMPLE")
        oos_metrics, oos_values = self._run_period(self._oos_start, self._oos_end, "OUT-OF-SAMPLE")

        is_result = self._compute_period_result(
            "IN-SAMPLE", self._is_start, self._is_end, is_metrics, is_values
        )
        oos_result = self._compute_period_result(
            "OUT-OF-SAMPLE", self._oos_start, self._oos_end, oos_metrics, oos_values
        )

        sharpe_decay = None
        if is_result.sharpe_ratio is not None and oos_result.sharpe_ratio is not None:
            sharpe_decay = is_result.sharpe_ratio - oos_result.sharpe_ratio

        return_retention = None
        if is_result.total_return_pct and oos_result.total_return_pct is not None:
            if is_result.total_return_pct != 0:
                return_retention = oos_result.total_return_pct / is_result.total_return_pct

        return WalkForwardResult(
            is_result=is_result,
            oos_result=oos_result,
            sharpe_decay=sharpe_decay,
            return_retention=return_retention,
        )

    def print_report(self, result: WalkForwardResult) -> None:
        """Print a Rich-formatted side-by-side comparison report."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
        except ImportError:
            self._print_report_plain(result)
            return

        console = Console()
        is_r = result.is_result
        oos_r = result.oos_result

        def fmt_pct(v: float | None, decimals: int = 2) -> str:
            if v is None:
                return "N/A"
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.{decimals}f}%"

        def fmt_float(v: float | None, decimals: int = 3) -> str:
            if v is None:
                return "N/A"
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.{decimals}f}"

        def color_val(text: str, is_good: bool | None) -> str:
            if is_good is True:
                return f"[green]{text}[/green]"
            if is_good is False:
                return f"[red]{text}[/red]"
            return f"[yellow]{text}[/yellow]"

        console.print("\n")
        console.rule("[bold cyan]WALK-FORWARD BACKTEST REPORT[/bold cyan]")
        console.print(f"  Tickers  : [bold]{', '.join(self._tickers)}[/bold]")
        console.print(f"  Model    : [bold]{self._model_provider} / {self._model_name}[/bold]")
        console.print(
            f"  IS period: [bold]{is_r.start_date}[/bold] to [bold]{is_r.end_date}[/bold]"
            f"  ({is_r.n_trading_days} trading days)"
        )
        console.print(
            f"  OOS period: [bold]{oos_r.start_date}[/bold] to [bold]{oos_r.end_date}[/bold]"
            f"  ({oos_r.n_trading_days} trading days)"
        )
        console.print()

        # Main metrics table
        table = Table(
            title="Performance Metrics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="bold", width=32)
        table.add_column("IN-SAMPLE", justify="right", width=18)
        table.add_column("OUT-OF-SAMPLE", justify="right", width=18)

        def add_row(label: str, is_val: str, oos_val: str, oos_good: bool | None = None) -> None:
            table.add_row(label, is_val, color_val(oos_val, oos_good))

        add_row("Initial Capital", f"${is_r.initial_capital:,.0f}", f"${oos_r.initial_capital:,.0f}")
        add_row("Final Portfolio Value", f"${is_r.final_value:,.0f}", f"${oos_r.final_value:,.0f}",
                oos_r.final_value >= oos_r.initial_capital)
        add_row("Total Return", fmt_pct(is_r.total_return_pct),
                fmt_pct(oos_r.total_return_pct), oos_r.total_return_pct > 0 if oos_r.total_return_pct is not None else None)
        add_row("Annualized Return", fmt_pct(is_r.annualized_return_pct),
                fmt_pct(oos_r.annualized_return_pct), oos_r.annualized_return_pct > 0 if oos_r.annualized_return_pct is not None else None)
        add_row("Annualized Volatility", fmt_pct(is_r.volatility_ann_pct),
                fmt_pct(oos_r.volatility_ann_pct), None)
        table.add_section()
        add_row("Sharpe Ratio", fmt_float(is_r.sharpe_ratio, 3),
                fmt_float(oos_r.sharpe_ratio, 3), oos_r.sharpe_ratio > 0 if oos_r.sharpe_ratio is not None else None)
        add_row("Sortino Ratio", fmt_float(is_r.sortino_ratio, 3),
                fmt_float(oos_r.sortino_ratio, 3), oos_r.sortino_ratio > 0 if oos_r.sortino_ratio is not None else None)
        add_row("Max Drawdown", fmt_pct(is_r.max_drawdown_pct),
                fmt_pct(oos_r.max_drawdown_pct), oos_r.max_drawdown_pct > -10 if oos_r.max_drawdown_pct is not None else None)
        add_row("Max Drawdown Date",
                is_r.max_drawdown_date or "N/A", oos_r.max_drawdown_date or "N/A")
        add_row("Calmar Ratio", fmt_float(is_r.calmar_ratio, 3),
                fmt_float(oos_r.calmar_ratio, 3), oos_r.calmar_ratio > 0 if oos_r.calmar_ratio is not None else None)
        add_row("% Positive Days", fmt_pct(is_r.positive_days_pct),
                fmt_pct(oos_r.positive_days_pct), oos_r.positive_days_pct > 50)
        table.add_section()
        add_row("SPY Buy-and-Hold Return", fmt_pct(is_r.benchmark_return_pct),
                fmt_pct(oos_r.benchmark_return_pct))
        add_row("SPY Annualized Return", fmt_pct(is_r.benchmark_ann_return_pct),
                fmt_pct(oos_r.benchmark_ann_return_pct))
        add_row("Alpha (annualized)", fmt_pct(is_r.alpha_ann_pct),
                fmt_pct(oos_r.alpha_ann_pct), oos_r.alpha_ann_pct > 0 if oos_r.alpha_ann_pct is not None else None)
        add_row("Beta", fmt_float(is_r.beta, 3),
                fmt_float(oos_r.beta, 3), None)

        console.print(table)

        # Overfitting diagnostics
        diag_table = Table(
            title="Overfitting Diagnostics",
            box=box.SIMPLE,
            header_style="bold yellow",
        )
        diag_table.add_column("Indicator", style="bold", width=32)
        diag_table.add_column("Value", justify="right", width=18)
        diag_table.add_column("Verdict", width=30)

        if result.sharpe_decay is not None:
            if result.sharpe_decay < 0.3:
                verdict = "[green]Minimal decay - robust[/green]"
            elif result.sharpe_decay < 0.8:
                verdict = "[yellow]Moderate decay - acceptable[/yellow]"
            else:
                verdict = "[red]High decay - likely overfitted[/red]"
            diag_table.add_row(
                "Sharpe Decay (IS - OOS)",
                fmt_float(result.sharpe_decay, 3),
                verdict,
            )

        if result.return_retention is not None:
            if result.return_retention >= 0.7:
                verdict = "[green]Strong retention[/green]"
            elif result.return_retention >= 0.3:
                verdict = "[yellow]Partial retention[/yellow]"
            elif result.return_retention >= 0:
                verdict = "[red]Weak retention[/red]"
            else:
                verdict = "[red]Sign reversal - overfit[/red]"
            diag_table.add_row(
                "Return Retention (OOS/IS)",
                fmt_float(result.return_retention, 3),
                verdict,
            )

        # OOS Sharpe > 0 = edge exists
        if oos_r.sharpe_ratio is not None:
            if oos_r.sharpe_ratio > 0.5:
                verdict = "[green]Clear positive edge[/green]"
            elif oos_r.sharpe_ratio > 0:
                verdict = "[yellow]Marginal edge[/yellow]"
            else:
                verdict = "[red]No edge detected[/red]"
            diag_table.add_row(
                "OOS Sharpe",
                fmt_float(oos_r.sharpe_ratio, 3),
                verdict,
            )

        console.print(diag_table)

        # Summary verdict
        console.print()
        self._print_verdict(console, result)
        console.print()

    def _print_verdict(self, console, result: WalkForwardResult) -> None:
        oos = result.oos_result
        is_profitable = oos.total_return_pct is not None and oos.total_return_pct > 0
        has_edge = oos.sharpe_ratio is not None and oos.sharpe_ratio > 0
        beats_benchmark = (
            oos.total_return_pct is not None
            and oos.benchmark_return_pct is not None
            and oos.total_return_pct > oos.benchmark_return_pct
        )
        low_decay = result.sharpe_decay is not None and result.sharpe_decay < 0.8

        score = sum([is_profitable, has_edge, beats_benchmark, low_decay])

        if score >= 3:
            verdict_color = "green"
            verdict_text = "PROMISING - Strategy shows genuine OOS edge"
        elif score == 2:
            verdict_color = "yellow"
            verdict_text = "MIXED - Strategy has potential but needs improvement"
        else:
            verdict_color = "red"
            verdict_text = "CAUTION - Strategy does not demonstrate robust OOS performance"

        console.print(
            f"  [bold {verdict_color}]VERDICT: {verdict_text}[/bold {verdict_color}]"
        )
        checks = [
            ("OOS profitable", is_profitable),
            ("OOS positive Sharpe", has_edge),
            ("Beats SPY buy-and-hold", beats_benchmark),
            ("Low IS/OOS Sharpe decay", low_decay),
        ]
        for check, passed in checks:
            icon = "[green]v[/green]" if passed else "[red]x[/red]"
            console.print(f"    {icon}  {check}")

    def _print_report_plain(self, result: WalkForwardResult) -> None:
        """Fallback plain-text report if Rich is not available."""
        is_r = result.is_result
        oos_r = result.oos_result

        def fmt(v, pct=False):
            if v is None:
                return "N/A"
            if pct:
                return f"{v:+.2f}%"
            return f"{v:.3f}"

        print("\n" + "=" * 60)
        print("WALK-FORWARD BACKTEST REPORT")
        print("=" * 60)
        rows = [
            ("Total Return", fmt(is_r.total_return_pct, True), fmt(oos_r.total_return_pct, True)),
            ("Annualized Return", fmt(is_r.annualized_return_pct, True), fmt(oos_r.annualized_return_pct, True)),
            ("Volatility (ann)", fmt(is_r.volatility_ann_pct, True), fmt(oos_r.volatility_ann_pct, True)),
            ("Sharpe Ratio", fmt(is_r.sharpe_ratio), fmt(oos_r.sharpe_ratio)),
            ("Sortino Ratio", fmt(is_r.sortino_ratio), fmt(oos_r.sortino_ratio)),
            ("Max Drawdown", fmt(is_r.max_drawdown_pct, True), fmt(oos_r.max_drawdown_pct, True)),
            ("Calmar Ratio", fmt(is_r.calmar_ratio), fmt(oos_r.calmar_ratio)),
            ("Alpha (ann)", fmt(is_r.alpha_ann_pct, True), fmt(oos_r.alpha_ann_pct, True)),
            ("Beta", fmt(is_r.beta), fmt(oos_r.beta)),
            ("SPY Return", fmt(is_r.benchmark_return_pct, True), fmt(oos_r.benchmark_return_pct, True)),
        ]
        print(f"{'Metric':<30} {'IN-SAMPLE':>15} {'OUT-OF-SAMPLE':>15}")
        print("-" * 62)
        for label, is_val, oos_val in rows:
            print(f"{label:<30} {is_val:>15} {oos_val:>15}")
        if result.sharpe_decay is not None:
            print(f"\nSharpe Decay (IS-OOS): {result.sharpe_decay:.3f}")

    def save_results(self, result: WalkForwardResult, output_dir: str = "results") -> str:
        """Persist results to a JSON file. Returns the file path."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"walk_forward_{timestamp}.json")

        data = {
            "generated_at": timestamp,
            "tickers": self._tickers,
            "model_provider": self._model_provider,
            "model_name": self._model_name,
            "is_period": asdict(result.is_result),
            "oos_period": asdict(result.oos_result),
            "diagnostics": {
                "sharpe_decay": result.sharpe_decay,
                "return_retention": result.return_retention,
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return filename
