"""CLI entry point for walk-forward IS/OOS backtesting.

Examples
--------
# Interactive model/analyst selection + custom dates:
poetry run python src/walk_forward_backtest.py \\
    --tickers AAPL,MSFT,NVDA \\
    --is-start 2023-01-01 --is-end 2023-12-31 \\
    --oos-start 2024-01-01 --oos-end 2024-12-31

# Fully automated (no prompts):
poetry run python src/walk_forward_backtest.py \\
    --tickers AAPL,MSFT,NVDA \\
    --is-start 2023-01-01 --is-end 2023-12-31 \\
    --oos-start 2024-01-01 --oos-end 2024-12-31 \\
    --model gpt-4o \\
    --analysts-all \\
    --no-save

# Short IS + OOS (cheaper, uses Ollama):
poetry run python src/walk_forward_backtest.py \\
    --tickers AAPL,MSFT \\
    --is-start 2024-01-01 --is-end 2024-06-30 \\
    --oos-start 2024-07-01 --oos-end 2024-12-31 \\
    --ollama \\
    --initial-capital 50000
"""

from __future__ import annotations

import sys
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from colorama import Fore, Style

from src.main import run_hedge_fund
from src.backtesting.walk_forward import WalkForwardAnalyzer
from src.cli.input import (
    parse_tickers,
    select_analysts,
    select_model,
    resolve_dates,
)
from src.utils.analysts import ANALYST_ORDER



# ------------------------------------------------------------------ #
# Argument parsing
# ------------------------------------------------------------------ #

def _build_parser() -> argparse.ArgumentParser:
    today = datetime.now()
    default_oos_end = today.strftime("%Y-%m-%d")
    default_oos_start = (today - relativedelta(months=6)).strftime("%Y-%m-%d")
    default_is_end = (today - relativedelta(months=6, days=1)).strftime("%Y-%m-%d")
    default_is_start = (today - relativedelta(months=12)).strftime("%Y-%m-%d")

    p = argparse.ArgumentParser(
        description="Walk-forward in-sample / out-of-sample backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Tickers
    p.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list (e.g. AAPL,MSFT). Reads config/tickers.yaml if omitted.",
    )

    # IS period
    p.add_argument(
        "--is-start",
        type=str,
        default=default_is_start,
        metavar="YYYY-MM-DD",
        help=f"In-sample start date (default: {default_is_start})",
    )
    p.add_argument(
        "--is-end",
        type=str,
        default=default_is_end,
        metavar="YYYY-MM-DD",
        help=f"In-sample end date (default: {default_is_end})",
    )

    # OOS period
    p.add_argument(
        "--oos-start",
        type=str,
        default=default_oos_start,
        metavar="YYYY-MM-DD",
        help=f"Out-of-sample start date (default: {default_oos_start})",
    )
    p.add_argument(
        "--oos-end",
        type=str,
        default=default_oos_end,
        metavar="YYYY-MM-DD",
        help=f"Out-of-sample end date (default: {default_oos_end})",
    )

    # Model / analyst selection
    p.add_argument("--model", type=str, default=None, help="LLM model name (e.g. gpt-4o)")
    p.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")
    p.add_argument("--analysts", type=str, default=None, help="Comma-separated analyst IDs")
    p.add_argument("--analysts-all", action="store_true", help="Use all available analysts")

    # Capital
    p.add_argument(
        "--initial-capital",
        dest="initial_capital",
        type=float,
        default=100_000.0,
        help="Initial capital for each period (default: 100000)",
    )
    p.add_argument(
        "--margin-requirement",
        dest="margin_requirement",
        type=float,
        default=0.0,
        help="Margin requirement ratio for short positions (default: 0.0)",
    )

    # Output
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to disk",
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="results",
        help="Directory to save JSON results (default: results/)",
    )

    return p


def _validate_dates(is_start: str, is_end: str, oos_start: str, oos_end: str) -> None:
    fmt = "%Y-%m-%d"
    for name, val in [
        ("--is-start", is_start),
        ("--is-end", is_end),
        ("--oos-start", oos_start),
        ("--oos-end", oos_end),
    ]:
        try:
            datetime.strptime(val, fmt)
        except ValueError:
            print(f"{Fore.RED}Invalid date for {name}: '{val}'. Use YYYY-MM-DD.{Style.RESET_ALL}")
            sys.exit(1)

    if is_start >= is_end:
        print(f"{Fore.RED}--is-start must be before --is-end.{Style.RESET_ALL}")
        sys.exit(1)
    if oos_start >= oos_end:
        print(f"{Fore.RED}--oos-start must be before --oos-end.{Style.RESET_ALL}")
        sys.exit(1)
    if is_end >= oos_start:
        print(
            f"{Fore.YELLOW}Warning: IS period ({is_end}) overlaps with or touches OOS period "
            f"({oos_start}). For a clean walk-forward test the IS end date should be strictly "
            f"before the OOS start date.{Style.RESET_ALL}"
        )


def _resolve_tickers(tickers_arg: str | None) -> list[str]:
    if tickers_arg:
        return parse_tickers(tickers_arg)
    # Fallback: read from config/tickers.yaml
    try:
        import yaml
        with open("config/tickers.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tickers = cfg.get("tickers", [])
        if tickers:
            print(
                f"{Fore.CYAN}No --tickers provided. "
                f"Using config/tickers.yaml: {', '.join(tickers)}{Style.RESET_ALL}"
            )
            return tickers
    except Exception:
        pass
    print(f"{Fore.RED}No tickers specified and config/tickers.yaml not found.{Style.RESET_ALL}")
    sys.exit(1)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve tickers
    tickers = _resolve_tickers(args.tickers)

    # Validate dates
    _validate_dates(args.is_start, args.is_end, args.oos_start, args.oos_end)

    # Select model (interactive if not provided)
    model_name, model_provider = select_model(args.ollama, args.model)

    # Select analysts (interactive if not provided)
    selected_analysts = select_analysts({
        "analysts_all": args.analysts_all,
        "analysts": args.analysts,
    })

    print(f"\n{Fore.CYAN}Walk-Forward Backtest Configuration{Style.RESET_ALL}")
    print(f"  Tickers        : {Fore.GREEN}{', '.join(tickers)}{Style.RESET_ALL}")
    print(f"  In-sample      : {args.is_start}  ->  {args.is_end}")
    print(f"  Out-of-sample  : {args.oos_start}  ->  {args.oos_end}")
    print(f"  Model          : {Fore.GREEN}{model_provider}/{model_name}{Style.RESET_ALL}")
    print(f"  Analysts       : {len(selected_analysts)} selected")
    print(f"  Initial capital: ${args.initial_capital:,.0f}")

    # Run walk-forward analysis
    analyzer = WalkForwardAnalyzer(
        agent=run_hedge_fund,
        tickers=tickers,
        is_start=args.is_start,
        is_end=args.is_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
        initial_capital=args.initial_capital,
        model_name=model_name,
        model_provider=model_provider,
        selected_analysts=selected_analysts,
        margin_requirement=args.margin_requirement,
    )

    try:
        result = analyzer.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Walk-forward backtest interrupted by user.{Style.RESET_ALL}")
        sys.exit(0)

    # Print report
    analyzer.print_report(result)

    # Save results
    if not args.no_save:
        filepath = analyzer.save_results(result, output_dir=args.output_dir)
        print(f"\n{Fore.GREEN}Results saved to: {filepath}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
