"""
src/backtesting/cli.py
──────────────────────
Forward-only backtest CLI.

Usage
-----
  python -m src.backtesting.cli --start 2024-01-02 --end 2024-01-31 \\
                                --tickers AAPL,MSFT --capital 100000

  python -m src.backtesting.cli --start 2023-01-01 --end 2025-12-31 \\
                                --tickers all --capital 100000

  python -m src.backtesting.cli --walk-forward \\
                                --is-start 2019-01-02 --is-end 2022-12-30 \\
                                --oos-start 2023-01-02 --oos-end 2025-12-31

`--tickers all` reads config/tickers.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.backtesting.forward_engine import ForwardBacktestEngine

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[2]
TICKERS_CONFIG = ROOT / "config" / "tickers.yaml"


def _load_all_tickers() -> list[str]:
    with TICKERS_CONFIG.open(encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, list):
        return cfg
    if isinstance(cfg, dict):
        return cfg.get("tickers", [])
    raise ValueError(f"unrecognised tickers.yaml shape: {type(cfg)}")


def _parse_tickers(arg: str) -> list[str]:
    if arg.lower() == "all":
        return _load_all_tickers()
    return [t.strip().upper() for t in arg.split(",") if t.strip()]


def _print_report(report: dict) -> None:
    base = report["metrics_base"]
    ext  = report["metrics_extended"]
    print()
    print("=" * 60)
    print("FORWARD BACKTEST RESULTS")
    print("=" * 60)
    print(f"Trading days     : {report['n_trading_days']}")
    print(f"Initial capital  : ${report['initial_capital']:,.2f}")
    print(f"Final equity     : ${report['final_value']:,.2f}")
    print(f"Total return     : {report['total_return_pct']:+.2f}%")
    print(f"Closed trades    : {report['n_closed_trades']}")
    print(f"Open positions   : {report['n_open_positions']}")
    print("-" * 60)
    print("Risk-adjusted")
    print(f"  Sharpe         : {_fmt(base.get('sharpe_ratio'))}")
    print(f"  Sortino        : {_fmt(base.get('sortino_ratio'))}")
    print(f"  Calmar         : {_fmt(ext.get('calmar_ratio'))}")
    print(f"  Omega          : {_fmt(ext.get('omega_ratio'))}")
    print(f"  Tail ratio     : {_fmt(ext.get('tail_ratio'))}")
    print("Path-shape")
    print(f"  Max drawdown   : {_fmt(base.get('max_drawdown'), pct=True)}")
    print(f"  Ulcer index    : {_fmt(ext.get('ulcer_index'))}")
    print(f"  Time u/water   : {ext.get('time_under_water_days')} days")
    print(f"  Win streak     : {ext.get('longest_win_streak')}")
    print(f"  Loss streak    : {ext.get('longest_loss_streak')}")
    print(f"  % positive days: {ext.get('pct_positive_days'):.1f}%")
    print("Trade-level")
    print(f"  Profit factor  : {_fmt(ext.get('profit_factor'))}")
    print(f"  R-multiple avg : {_fmt(ext.get('r_multiple_avg'))}")
    print("=" * 60)


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "N/A"
    if pct:
        return f"{v:+.2f}%"
    return f"{v:+.3f}"


def _save_report(report: dict, label: str) -> Path:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"forward_{label}_{ts}.json"
    safe = {
        k: v for k, v in report.items()
        if k not in ("equity_curve",)   # potentially huge → write separately
    }
    safe["equity_curve_count"] = len(report.get("equity_curve", []))
    path.write_text(json.dumps(safe, default=str, indent=2), encoding="utf-8")

    eq_path = out_dir / f"forward_{label}_{ts}_equity.csv"
    rows = ["Date,PortfolioValue"]
    for p in report.get("equity_curve", []):
        rows.append(f"{p['Date'].date() if hasattr(p['Date'], 'date') else p['Date']},{p['Portfolio Value']:.2f}")
    eq_path.write_text("\n".join(rows), encoding="utf-8")

    print(f"Saved: {path.name} + {eq_path.name}")
    return path


# ── main ──────────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    p = argparse.ArgumentParser(description="Athanor Alpha — Forward-only backtest")
    p.add_argument("--start", type=str, help="YYYY-MM-DD start date")
    p.add_argument("--end",   type=str, help="YYYY-MM-DD end date")
    p.add_argument("--tickers", type=str, default="all",
                   help="Comma-separated tickers, or 'all' for config/tickers.yaml")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--no-cache", action="store_true", help="Disable state cache")
    p.add_argument("--label", type=str, default="run", help="Label for results filenames")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run IS + OOS comparison (uses --is-start/--is-end/--oos-start/--oos-end)")
    p.add_argument("--is-start",  type=str, default="2019-01-02")
    p.add_argument("--is-end",    type=str, default="2022-12-30")
    p.add_argument("--oos-start", type=str, default="2023-01-02")
    p.add_argument("--oos-end",   type=str, default=datetime.now().strftime("%Y-%m-%d"))
    args = p.parse_args()

    tickers = _parse_tickers(args.tickers)
    if not tickers:
        print("No tickers resolved.", file=sys.stderr)
        return 1

    if args.walk_forward:
        from src.backtesting.walk_forward import run_walk_forward
        result = run_walk_forward(
            tickers=tickers, capital=args.capital,
            is_start=args.is_start, is_end=args.is_end,
            oos_start=args.oos_start, oos_end=args.oos_end,
            use_cache=not args.no_cache,
        )
        _print_report(result["is"]);  print("\n>>> ABOVE: IN-SAMPLE\n")
        _print_report(result["oos"]); print("\n>>> ABOVE: OUT-OF-SAMPLE\n")
        print(f"Sharpe decay (IS-OOS) : {_fmt(result.get('sharpe_decay'))}")
        print(f"Return retention      : {_fmt(result.get('return_retention'))}")
        _save_report(result["is"],  f"{args.label}_IS")
        _save_report(result["oos"], f"{args.label}_OOS")
        return 0

    if not args.start or not args.end:
        print("--start and --end required (unless --walk-forward).", file=sys.stderr)
        return 1

    engine = ForwardBacktestEngine(
        tickers=tickers, start=args.start, end=args.end,
        initial_capital=args.capital, use_cache=not args.no_cache,
    )
    report = engine.run()
    _print_report(report)
    _save_report(report, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
