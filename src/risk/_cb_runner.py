"""
src/risk/_cb_runner.py — Fase 8
CLI entry point for circuit breaker checks (called from PS1 cron).
Exit 0 = all OK or only soft alerts. Exit 2 = hard CB triggered.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from src.risk.circuit_breakers import check_all, any_halt_opens, any_halt_all

try:
    from src.execution.alpaca_adapter import AlpacaBrokerAdapter
    adapter = AlpacaBrokerAdapter()
except Exception:
    adapter = None

statuses = check_all(adapter)

hard_triggered = False
for s in statuses:
    tag = "TRIGGERED" if s.triggered else "OK"
    print(f"[{s.cb_id}] {tag} [{s.severity}] — {s.reason}")
    if s.triggered and s.severity == "CRITICAL":
        hard_triggered = True

if any_halt_all(statuses):
    print("\n>>> CB5 ACTIVE: ALL new orders halted <<<")
elif any_halt_opens(statuses):
    print("\n>>> CB active: OPEN orders halted <<<")

sys.exit(2 if hard_triggered else 0)
