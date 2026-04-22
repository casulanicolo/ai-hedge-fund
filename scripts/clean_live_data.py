"""
scripts/clean_live_data.py — Pre-produzione DB cleanup

Rimuove le predizioni "live/paper" accumulate con agenti instabili.
Preserva TUTTO il dato backtest (run_id LIKE 'backtest_seed_%').

Operazioni (in ordine, con commit unico finale):
  1. DELETE outcomes collegati a predizioni non-backtest
  2. DELETE predictions non-backtest
  3. DELETE agent_weights (ripartono da 1.0 su dati puliti)
  4. DELETE signal_cache (vecchi segnali cachati)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.init_db import get_connection

conn = get_connection()


def count(table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ── Prima ────────────────────────────────────────────────────────────────────
print("=== BEFORE ===")
print(f"  predictions  : {count('predictions'):>10,}")
print(f"  outcomes     : {count('outcomes'):>10,}")
print(f"  agent_weights: {count('agent_weights'):>10,}")
print(f"  signal_cache : {count('signal_cache'):>10,}")

# Quante righe backtest (da preservare)?
n_bt_pred = conn.execute(
    "SELECT COUNT(*) FROM predictions WHERE run_id LIKE 'backtest_seed_%'"
).fetchone()[0]
n_live_pred = conn.execute(
    "SELECT COUNT(*) FROM predictions WHERE run_id NOT LIKE 'backtest_seed_%'"
).fetchone()[0]
n_live_out = conn.execute(
    "SELECT COUNT(*) FROM outcomes WHERE prediction_id IN "
    "(SELECT id FROM predictions WHERE run_id NOT LIKE 'backtest_seed_%')"
).fetchone()[0]

print(f"\n  backtest predictions (KEEP) : {n_bt_pred:>10,}")
print(f"  live predictions   (DELETE): {n_live_pred:>10,}")
print(f"  live outcomes      (DELETE): {n_live_out:>10,}")

# ── Esecuzione ───────────────────────────────────────────────────────────────
print("\n=== EXECUTING ===")

conn.execute(
    "DELETE FROM outcomes WHERE prediction_id IN "
    "(SELECT id FROM predictions WHERE run_id NOT LIKE 'backtest_seed_%')"
)
print(f"  outcomes deleted : {conn.execute('SELECT changes()').fetchone()[0]:>10,}")

conn.execute("DELETE FROM predictions WHERE run_id NOT LIKE 'backtest_seed_%'")
print(f"  predictions deleted: {conn.execute('SELECT changes()').fetchone()[0]:>10,}")

conn.execute("DELETE FROM agent_weights")
print(f"  agent_weights deleted: {conn.execute('SELECT changes()').fetchone()[0]:>8,}")

conn.execute("DELETE FROM signal_cache")
print(f"  signal_cache deleted: {conn.execute('SELECT changes()').fetchone()[0]:>9,}")

conn.commit()
conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
conn.close()

# ── Dopo ─────────────────────────────────────────────────────────────────────
conn2 = get_connection()
print("\n=== AFTER ===")
print(f"  predictions  : {conn2.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]:>10,}")
print(f"  outcomes     : {conn2.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]:>10,}")
print(f"  agent_weights: {conn2.execute('SELECT COUNT(*) FROM agent_weights').fetchone()[0]:>10,}")
print(f"  signal_cache : {conn2.execute('SELECT COUNT(*) FROM signal_cache').fetchone()[0]:>10,}")
conn2.close()

print("\n=== DONE — DB pronto per produzione ===")
