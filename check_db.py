"""
Diagnostica rapida del database SQLite di Athanor Alpha.
Uso: python check_db.py
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("db/hedge_fund.db")

if not DB_PATH.exists():
    print(f"[ERRORE] Database non trovato: {DB_PATH}")
    raise SystemExit(1)

conn = sqlite3.connect(DB_PATH)

# Tabelle presenti
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()
print("=== TABELLE NEL DATABASE ===")
for (t,) in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {count} righe")

# Ultime predictions
print("\n=== ULTIME 5 PREDICTIONS ===")
try:
    rows = conn.execute(
        "SELECT run_id, agent_id, ticker, signal, confidence, created_at "
        "FROM predictions ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    if rows:
        for r in rows:
            print(f"  {r}")
    else:
        print("  (nessuna prediction ancora)")
except Exception as e:
    print(f"  [WARN] {e}")

# Agent weights
print("\n=== AGENT WEIGHTS (campione) ===")
try:
    rows = conn.execute(
        "SELECT agent_id, ticker, weight, updated_at "
        "FROM agent_weights ORDER BY updated_at DESC LIMIT 10"
    ).fetchall()
    if rows:
        for r in rows:
            print(f"  {r}")
    else:
        print("  (nessun peso ancora — verranno creati al primo run)")
except Exception as e:
    print(f"  [WARN] {e}")

conn.close()
print("\n[OK] Diagnostica completata.")
