"""Diagnosi schema DB Athanor.
Lancia con: .\.venv\Scripts\python.exe scripts\list_tables.py
"""
import sqlite3
import json
from pathlib import Path

DB_PATH = Path("db/hedge_fund.db")

if not DB_PATH.exists():
    print(f"[ERRORE] Database non trovato: {DB_PATH.resolve()}")
    raise SystemExit(1)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1. Lista tabelle
tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]

print("=" * 60)
print("TABELLE PRESENTI NEL DB")
print("=" * 60)
for t in tables:
    count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  - {t:40s} ({count} righe)")

# 2. Verifica se portfolio_decisions esiste
print()
print("=" * 60)
print("CHECK TABELLA portfolio_decisions")
print("=" * 60)
if "portfolio_decisions" in tables:
    print("  [OK] La tabella ESISTE nel DB.")
    schema = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='portfolio_decisions'"
    ).fetchone()
    if schema:
        print("  Schema attuale:")
        for line in schema[0].splitlines():
            print(f"    {line}")
else:
    print("  [MANCA] La tabella NON esiste nel DB.")
    print("  Il codice in init_db.py la legge/scrive ma nessuno l'ha creata.")

conn.close()
