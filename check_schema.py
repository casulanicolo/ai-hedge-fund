"""
Ispeziona lo schema reale di tutte le tabelle.
Uso: python check_schema.py
"""
import sqlite3
from pathlib import Path

conn = sqlite3.connect("db/hedge_fund.db")

tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()

for (t,) in tables:
    if t == "sqlite_sequence":
        continue
    print(f"\n=== {t} ===")
    cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
    for c in cols:
        print(f"  col[{c[0]}] {c[1]:30s} type={c[2]}")
    # mostra prime 3 righe
    rows = conn.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
    if rows:
        print(f"  --- prime {len(rows)} righe ---")
        for r in rows:
            print(f"  {r}")
    else:
        print("  (vuota)")

conn.close()
