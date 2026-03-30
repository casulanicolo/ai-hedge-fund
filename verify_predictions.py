import sqlite3

conn = sqlite3.connect('db/hedge_fund.db')

print("=== Ultimi run loggati ===")
cursor = conn.execute("""
    SELECT run_id, COUNT(*) as rows, MIN(timestamp) as ts
    FROM predictions
    GROUP BY run_id
    ORDER BY ts DESC
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"  run_id={row[0][:8]}...  righe={row[1]}  ts={row[2]}")

print()
print("=== Segnali dell'ultimo run ===")
cursor = conn.execute("""
    SELECT agent_id, ticker, signal, ROUND(confidence,2) as conf
    FROM predictions
    WHERE run_id = (SELECT run_id FROM predictions ORDER BY timestamp DESC LIMIT 1)
    ORDER BY agent_id, ticker
""")
for row in cursor.fetchall():
    print(f"  {row[0]:30s}  {row[1]:6s}  {row[2]:4s}  conf={row[3]:.2f}")

conn.close()
