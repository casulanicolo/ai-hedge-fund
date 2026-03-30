"""
insert_test_predictions.py

Inserisce 4 predizioni con date passate nel DB per testare l'outcome tracker.
Le date sono abbastanza indietro da avere T+1d, T+5d e T+20d già disponibili.

Esegui dalla root del progetto: python insert_test_predictions.py
"""

import sqlite3
import os
import uuid

DB_PATH = os.path.join("db", "hedge_fund.db")

# Predizioni di test con date passate (gennaio-febbraio 2026)
# Usiamo ticker reali e date in cui il mercato era aperto
TEST_PREDICTIONS = [
    {
        "run_id":        str(uuid.uuid4()),
        "agent_id":      "warren_buffett_agent",
        "ticker":        "AAPL",
        "signal":        "BUY",
        "confidence":    0.75,
        "reasoning_hash": "test_hash_001",
        "timestamp":     "2026-01-05T15:00:00+00:00",  # lunedì 5 gennaio 2026
    },
    {
        "run_id":        str(uuid.uuid4()),
        "agent_id":      "warren_buffett_agent",
        "ticker":        "NVDA",
        "signal":        "SELL",
        "confidence":    0.60,
        "reasoning_hash": "test_hash_002",
        "timestamp":     "2026-01-05T15:00:00+00:00",
    },
    {
        "run_id":        str(uuid.uuid4()),
        "agent_id":      "ben_graham_agent",
        "ticker":        "AAPL",
        "signal":        "HOLD",
        "confidence":    0.50,
        "reasoning_hash": "test_hash_003",
        "timestamp":     "2026-02-02T15:00:00+00:00",  # lunedì 2 febbraio 2026
    },
    {
        "run_id":        str(uuid.uuid4()),
        "agent_id":      "ben_graham_agent",
        "ticker":        "NVDA",
        "signal":        "BUY",
        "confidence":    0.80,
        "reasoning_hash": "test_hash_004",
        "timestamp":     "2026-02-02T15:00:00+00:00",
    },
]


def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database non trovato: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    print("Inserisco predizioni di test con date passate...")
    inserted_ids = []

    for pred in TEST_PREDICTIONS:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (run_id, agent_id, ticker, signal, confidence, reasoning_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pred["run_id"],
                pred["agent_id"],
                pred["ticker"],
                pred["signal"],
                pred["confidence"],
                pred["reasoning_hash"],
                pred["timestamp"],
            ),
        )
        inserted_ids.append(cursor.lastrowid)
        print(f"  ✅ Inserita prediction_id={cursor.lastrowid} | {pred['agent_id']} | {pred['ticker']} | {pred['signal']} | {pred['timestamp'][:10]}")

    conn.commit()
    conn.close()

    total = conn.execute if False else None  # non usato, solo per chiarezza

    print(f"\n✅ Inserite {len(inserted_ids)} predizioni di test (id: {inserted_ids})")
    print("Ora esegui: python -m src.feedback.outcome_tracker")


if __name__ == "__main__":
    main()
