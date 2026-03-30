"""
fix_api_key.py
--------------
Sostituisce api_key=api_key con api_key=None in tutti i philosophy agents.
Esegui una volta sola dalla root del progetto.
"""
import os

FILES = [
    "src/agents/warren_buffett.py",
    "src/agents/ben_graham.py",
    "src/agents/bill_ackman.py",
    "src/agents/cathie_wood.py",
    "src/agents/charlie_munger.py",
    "src/agents/michael_burry.py",
    "src/agents/mohnish_pabrai.py",
    "src/agents/peter_lynch.py",
    "src/agents/phil_fisher.py",
    "src/agents/rakesh_jhunjhunwala.py",
    "src/agents/stanley_druckenmiller.py",
    "src/agents/aswath_damodaran.py",
    "src/agents/valuation.py",
    "src/agents/growth_agent.py",
]

for path in FILES:
    if not os.path.exists(path):
        print(f"SKIP (non trovato): {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    fixed = original.replace("api_key=api_key", "api_key=None")
    count = original.count("api_key=api_key")

    if count == 0:
        print(f"OK (nessuna modifica): {path}")
        continue

    with open(path, "w", encoding="utf-8") as f:
        f.write(fixed)

    print(f"FIXED ({count} sostituzioni): {path}")

print("\nDone.")
