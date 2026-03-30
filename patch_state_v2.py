"""
patch_state_v2.py — fixed version
Inserisce datetime import DOPO from __future__, non prima.
"""
from datetime import date

path = "src/graph/state.py"
txt = open(path, encoding="utf-8").read()

# ── 1. Fix datetime import position ──────────────────────────────────────────
# Remove any datetime import we may have wrongly added at the top
if txt.startswith("from datetime import date\n"):
    txt = txt[len("from datetime import date\n"):]
    print("✓ Removed misplaced datetime import from top")

# Insert after the __future__ import line
FUTURE_LINE = "from __future__ import annotations"
if FUTURE_LINE in txt and "from datetime import" not in txt:
    txt = txt.replace(
        FUTURE_LINE,
        FUTURE_LINE + "\nfrom datetime import date",
        1
    )
    print("✓ Inserted datetime import after __future__")
elif "from datetime import" in txt:
    # Make sure 'date' is in the import
    import re
    m = re.search(r"from datetime import ([^\n]+)", txt)
    if m and "date" not in m.group(1):
        txt = txt.replace(m.group(0), m.group(0) + ", date", 1)
        print("✓ Added date to existing datetime import")
    else:
        print("✓ datetime.date already imported")

# ── 2. Fix function signature (add end_date / start_date params) ──────────────
OLD_SIG = (
    "def make_initial_state(\n"
    "    run_id: str,\n"
    "    tickers: list[str],\n"
    "    start_ts: str,\n"
    "    feedback_history: Optional[FeedbackHistory] = None,\n"
    "    agent_weights: Optional[AgentWeights] = None,\n"
    ") -> AgentState:"
)
NEW_SIG = (
    "def make_initial_state(\n"
    "    run_id: str,\n"
    "    tickers: list[str],\n"
    "    start_ts: str,\n"
    "    feedback_history: Optional[FeedbackHistory] = None,\n"
    "    agent_weights: Optional[AgentWeights] = None,\n"
    "    end_date: Optional[str] = None,\n"
    "    start_date: Optional[str] = None,\n"
    ") -> AgentState:"
)

if OLD_SIG in txt:
    txt = txt.replace(OLD_SIG, NEW_SIG, 1)
    print("✓ Function signature updated")
elif "end_date: Optional[str] = None" in txt:
    print("✓ Signature already updated")
else:
    print("WARNING: could not find signature — check state.py manually")

# ── 3. Add keys to data dict ──────────────────────────────────────────────────
OLD_DATA = (
    '        data={\n'
    '            "prefetched_data":  {},\n'
    '            "analyst_signals":  {},\n'
    '            "feedback_history": feedback_history or {},\n'
    '            "agent_weights":    agent_weights or {},\n'
    '            "risk_output":      {},\n'
    '            "portfolio_output": [],\n'
    '        },'
)
NEW_DATA = (
    '        data={\n'
    '            "prefetched_data":  {},\n'
    '            "analyst_signals":  {},\n'
    '            "feedback_history": feedback_history or {},\n'
    '            "agent_weights":    agent_weights or {},\n'
    '            "risk_output":      {},\n'
    '            "portfolio_output": [],\n'
    '            # Convenience keys read directly by legacy philosophy agents\n'
    '            "tickers":    tickers,\n'
    '            "end_date":   end_date or date.today().isoformat(),\n'
    '            "start_date": start_date,\n'
    '        },'
)

if OLD_DATA in txt:
    txt = txt.replace(OLD_DATA, NEW_DATA, 1)
    print("✓ data{} keys added")
elif '"end_date":   end_date' in txt:
    print("✓ data{} keys already present")
else:
    print("WARNING: could not find data{} block — check state.py manually")

# ── 4. Write and verify ───────────────────────────────────────────────────────
open(path, "w", encoding="utf-8").write(txt)
print("✓ src/graph/state.py written")

try:
    compile(open(path, encoding="utf-8").read(), path, "exec")
    print("✓ Syntax check passed")
except SyntaxError as e:
    print(f"ERROR: syntax error: {e}")
    print("First 20 lines of file:")
    for i, line in enumerate(open(path, encoding="utf-8").readlines()[:20], 1):
        print(f"  {i:3}: {line}", end="")
