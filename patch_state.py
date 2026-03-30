"""
patch_state.py
Adds tickers, end_date, start_date to data{} inside make_initial_state
so all legacy philosophy agents find them without crashing.
Run once from the project root:
    python patch_state.py
"""
import re
from datetime import date

path = "src/graph/state.py"
txt = open(path, encoding="utf-8").read()

OLD = '''        data={
            "prefetched_data":  {},
            "analyst_signals":  {},
            "feedback_history": feedback_history or {},
            "agent_weights":    agent_weights or {},
            "risk_output":      {},
            "portfolio_output": [],
        },'''

NEW = '''        data={
            "prefetched_data":  {},
            "analyst_signals":  {},
            "feedback_history": feedback_history or {},
            "agent_weights":    agent_weights or {},
            "risk_output":      {},
            "portfolio_output": [],
            # Convenience keys read directly by legacy philosophy agents
            "tickers":   tickers,
            "end_date":  end_date or date.today().isoformat(),
            "start_date": start_date,
        },'''

if OLD not in txt:
    print("ERROR: could not find the target block — state.py may have changed.")
    print("Looking for:")
    print(repr(OLD))
    exit(1)

txt = txt.replace(OLD, NEW, 1)

# Also add end_date and start_date parameters to the function signature
OLD_SIG = '''def make_initial_state(
    run_id: str,
    tickers: list[str],
    start_ts: str,
    feedback_history: Optional[FeedbackHistory] = None,
    agent_weights: Optional[AgentWeights] = None,
) -> AgentState:'''

NEW_SIG = '''def make_initial_state(
    run_id: str,
    tickers: list[str],
    start_ts: str,
    feedback_history: Optional[FeedbackHistory] = None,
    agent_weights: Optional[AgentWeights] = None,
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> AgentState:'''

if OLD_SIG in txt:
    txt = txt.replace(OLD_SIG, NEW_SIG, 1)
    print("✓ Function signature updated")
else:
    print("WARNING: could not update function signature — may already be updated")

# Add date import if missing
if "from datetime import" not in txt:
    txt = "from datetime import date\n" + txt
    print("✓ Added datetime import")
elif "from datetime import" in txt and "date" not in txt.split("from datetime import")[1].split("\n")[0]:
    txt = txt.replace("from datetime import", "from datetime import date, ", 1)
    print("✓ Added date to existing datetime import")
else:
    print("✓ datetime.date already imported")

open(path, "w", encoding="utf-8").write(txt)
print("✓ src/graph/state.py patched successfully")

# Quick verification
exec_ns = {}
try:
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), exec_ns)
    print("✓ Syntax check passed")
except SyntaxError as e:
    print(f"ERROR: syntax error after patch: {e}")
