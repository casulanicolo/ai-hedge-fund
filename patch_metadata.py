"""
patch_metadata.py
Aggiunge show_reasoning (e altri campi comuni) al metadata
in make_initial_state dentro src/graph/state.py
"""

path = "src/graph/state.py"
txt = open(path, encoding="utf-8").read()

OLD = (
    '        metadata={\n'
    '            "run_id":   run_id,\n'
    '            "tickers":  tickers,\n'
    '            "start_ts": start_ts,\n'
    '        },'
)

NEW = (
    '        metadata={\n'
    '            "run_id":         run_id,\n'
    '            "tickers":        tickers,\n'
    '            "start_ts":       start_ts,\n'
    '            "show_reasoning": False,\n'
    '            "model_name":     "claude-sonnet-4-5",\n'
    '            "model_provider": "Anthropic",\n'
    '        },'
)

if OLD in txt:
    txt = txt.replace(OLD, NEW, 1)
    open(path, "w", encoding="utf-8").write(txt)
    print("✓ metadata aggiornato in state.py")
elif '"show_reasoning"' in txt:
    print("✓ show_reasoning già presente")
else:
    print("WARNING: blocco metadata non trovato — stampo contesto:")
    idx = txt.find('"run_id"')
    print(repr(txt[max(0,idx-50):idx+200]))

try:
    compile(open(path, encoding="utf-8").read(), path, "exec")
    print("✓ Syntax check OK")
except SyntaxError as e:
    print(f"ERROR: {e}")
