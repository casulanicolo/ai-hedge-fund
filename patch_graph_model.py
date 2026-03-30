"""
patch_graph_model.py
Aggiunge model_name e model_provider Anthropic al make_initial_state
nel blocco __main__ di src/graph/graph.py
"""

path = "src/graph/graph.py"
txt = open(path, encoding="utf-8").read()

OLD = (
    '    state = make_initial_state(\n'
    '        run_id=str(uuid.uuid4()),\n'
    '        tickers=["AAPL", "NVDA"],\n'
    '        start_ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),\n'
    '    )'
)

NEW = (
    '    state = make_initial_state(\n'
    '        run_id=str(uuid.uuid4()),\n'
    '        tickers=["AAPL", "NVDA"],\n'
    '        start_ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),\n'
    '    )\n'
    '    # Inject Anthropic model so all agents use the right provider\n'
    '    state["metadata"]["model_name"] = "claude-sonnet-4-5"\n'
    '    state["metadata"]["model_provider"] = "Anthropic"'
)

if OLD in txt:
    txt = txt.replace(OLD, NEW, 1)
    open(path, "w", encoding="utf-8").write(txt)
    print("✓ graph.py patched — Anthropic model injected into test state")
elif 'state["metadata"]["model_provider"] = "Anthropic"' in txt:
    print("✓ Already patched")
else:
    print("ERROR: could not find target block — printing context:")
    idx = txt.find("make_initial_state")
    print(repr(txt[idx:idx+400]))
