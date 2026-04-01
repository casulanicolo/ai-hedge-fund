"""Patcha run_pipeline.py per aggiungere il routing multi-provider."""
from pathlib import Path

pipeline = Path("src/run_pipeline.py")
text = pipeline.read_text(encoding="utf-8")

old = '        "metadata": {'
new = '''        "metadata": {
            # Routing multi-provider (agent_id -> model)
            "agent_model_map": _load_agent_model_map(),'''

if '"agent_model_map"' in text:
    print("SKIP: agent_model_map gia presente.")
else:
    text = text.replace(old, new, 1)
    pipeline.write_text(text, encoding="utf-8")
    print("OK: agent_model_map aggiunto a metadata.")
