"""Patcha llm.py per leggere agent_model_map dallo stato."""
from pathlib import Path

llm_file = Path("src/utils/llm.py")
text = llm_file.read_text(encoding="utf-8")

old = '''    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "gpt-4.1"
    model_provider = state.get("metadata", {}).get("model_provider") or "OPENAI"'''

new = '''    # Cerca nella mappa di routing per agente
    agent_model_map = state.get("metadata", {}).get("agent_model_map", {})
    if agent_name and agent_name in agent_model_map:
        entry = agent_model_map[agent_name]
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return entry[0], entry[1]

    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "claude-sonnet-4-5-20251001"
    model_provider = state.get("metadata", {}).get("model_provider") or "Anthropic"'''

if "agent_model_map" in text:
    print("SKIP: patch gia applicata.")
else:
    text = text.replace(old, new, 1)
    llm_file.write_text(text, encoding="utf-8")
    print("OK: routing agente aggiunto a get_agent_model_config.")
