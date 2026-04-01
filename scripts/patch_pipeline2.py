"""Aggiunge la funzione _load_agent_model_map a run_pipeline.py."""
from pathlib import Path

pipeline = Path("src/run_pipeline.py")
text = pipeline.read_text(encoding="utf-8")

func = '''
def _load_agent_model_map() -> dict:
    """Carica la mappa agente->modello da agent_router."""
    try:
        from src.llm.agent_router import get_model_for_agent, _load, _AGENT_MAP
        _load()
        return dict(_AGENT_MAP)
    except Exception as e:
        log.warning(f"agent_router non disponibile, uso fallback globale: {e}")
        return {}

'''

marker = "def run_pipeline("
if "_load_agent_model_map" in text:
    print("SKIP: funzione gia presente.")
else:
    text = text.replace(marker, func + marker, 1)
    pipeline.write_text(text, encoding="utf-8")
    print("OK: _load_agent_model_map aggiunta.")
