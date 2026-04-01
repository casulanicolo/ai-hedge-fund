from pathlib import Path

pipeline = Path("src/run_pipeline.py")
text = pipeline.read_text(encoding="utf-8")

func = '''

def _load_agent_model_map() -> dict:
    """Carica la mappa agente->modello da agent_router."""
    try:
        from src.llm.agent_router import _load, _AGENT_MAP
        _load()
        return dict(_AGENT_MAP)
    except Exception as e:
        log.warning(f"agent_router non disponibile, uso fallback globale: {e}")
        return {}

'''

marker = "def _build_initial_state("
if "_load_agent_model_map" in text:
    print("SKIP: funzione gia presente.")
elif marker not in text:
    print("ERRORE: marker non trovato nel file.")
else:
    text = text.replace(marker, func + marker, 1)
    pipeline.write_text(text, encoding="utf-8")
    print("OK: _load_agent_model_map inserita.")
