"""
Routing agenti per tier di costo.
Legge config/agent_tiers.yaml e restituisce (model_name, model_provider) per ogni agente.
"""

from pathlib import Path
import yaml

_TIERS: dict = {}
_AGENT_MAP: dict[str, tuple[str, str]] = {}


def _load():
    global _TIERS, _AGENT_MAP
    if _AGENT_MAP:
        return
    cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "agent_tiers.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _TIERS = cfg.get("tiers", {})
    for tier_name, tier_data in _TIERS.items():
        model_name = tier_data["model_name"]
        model_provider = tier_data["model_provider"]
        for agent_id in tier_data.get("agents", []):
            _AGENT_MAP[agent_id] = (model_name, model_provider)


def get_model_for_agent(agent_id: str) -> tuple[str, str]:
    """
    Restituisce (model_name, model_provider) per l'agente dato.
    Se l'agente non e' mappato, usa il fallback PREMIUM (Sonnet).
    """
    _load()
    fallback = ("claude-sonnet-4-5-20251001", "Anthropic")
    return _AGENT_MAP.get(agent_id, fallback)


def get_tier_summary() -> dict:
    """Restituisce un riepilogo dei tier per il logging."""
    _load()
    summary = {}
    for tier_name, tier_data in _TIERS.items():
        summary[tier_name] = {
            "model": tier_data["model_name"],
            "agents": tier_data.get("agents", []),
        }
    return summary
