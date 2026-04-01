"""Helper functions for LLM"""

import json
import logging
import time
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 6,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """

    # ------------------------------------------------------------------
    # FASE 2 — Cache hook
    # Controlla la cache PRIMA di qualsiasi chiamata LLM.
    # Se il segnale è ancora valido (< 24h, prezzo stabile, nessun 8-K)
    # ricostruisce il Pydantic dalla cache e ritorna subito.
    # ------------------------------------------------------------------
    if agent_name and state:
        cached_result = _try_return_from_cache(agent_name, state, pydantic_model)
        if cached_result is not None:
            return cached_result
    # ------------------------------------------------------------------

    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        # Use system defaults when no state or agent_name is provided
        model_name = "gpt-4.1"
        model_provider = "OPENAI"

    # Extract API keys from state if available
    api_keys = None
    if state:
        request = state.get("metadata", {}).get("request")
        if request and hasattr(request, 'api_keys'):
            api_keys = request.api_keys

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider, api_keys)

    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                # ------------------------------------------------------------------
                # FASE 2 — Salva il segnale in cache dopo una chiamata LLM riuscita
                # ------------------------------------------------------------------
                if agent_name and state:
                    _save_result_to_cache(agent_name, state, result)
                # ------------------------------------------------------------------
                return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            wait = 60 if "429" in str(e) else 2 ** attempt
            time.sleep(wait)
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


# ----------------------------------------------------------------------
# FASE 2 — Funzioni helper per cache
# ----------------------------------------------------------------------

def _try_return_from_cache(
    agent_name: str,
    state: dict,
    pydantic_model: type[BaseModel],
) -> BaseModel | None:
    """
    Controlla la cache per ogni ticker nello stato.
    Se TUTTI i ticker hanno un segnale cached valido, non serve fare nulla.
    Se anche solo UN ticker non ha cache, ritorna None → il LLM viene chiamato.

    Nota: gli agenti in Athanor Alpha processano un ticker alla volta,
    quindi in pratica questo controllo riguarda sempre un singolo ticker.
    """
    try:
        from src.data.cache_guard import should_use_cache, get_cached_signal_safe

        # Cerca il ticker corrente nello stato
        ticker = _get_current_ticker(state)
        if not ticker:
            return None

        if not should_use_cache(agent_name, ticker, state):
            return None

        # Cache valida: ricostruisci il Pydantic
        cached = get_cached_signal_safe(agent_name, ticker)
        result = _build_pydantic_from_cache(cached, pydantic_model)

        if result is not None:
            logger.info(
                f"[llm] CACHE HIT — {agent_name}/{ticker} "
                f"→ {cached.get('signal', '?')} ({cached.get('confidence', 0):.0f}%) "
                f"[LLM saltato]"
            )
        return result

    except Exception as e:
        # Mai bloccare la pipeline per un errore di cache
        logger.warning(f"[llm] Errore hook cache (ignorato): {e}")
        return None


def _save_result_to_cache(agent_name: str, state: dict, result: BaseModel) -> None:
    """
    Salva il risultato di una chiamata LLM riuscita nella cache.
    Estrae signal, confidence e reasoning dal Pydantic model.
    """
    try:
        from src.data.signal_cache import save_signal

        ticker = _get_current_ticker(state)
        if not ticker:
            return

        # Estrae i campi dal Pydantic (con fallback sicuri)
        signal     = _extract_field(result, ["signal", "action", "recommendation"], "neutral")
        confidence = float(_extract_field(result, ["confidence", "confidence_score", "score"], 0.0))
        reasoning  = str(_extract_field(result, ["reasoning", "rationale", "explanation"], ""))

        save_signal(agent_name, ticker, signal, confidence, reasoning)

    except Exception as e:
        logger.warning(f"[llm] Errore salvataggio cache (ignorato): {e}")


def _get_current_ticker(state: dict) -> str | None:
    """
    Estrae il ticker corrente dallo stato LangGraph.
    Gli agenti di Athanor Alpha lavorano su un ticker alla volta
    passato in state["data"]["current_ticker"] oppure come primo
    elemento di state["data"]["tickers"].
    """
    try:
        data = state.get("data", {})
        # Prima opzione: campo dedicato
        ticker = data.get("current_ticker")
        if ticker:
            return str(ticker)
        # Seconda opzione: primo ticker della lista
        tickers = data.get("tickers", [])
        if tickers:
            return str(tickers[0])
        return None
    except Exception:
        return None


def _build_pydantic_from_cache(cached: dict, pydantic_model: type[BaseModel]) -> BaseModel | None:
    """
    Ricostruisce un'istanza Pydantic a partire dal dict cached.
    Prova prima a deserializzare direttamente, poi fa un mapping
    campo per campo con valori di default sicuri.
    """
    try:
        # Prova deserializzazione diretta se i campi coincidono
        fields = pydantic_model.model_fields
        kwargs = {}
        for field_name, field in fields.items():
            if field_name in cached:
                kwargs[field_name] = cached[field_name]
            elif field.annotation == str:
                kwargs[field_name] = cached.get("reasoning", "Cached signal")
            elif field.annotation == float:
                kwargs[field_name] = cached.get("confidence", 0.0)
            elif field.annotation == int:
                kwargs[field_name] = int(cached.get("confidence", 0))
            elif hasattr(field.annotation, "__args__"):
                # Literal type: usa il valore cached o il primo allowed
                signal_val = cached.get("signal", field.annotation.__args__[0])
                # Verifica che il valore sia tra quelli ammessi
                if signal_val in field.annotation.__args__:
                    kwargs[field_name] = signal_val
                else:
                    kwargs[field_name] = field.annotation.__args__[0]
            else:
                kwargs[field_name] = None
        return pydantic_model(**kwargs)
    except Exception as e:
        logger.warning(f"[llm] Impossibile ricostruire Pydantic dalla cache: {e}")
        return None


def _extract_field(obj: any, field_names: list[str], default: any) -> any:
    """Estrae il primo campo disponibile da un oggetto Pydantic o dict."""
    for name in field_names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return default


# ----------------------------------------------------------------------
# Funzioni originali (invariate)
# ----------------------------------------------------------------------

def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")

    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)

    # Cerca nella mappa di routing per agente
    agent_model_map = state.get("metadata", {}).get("agent_model_map", {})
    if agent_name and agent_name in agent_model_map:
        entry = agent_model_map[agent_name]
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return entry[0], entry[1]

    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "claude-sonnet-4-5-20251001"
    model_provider = state.get("metadata", {}).get("model_provider") or "Anthropic"

    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value

    return model_name, model_provider
