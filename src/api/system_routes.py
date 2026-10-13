"""System routes: health, sidebar state, sessions, LLM provider, Gemini key, settings, answer feedback.

``/health`` and ``/sidebar-state`` only read the cache the background health loop fills; the loop logs
state changes only. ``install_llm`` is the one place that (re)builds provider, condense provider and
generator together — provider switch, Gemini key and the fallback toggle all go through it.
"""
import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.api.access import player_source_ids
from src.api.schemas import (
    ConfigUpdateRequest,
    FeedbackRequest,
    GeminiKeyStatus,
    HealthResponse,
    ProviderInfo,
    SidebarState,
    SwitchProviderRequest,
)
from src.config.manager import Settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Answer ratings from the chat, one JSON object per line (gitignored like the logs)
_FEEDBACK_PATH = Path("logs/feedback.jsonl")


# Health state, refreshed by _health_loop; /health and /sidebar-state only read it
_health_cache: dict = {"chroma": False, "llm": False}


_HEALTH_INTERVAL_S = 60.0


# llm settings that change how the provider objects are built; the rest (temperature, timeout, ...) is
# read per call and needs no rebuild
_LLM_BUILD_KEYS = {"fallback_enabled"}


async def install_llm(settings: Settings) -> None:
    """Build provider, condense provider and generator from ``settings`` and swap all three in at once.

    The one place for every change that affects how the LLM objects are built — provider switch, Gemini
    key, fallback toggle. No ``await`` between the three assignments, so a request sees either the old or
    the new set. The LLM health is refreshed right after, so the sidebar doesn't show the old provider's
    state for up to a minute. Raises whatever building the active provider raises; nothing is swapped then.
    """
    import src.main as main_module
    from src.generation.provider_factory import ProviderFactory

    provider, condense, generator = ProviderFactory.build_all(settings)
    main_module.provider, main_module.condense_provider, main_module.generator = provider, condense, generator
    try:
        update_health(_health_cache["chroma"], await provider.health_check())
    except Exception as e:
        logger.warning("Health check after the LLM rebuild failed (%s: %s)", type(e).__name__, e)


@router.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Append a thumbs up/down (or its withdrawal) for one chat answer to logs/feedback.jsonl."""
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **req.model_dump()}
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Feedback {req.rating} for message {req.message_id} ({req.role})")
    return {"saved": True}


@router.get("/feedback")
async def list_feedback():
    """Current rating per answer, newest first — the last line per message_id wins, withdrawn ones are left out."""
    if not _FEEDBACK_PATH.exists():
        return []
    latest: dict[str, dict] = {}
    for line in _FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            latest[entry["message_id"]] = entry
        except (ValueError, KeyError, TypeError):
            continue  # a broken line (e.g. hand-edited) must not hide the others
    rated = [e for e in latest.values() if e.get("rating")]
    return sorted(rated, key=lambda e: e.get("ts", ""), reverse=True)


def update_health(chroma_ok: bool, llm_ok: bool) -> None:
    """Store a health result; only a change is logged (WARNING when something went down, INFO on
    recovery) — a line every 60 s would bury the moment the LLM actually went away."""
    previous = (_health_cache["chroma"], _health_cache["llm"])
    _health_cache.update({"chroma": chroma_ok, "llm": llm_ok})
    if previous != (chroma_ok, llm_ok):
        level = logging.INFO if chroma_ok and llm_ok else logging.WARNING
        logger.log(level, "Health changed: chroma=%s llm=%s (was chroma=%s llm=%s)", chroma_ok, llm_ok, *previous)


async def _health_loop(interval: float = _HEALTH_INTERVAL_S):
    """Background loop that refreshes the health cache periodically."""
    while True:
        try:
            import src.main as main_module
            chroma_ok = await run_in_threadpool(main_module.vectorstore.health_check)
            llm_ok = await main_module.provider.health_check()
            update_health(chroma_ok, llm_ok)
        except Exception as e:
            logger.warning("Background health check failed (%s: %s)", type(e).__name__, e)
        await asyncio.sleep(interval)


@router.get("/health", response_model=HealthResponse)
async def health():
    """Check the health status of the database and LLM provider."""
    chroma_ok = _health_cache["chroma"]
    llm_ok = _health_cache["llm"]

    import src.main as main_module
    has_sources = bool(main_module.config.settings.ingestion.sources)
    status = "healthy" if (chroma_ok and llm_ok) else "degraded"
    return HealthResponse(status=status, chromadb=chroma_ok, llm=llm_ok, sources_configured=has_sources,
                          reranker=main_module.retriever.reranker_device)


@router.get("/sidebar-state", response_model=SidebarState)
async def sidebar_state():
    """Combined endpoint for all sidebar data — one round-trip instead of four."""
    import src.main as main_module
    from src.generation.providers.gemini import get_api_key_status

    # Provider info
    prov = main_module.provider
    provider = ProviderInfo(provider=prov.provider, model=prov.model)

    # Gemini key status
    gem_raw = get_api_key_status(main_module.config.settings.llm.gemini)
    gemini_status = GeminiKeyStatus(**gem_raw)

    # Available categories (derived from sources config)
    cats: set[str] = set()
    for s in main_module.config.settings.ingestion.sources:
        if s.default_category:
            cats.add(s.default_category)
        for v in (s.category_map or {}).values():
            if isinstance(v, dict):
                cats.add(v["category"])
            else:
                cats.add(v)

    return SidebarState(
        health=await health(),
        provider=provider,
        gemini_status=gemini_status,
        available_categories=sorted(cats),
        player_source_ids=player_source_ids(main_module.config.settings.ingestion.sources),
        llm_timeout=getattr(main_module.config.settings.llm, prov.provider).timeout,
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve the chat history and details for a specific session."""
    import src.main as main_module
    cm = main_module.conversation_manager
    session = cm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    return {
        "session_id": session.session_id,
        "messages": [{"role": m.role, "content": m.content} for m in session.messages],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and its history from the manager."""
    import src.main as main_module
    cm = main_module.conversation_manager
    if not cm.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    return {"status": "deleted"}


@router.get("/provider", response_model=ProviderInfo)
async def get_provider():
    """Get information about the currently active LLM provider and model."""
    import src.main as main_module
    return ProviderInfo(provider=main_module.provider.provider, model=main_module.provider.model)


@router.get("/provider/gemini/status")
async def gemini_key_status():
    """Report whether a Gemini API key is available, without exposing it."""
    import src.main as main_module
    from src.generation.providers.gemini import get_api_key_status
    return get_api_key_status(main_module.config.settings.llm.gemini)


@router.post("/provider/gemini/key")
async def set_gemini_key(payload: dict):
    """Set a runtime Gemini API key (process-local, not persisted to disk).

    Body: {"api_key": "..."}  — pass an empty string or null to clear.
    The LLM objects are always rebuilt, so the key reaches Gemini as the active provider *and* as the
    fallback (before, only an active Gemini was rebuilt — the fallback kept running without a key).
    """
    import src.main as main_module
    from src.generation.providers.gemini import set_runtime_api_key

    key = (payload or {}).get("api_key")
    if key is not None and not isinstance(key, str):
        raise HTTPException(status_code=400, detail="api_key muss ein String sein")

    set_runtime_api_key(key.strip() if isinstance(key, str) else None)

    settings = main_module.config.settings
    try:
        await install_llm(settings)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Key vom Provider abgelehnt: {e}") from e

    # Never the key itself
    logger.info("Gemini runtime key %s (LLM rebuilt, active provider: %s)",
                "set" if key and key.strip() else "cleared", settings.llm.provider)
    return {"status": "ok", "rebuilt_active_provider": settings.llm.provider == "gemini"}


@router.get("/config")
async def get_config():
    """Return the current UI-editable settings plus a handful of read-only
    fields (embeddings model, vectorstore mode) the Settings page displays.
    Secrets and infrastructure knobs are never included."""
    import src.main as main_module
    return main_module.config.editable_snapshot()


@router.put("/config")
async def update_config(request: ConfigUpdateRequest):
    """Apply a partial settings update, persist to the settings.local.yaml sidecar, and mutate
    the live config in place so running services pick up the new values on
    the next call.

    Unknown keys are silently dropped. Invalid values raise a Pydantic
    ValidationError, which FastAPI surfaces as HTTP 422.
    """
    from pydantic import ValidationError

    import src.main as main_module

    updates = request.model_dump(exclude_none=True)
    try:
        snapshot = main_module.config.save_settings(updates)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        logger.error(f"save_settings failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Values are safe to log: the allow-list (_EDITABLE_KEYS) holds no secrets and no paths
    logger.info("Settings updated: %s", updates)
    if _LLM_BUILD_KEYS & set(updates.get("llm") or {}):
        try:
            await install_llm(main_module.config.settings)
        except Exception as e:
            logger.exception("LLM rebuild after the settings change failed")
            raise HTTPException(status_code=500, detail=f"Gespeichert, aber LLM-Neuaufbau fehlgeschlagen: {e}") from e
    return snapshot


@router.post("/provider", response_model=ProviderInfo)
async def switch_provider(request: SwitchProviderRequest):
    """Dynamically switch the active LLM provider at runtime (all LLM objects rebuilt via ``install_llm``)."""
    import src.main as main_module

    settings = main_module.config.settings

    previous_provider = settings.llm.provider
    try:
        settings.llm.provider = request.provider
        await install_llm(settings)
        logger.info(f"Provider switched to: {request.provider}")
    except Exception as e:
        settings.llm.provider = previous_provider
        logger.error(f"Provider switch to {request.provider} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Provider-Wechsel fehlgeschlagen: {e}") from e

    return ProviderInfo(
        provider=main_module.provider.provider,
        model=main_module.provider.model,
    )
