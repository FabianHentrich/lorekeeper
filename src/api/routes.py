import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ConfigUpdateRequest,
    GeminiKeyStatus,
    HealthResponse,
    IngestJobResponse,
    IngestStatusResponse,
    ProviderInfo,
    QueryRequest,
    QueryResponse,
    SidebarState,
    SourceReference,
    StatsResponse,
    SwitchProviderRequest,
)
from src.conversation.manager import needs_condense
from src.generation.generator import Generator
from src.generation.providers.base import StreamResult

logger = logging.getLogger(__name__)

router = APIRouter()

# Ingestion job storage (in-memory)
_ingest_jobs: dict[str, IngestStatusResponse] = {}

# Keep strong refs to background tasks so the GC can't cancel them mid-run.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro) -> asyncio.Task:
    """Create a background task and retain a strong reference until it completes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

# Health check cache (TTL = 30s)
_health_cache: dict = {"ts": 0.0, "chroma": False, "llm": False}


def _get_services():
    """Lazy import to avoid circular imports at module level."""
    from src.main import (
        condense_provider,
        conversation_manager,
        generator,
        prompt_manager,
        provider,
        retriever,
        vectorstore,
    )
    return conversation_manager, generator, condense_provider, prompt_manager, retriever, vectorstore, provider


def _ms(since: float) -> int:
    """Milliseconds elapsed since a ``time.perf_counter()`` reading."""
    return round((time.perf_counter() - since) * 1000)


def _log_timings(timings: dict) -> None:
    """One INFO line per answered query — the basis for spotting slow stages."""
    stages = " ".join(f"{k}={v}" for k, v in timings.get("retrieval_stages_ms", {}).items())
    logger.info(
        f"Query timings: condense={timings.get('condense_ms', 0)}ms "
        f"retrieval={timings.get('retrieval_ms', 0)}ms ({stages}) "
        f"first_token={timings.get('first_token_ms', 0)}ms total={timings.get('total_ms', 0)}ms "
        f"chunks={timings.get('chunks', 0)} prompt_chars={timings.get('prompt_chars', 0)}"
    )


async def _prepare_query(request: QueryRequest, timings: dict):
    """Shared condense+retrieve step. Returns (session, chunks, system_prompt,
    qa_prompt, search_query) — qa_prompt is None when no chunks passed the
    threshold; search_query is the (possibly condensed) question used for retrieval.
    Condense/retrieval durations, chunk count and prompt size are written into ``timings``."""
    cm, gen, condense_prov, pm, ret, _vs, _prov = _get_services()

    session = cm.get_or_create_session(request.session_id)
    question = request.question

    history = cm.get_history_for_condense(session.session_id)
    if history and cm.config.condense_question:
        if needs_condense(question):
            condense_prompt = pm.render_condense(history=history, question=question)
            t0 = time.perf_counter()
            try:
                question = await gen.condense_question(condense_prompt, condense_prov)
                logger.info(f"Condensed question: {request.question!r} → {question!r}")
            except Exception as e:
                logger.warning(f"Condense failed ({e}); retrieving with the original question")
            timings["condense_ms"] = _ms(t0)
        else:
            logger.info(f"Condense skipped (standalone question): {question!r}")

    stages: dict[str, float] = {}
    t0 = time.perf_counter()
    chunks = await ret.retrieve(
        query=question,
        top_k=request.top_k,
        metadata_filters=request.metadata_filters,
        top_k_rerank=request.top_k_rerank,
        max_per_source=request.max_per_source,
        hybrid=request.hybrid_search,
        timings=stages,
    )
    timings["retrieval_ms"] = _ms(t0)
    timings["retrieval_stages_ms"] = {k: round(v) for k, v in stages.items()}
    timings["chunks"] = len(chunks)

    if chunks:
        chunk_dicts = [
            {"source_file": c.source_file, "heading": c.heading, "content": c.content}
            for c in chunks
        ]
        system_prompt = pm.get_system_prompt()
        # Condensed question: resolves follow-up references ("sein Vater") — the QA prompt has no chat history
        qa_prompt = pm.render_qa(chunks=chunk_dicts, question=question)
    else:
        chunk_dicts = []
        system_prompt = ""
        qa_prompt = None

    timings["prompt_chars"] = len(system_prompt) + len(qa_prompt or "")
    return session, chunks, system_prompt, qa_prompt, question


def _chunk_to_source_dict(c) -> dict:
    """Format a retrieved chunk into a source dictionary for the API response."""
    return {
        "file": c.source_file,
        "source_path": c.metadata.get("source_path", ""),
        "document_type": c.document_type,
        "heading": c.heading,
        "chunk_preview": c.content[:100],
        "score": c.score,
    }


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute a query against the knowledge base and return the generated answer and sources."""
    start = time.time()
    t_start = time.perf_counter()
    _, gen, _, pm, _, _, _ = _get_services()
    timings: dict = {}
    session, chunks, system_prompt, qa_prompt, search_query = await _prepare_query(request, timings)

    if qa_prompt is not None:
        try:
            response = await gen.generate(system_prompt=system_prompt, qa_prompt=qa_prompt)
            answer = response.content
            model_used = response.model
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise HTTPException(status_code=503, detail=f"LLM nicht erreichbar: {e}") from e
    else:
        answer = pm.render_no_context(question=request.question)
        model_used = "none"

    session.add_message("user", request.question)
    session.add_message("assistant", answer)

    sources = [SourceReference(**_chunk_to_source_dict(c)) for c in chunks]
    latency_ms = (time.time() - start) * 1000
    timings["total_ms"] = _ms(t_start)
    _log_timings(timings)

    return QueryResponse(
        answer=answer,
        sources=sources,
        session_id=session.session_id,
        retrieval_scores=[c.score for c in chunks],
        model_used=model_used,
        latency_ms=latency_ms,
        search_query=search_query,
        timings=timings,
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Execute a query and stream the generated answer tokens over server-sent events (SSE)."""
    t_start = time.perf_counter()
    _, gen, _, pm, _, _, prov = _get_services()
    timings: dict = {}
    session, chunks, system_prompt, qa_prompt, search_query = await _prepare_query(request, timings)
    sources = [_chunk_to_source_dict(c) for c in chunks]

    async def event_stream():
        """Yield SSE 'token' events during generation and a final 'done' event with metadata."""
        full_answer = ""
        stream_ctx = StreamResult()

        try:
            if qa_prompt is not None:
                async for token in gen.generate_stream(
                    system_prompt=system_prompt,
                    qa_prompt=qa_prompt,
                    stream_result=stream_ctx,
                ):
                    if not full_answer:
                        timings["first_token_ms"] = _ms(t_start)
                    full_answer += token
                    event = json.dumps({"type": "token", "content": token})
                    yield f"data: {event}\n\n"
            else:
                full_answer = pm.render_no_context(question=request.question)
                timings["first_token_ms"] = _ms(t_start)
                event = json.dumps({"type": "token", "content": full_answer})
                yield f"data: {event}\n\n"
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        session.add_message("user", request.question)
        session.add_message("assistant", full_answer)

        usage = dict(stream_ctx.usage)
        if usage:
            session.add_usage(usage)
        timings["total_ms"] = _ms(t_start)
        _log_timings(timings)

        done_event = json.dumps({
            "type": "done",
            "session_id": session.session_id,
            "sources": sources,
            "model_used": stream_ctx.model or prov.model,
            "search_query": search_query,
            "timings": timings,
            "usage": usage,
            "session_usage": session.usage_totals,
        })
        yield f"data: {done_event}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _start_ingest_job(run_sync) -> str:
    """Create a job, launch the sync orchestrator run in a thread, and return
    the job_id. `run_sync` is a zero-arg callable returning an IngestionResult."""
    import src.main as main_module

    # Parallel runs race on delete/upsert of the same chunks
    if any(j.status in ("queued", "running") for j in _ingest_jobs.values()):
        raise HTTPException(status_code=409, detail="Eine Indexierung läuft bereits")

    job_id = str(uuid.uuid4())
    _ingest_jobs[job_id] = IngestStatusResponse(job_id=job_id, status="queued")

    def _update_progress(result):
        """Copy orchestrator progress fields onto the tracked job status."""
        job = _ingest_jobs[job_id]
        job.phase = result.phase
        job.documents_processed = result.documents_processed
        job.documents_total = result.documents_total
        job.documents_skipped = result.documents_skipped
        job.chunks_created = result.chunks_created
        job.chunks_updated = result.chunks_updated
        job.chunks_deleted = result.chunks_deleted
        job.duration_seconds = result.duration_seconds

    async def _runner():
        """Run the sync orchestrator off-thread and update job state on success/error."""
        _ingest_jobs[job_id].status = "running"
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, run_sync, _update_progress
            )
            _update_progress(result)
            _ingest_jobs[job_id].status = "done"
            _ingest_jobs[job_id].errors = result.errors
            main_module.bm25_index.invalidate()
        except Exception as e:
            _ingest_jobs[job_id].status = "error"
            _ingest_jobs[job_id].errors.append(str(e))

    _spawn_background(_runner())
    return job_id


@router.post("/ingest", response_model=IngestJobResponse)
async def ingest(full: bool = False):
    """Start an asynchronous ingestion job to process and embed all configured sources.

    Incremental by default (unchanged files are skipped); ``?full=true`` re-chunks and
    re-embeds every file, e.g. after chunking settings changed.
    """
    import src.main as main_module
    from src.ingestion.orchestrator import IngestionOrchestrator
    _, _, _, _, _, vs, _ = _get_services()

    def _run(progress_cb):
        """Run ingestion across all configured sources."""
        return IngestionOrchestrator(config=main_module.config).run(
            vectorstore=vs, progress_callback=progress_cb, full=full,
        )

    return IngestJobResponse(job_id=_start_ingest_job(_run), status="queued")


@router.get("/ingest/active", response_model=IngestStatusResponse | None)
async def ingest_active():
    """The queued or running ingest job, or null — lets a client attach to it after a 409."""
    return next((j for j in _ingest_jobs.values() if j.status in ("queued", "running")), None)


@router.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
async def ingest_status(job_id: str):
    """Get the current progress and status of an ingestion job."""
    if job_id not in _ingest_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _ingest_jobs[job_id]


async def _health_loop(interval: float = 60.0):
    """Background loop that refreshes the health cache periodically."""
    while True:
        try:
            _, _, _, _, _, vs, prov = _get_services()
            chroma_ok = await run_in_threadpool(vs.health_check)
            llm_ok = await prov.health_check()
            _health_cache.update({"ts": time.monotonic(), "chroma": chroma_ok, "llm": llm_ok})
        except Exception as e:
            logger.warning("Background health check failed: %s", e)
        await asyncio.sleep(interval)


@router.get("/health", response_model=HealthResponse)
async def health():
    """Check the health status of the database and LLM provider."""
    chroma_ok = _health_cache["chroma"]
    llm_ok = _health_cache["llm"]

    import src.main as main_module
    has_sources = bool(main_module.config.settings.ingestion.sources)
    status = "healthy" if (chroma_ok and llm_ok) else "degraded"
    return HealthResponse(status=status, chromadb=chroma_ok, llm=llm_ok, sources_configured=has_sources)


@router.get("/sidebar-state", response_model=SidebarState)
async def sidebar_state():
    """Combined endpoint for all sidebar data — one round-trip instead of four."""
    import src.main as main_module
    from src.generation.providers.gemini import get_api_key_status

    # Health (from background cache)
    chroma_ok = _health_cache["chroma"]
    llm_ok = _health_cache["llm"]
    has_sources = bool(main_module.config.settings.ingestion.sources)
    h_status = "healthy" if (chroma_ok and llm_ok) else "degraded"
    health = HealthResponse(status=h_status, chromadb=chroma_ok, llm=llm_ok, sources_configured=has_sources)

    # Provider info
    _, _, _, _, _, _, prov = _get_services()
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
        health=health,
        provider=provider,
        gemini_status=gemini_status,
        available_categories=sorted(cats),
        player_source_ids=[s.id for s in main_module.config.settings.ingestion.sources if s.player_visible],
        llm_timeout=getattr(main_module.config.settings.llm, prov.provider).timeout,
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Retrieve the chat history and details for a specific session."""
    cm, *_ = _get_services()
    session = cm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "messages": [{"role": m.role, "content": m.content} for m in session.messages],
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and its history from the manager."""
    cm, *_ = _get_services()
    if not cm.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """Get statistics about the vector store, like total chunk count."""
    _, _, _, _, _, vs, _ = _get_services()
    return StatsResponse(chunk_count=vs.count())


@router.get("/provider", response_model=ProviderInfo)
async def get_provider():
    """Get information about the currently active LLM provider and model."""
    _, _, _, _, _, _, prov = _get_services()
    return ProviderInfo(provider=prov.provider, model=prov.model)


@router.get("/sources")
async def list_sources():
    """List all currently configured sources."""
    import src.main as main_module
    return {"sources": [s.model_dump() for s in main_module.config.settings.ingestion.sources]}


def _scan_path(path: Path, supported_formats: set[str]) -> list[dict]:
    """Scan top-level entries under a path. Shared by source and freeform scan."""
    entries = []
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append({"name": child.name, "type": "folder"})
        elif child.suffix.lower() in supported_formats:
            entries.append({"name": child.name, "type": "file"})
    logger.debug(f"Scanned {path}: {len(entries)} entries")
    return entries


@router.get("/sources/{source_id:path}/folders")
async def list_source_folders(source_id: str):
    """List top-level entries (folders and files) under a source's path."""
    import src.main as main_module
    source = next(
        (s for s in main_module.config.settings.ingestion.sources if s.id == source_id),
        None,
    )
    if source is None:
        logger.warning(f"list_source_folders: unknown source_id={source_id}")
        raise HTTPException(status_code=404, detail=f"Unknown source_id: {source_id}")

    src_path = Path(source.path).resolve()
    if not src_path.exists():
        logger.warning(f"list_source_folders: path does not exist: {src_path} (source={source_id})")
        return {"folders": [], "error": f"Path does not exist: {src_path}"}
    if src_path.is_file():
        return {"folders": [], "note": "File source — no subfolders"}

    supported = set(main_module.config.settings.ingestion.supported_formats)
    return {"folders": _scan_path(src_path, supported)}


@router.post("/sources/scan")
async def scan_path(payload: dict):
    """Scan a freeform path (not yet a configured source). Body: {"path": "..."}."""
    import src.main as main_module
    raw_path = (payload or {}).get("path", "")
    if not raw_path:
        raise HTTPException(status_code=400, detail="Missing 'path' field")

    p = Path(raw_path).resolve()
    logger.info(f"Scanning path: {p}")
    if not p.exists():
        logger.warning(f"Scan: path does not exist: {p}")
        return {"folders": [], "is_file": False, "error": f"Path does not exist: {p}"}
    if p.is_file():
        logger.info(f"Scan: path is a single file: {p}")
        return {"folders": [], "is_file": True}

    supported = set(main_module.config.settings.ingestion.supported_formats)
    return {"folders": _scan_path(p, supported), "is_file": False}


@router.put("/sources")
async def update_sources(payload: dict):
    """Replace the entire sources list. Body: {"sources": [SourceConfig, ...]}."""
    import src.main as main_module
    from src.config.manager import SourceConfig
    raw_sources = payload.get("sources")
    if raw_sources is None:
        raise HTTPException(status_code=400, detail="Missing 'sources' field")
    try:
        new_sources = [SourceConfig(**s) for s in raw_sources]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid source config: {e}") from e
    ids = [s.id for s in new_sources]
    # Only new ids are checked — an existing (legacy) id must never block saving the list.
    # '.'/'..' get normalized away by HTTP clients, so /sources/{id}/... could not reach them.
    existing = {s.id for s in main_module.config.settings.ingestion.sources}
    bad = [i for i in ids if i not in existing
           and (not i.strip() or i != i.strip() or i in (".", ".."))]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid source id(s): {bad} — must not be "
                            "empty, '.', '..' or have leading/trailing whitespace")
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=400, detail="Duplicate source ids")
    main_module.config.settings.ingestion.sources = new_sources
    main_module.config.save_sources()
    return {"sources": [s.model_dump() for s in new_sources]}


@router.post("/sources/{source_id:path}/reindex", response_model=IngestJobResponse)
async def reindex_source(source_id: str):
    """Delete + re-ingest a single source."""
    import src.main as main_module
    from src.ingestion.orchestrator import IngestionOrchestrator
    _, _, _, _, _, vs, _ = _get_services()

    if not any(s.id == source_id for s in main_module.config.settings.ingestion.sources):
        raise HTTPException(status_code=404, detail=f"Unknown source_id: {source_id}")

    def _run(progress_cb):
        """Drop existing chunks for this source, then re-ingest just its files."""
        vs.delete_by_source_id(source_id)
        return IngestionOrchestrator(config=main_module.config).run(
            vectorstore=vs, only_source_id=source_id, progress_callback=progress_cb,
        )

    return IngestJobResponse(job_id=_start_ingest_job(_run), status="queued")


@router.delete("/sources/{source_id:path}")
async def delete_source(source_id: str):
    """Remove a source from config and delete all its chunks."""
    import src.main as main_module
    _, _, _, _, _, vs, _ = _get_services()

    sources = main_module.config.settings.ingestion.sources
    if not any(s.id == source_id for s in sources):
        raise HTTPException(status_code=404, detail=f"Unknown source_id: {source_id}")

    deleted = vs.delete_by_source_id(source_id)
    main_module.config.settings.ingestion.sources = [s for s in sources if s.id != source_id]
    main_module.config.save_sources()
    return {"deleted_chunks": deleted}


@router.post("/sources/recategorize")
async def recategorize_endpoint():
    """Trigger a recategorization task to update metadata for existing configurations."""
    from src.ingestion.recategorize import recategorize
    _, _, _, _, _, vs, _ = _get_services()
    return await run_in_threadpool(recategorize, vectorstore=vs)


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
    If the active provider is currently Gemini, it is rebuilt so the new key
    takes effect immediately.
    """
    import src.main as main_module
    from src.generation.provider_factory import ProviderFactory
    from src.generation.providers.gemini import set_runtime_api_key

    key = (payload or {}).get("api_key")
    if key is not None and not isinstance(key, str):
        raise HTTPException(status_code=400, detail="api_key must be a string")

    set_runtime_api_key(key.strip() if isinstance(key, str) else None)

    # Hot-rebuild the active provider if it's Gemini, so the new key applies.
    settings = main_module.config.settings
    rebuilt = False
    if settings.llm.provider == "gemini":
        try:
            main_module.provider = ProviderFactory.create(settings.llm)
            main_module.generator = Generator(
                provider=main_module.provider,
                fallback_provider=ProviderFactory.create_fallback(settings.llm),
            )
            rebuilt = True
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Key rejected by provider: {e}") from e

    return {"status": "ok", "rebuilt_active_provider": rebuilt}


@router.get("/config")
async def get_config():
    """Return the current UI-editable settings plus a handful of read-only
    fields (embeddings model, vectorstore mode) the Settings page displays.
    Secrets and infrastructure knobs are never included."""
    import src.main as main_module
    return main_module.config.editable_snapshot()


@router.put("/config")
async def update_config(request: ConfigUpdateRequest):
    """Apply a partial settings update, persist to settings.yaml, and mutate
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

    logger.info(f"Settings updated: {list(updates.keys())}")
    return snapshot


@router.post("/admin/wipe")
async def wipe_collection(payload: dict):
    """Drop the entire ChromaDB collection. Requires confirm='DELETE' in body."""
    if payload.get("confirm") != "DELETE":
        raise HTTPException(status_code=400, detail="Set confirm='DELETE' to proceed")
    _, _, _, _, _, vs, _ = _get_services()
    vs.wipe_collection()
    return {"status": "wiped"}


@router.post("/provider", response_model=ProviderInfo)
async def switch_provider(request: SwitchProviderRequest):
    """Dynamically switch the active LLM provider at runtime."""
    import src.main as main_module
    from src.generation.provider_factory import ProviderFactory

    settings = main_module.config.settings

    if request.provider not in ("ollama", "gemini"):
        raise HTTPException(status_code=400, detail=f"Unbekannter Provider: {request.provider}")

    previous_provider = settings.llm.provider
    try:
        settings.llm.provider = request.provider
        new_provider = ProviderFactory.create(settings.llm)
        new_condense = ProviderFactory.create_condense_provider(
            settings.llm, settings.conversation,
        )
        new_generator = Generator(
            provider=new_provider,
            fallback_provider=ProviderFactory.create_fallback(settings.llm),
        )
        # All created successfully — commit the switch
        main_module.provider = new_provider
        main_module.condense_provider = new_condense
        main_module.generator = new_generator
        logger.info(f"Provider switched to: {request.provider}")
    except Exception as e:
        settings.llm.provider = previous_provider
        logger.error(f"Provider switch to {request.provider} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Provider-Wechsel fehlgeschlagen: {e}") from e

    return ProviderInfo(
        provider=main_module.provider.provider,
        model=main_module.provider.model,
    )
