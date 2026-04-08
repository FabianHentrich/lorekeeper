import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    HealthResponse,
    IngestJobResponse,
    IngestStatusResponse,
    ProviderInfo,
    QueryRequest,
    QueryResponse,
    SourceReference,
    StatsResponse,
    SwitchProviderRequest,
)

from src.generation.generator import Generator

logger = logging.getLogger(__name__)

router = APIRouter()

# Ingestion job storage (in-memory)
_ingest_jobs: dict[str, IngestStatusResponse] = {}

# Health check cache (TTL = 30s)
_health_cache: dict = {"ts": 0.0, "chroma": False, "llm": False}
_HEALTH_TTL = 30.0


def _get_services():
    """Lazy import to avoid circular imports at module level."""
    from src.main import (
        conversation_manager,
        generator,
        condense_provider,
        prompt_manager,
        retriever,
        vectorstore,
        provider,
    )
    return conversation_manager, generator, condense_provider, prompt_manager, retriever, vectorstore, provider


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.time()
    cm, gen, condense_prov, pm, ret, vs, prov = _get_services()

    session = cm.get_or_create_session(request.session_id)

    question = request.question

    # Condense question if there's conversation history
    history = cm.get_history_for_condense(session.session_id)
    if history and cm.config.condense_question:
        condense_prompt = pm.render_condense(history=history, question=question)
        question = await gen.condense_question(condense_prompt, condense_prov)

    # Retrieve relevant chunks
    chunks = await ret.retrieve(
        query=question,
        top_k=request.top_k,
        metadata_filters=request.metadata_filters,
        top_k_rerank=request.top_k_rerank,
        max_per_source=request.max_per_source,
    )

    # Generate answer
    if chunks:
        chunk_dicts = [
            {"source_file": c.source_file, "heading": c.heading, "content": c.content}
            for c in chunks
        ]
        system_prompt = pm.get_system_prompt()
        qa_prompt = pm.render_qa(chunks=chunk_dicts, question=request.question)
        try:
            response = await gen.generate(system_prompt=system_prompt, qa_prompt=qa_prompt)
            answer = response.content
            model_used = response.model
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise HTTPException(status_code=503, detail=f"LLM nicht erreichbar: {e}")
    else:
        answer = pm.render_no_context(question=request.question)
        model_used = "none"

    # Update session
    session.add_message("user", request.question)
    session.add_message("assistant", answer)

    sources = [
        SourceReference(
            file=c.source_file,
            source_path=c.metadata.get("source_path", ""),
            document_type=c.document_type,
            heading=c.heading,
            chunk_preview=c.content[:100],
            score=c.score,
        )
        for c in chunks
    ]

    latency_ms = (time.time() - start) * 1000

    return QueryResponse(
        answer=answer,
        sources=sources,
        session_id=session.session_id,
        retrieval_scores=[c.score for c in chunks],
        model_used=model_used,
        latency_ms=latency_ms,
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    cm, gen, condense_prov, pm, ret, vs, prov = _get_services()

    session = cm.get_or_create_session(request.session_id)
    question = request.question

    history = cm.get_history_for_condense(session.session_id)
    if history and cm.config.condense_question:
        condense_prompt = pm.render_condense(history=history, question=question)
        question = await gen.condense_question(condense_prompt, condense_prov)

    chunks = await ret.retrieve(
        query=question,
        top_k=request.top_k,
        metadata_filters=request.metadata_filters,
        top_k_rerank=request.top_k_rerank,
        max_per_source=request.max_per_source,
    )

    sources = [
        {
            "file": c.source_file,
            "source_path": c.metadata.get("source_path", ""),
            "document_type": c.document_type,
            "heading": c.heading,
            "chunk_preview": c.content[:100],
            "score": c.score,
        }
        for c in chunks
    ]

    async def event_stream():
        full_answer = ""

        if chunks:
            chunk_dicts = [
                {"source_file": c.source_file, "heading": c.heading, "content": c.content}
                for c in chunks
            ]
            system_prompt = pm.get_system_prompt()
            qa_prompt = pm.render_qa(chunks=chunk_dicts, question=request.question)

            async for token in gen.generate_stream(system_prompt=system_prompt, qa_prompt=qa_prompt):
                full_answer += token
                event = json.dumps({"type": "token", "content": token})
                yield f"data: {event}\n\n"
        else:
            full_answer = pm.render_no_context(question=request.question)
            event = json.dumps({"type": "token", "content": full_answer})
            yield f"data: {event}\n\n"

        session.add_message("user", request.question)
        session.add_message("assistant", full_answer)

        usage = dict(getattr(gen, "last_usage", {}) or {})
        if usage:
            session.add_usage(usage)

        done_event = json.dumps({
            "type": "done",
            "session_id": session.session_id,
            "sources": sources,
            "model_used": prov.model if hasattr(prov, "model") else "unknown",
            "usage": usage,
            "session_usage": session.usage_totals,
        })
        yield f"data: {done_event}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestJobResponse)
async def ingest():
    from src.ingestion.orchestrator import IngestionOrchestrator
    _, _, _, _, _, vs, _ = _get_services()

    job_id = str(uuid.uuid4())
    _ingest_jobs[job_id] = IngestStatusResponse(job_id=job_id, status="queued")

    async def run_ingestion():
        _ingest_jobs[job_id].status = "running"
        try:
            orchestrator = IngestionOrchestrator()
            result = orchestrator.run(vectorstore=vs)
            _ingest_jobs[job_id].status = "done"
            _ingest_jobs[job_id].documents_processed = result.documents_processed
            _ingest_jobs[job_id].chunks_created = result.chunks_created
            _ingest_jobs[job_id].chunks_updated = result.chunks_updated
            _ingest_jobs[job_id].chunks_deleted = result.chunks_deleted
            _ingest_jobs[job_id].errors = result.errors
            _ingest_jobs[job_id].duration_seconds = result.duration_seconds
        except Exception as e:
            _ingest_jobs[job_id].status = "error"
            _ingest_jobs[job_id].errors.append(str(e))

    asyncio.create_task(run_ingestion())
    return IngestJobResponse(job_id=job_id, status="queued")


@router.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
async def ingest_status(job_id: str):
    if job_id not in _ingest_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _ingest_jobs[job_id]


@router.get("/health", response_model=HealthResponse)
async def health():
    now = time.monotonic()
    if now - _health_cache["ts"] < _HEALTH_TTL:
        chroma_ok, llm_ok = _health_cache["chroma"], _health_cache["llm"]
    else:
        _, _, _, _, _, vs, prov = _get_services()
        chroma_ok = vs.health_check()
        llm_ok = await prov.health_check()
        _health_cache.update({"ts": now, "chroma": chroma_ok, "llm": llm_ok})

    status = "healthy" if (chroma_ok and llm_ok) else "degraded"
    return HealthResponse(status=status, chromadb=chroma_ok, llm=llm_ok)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
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
    cm, *_ = _get_services()
    if not cm.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.get("/stats", response_model=StatsResponse)
async def stats():
    _, _, _, _, _, vs, _ = _get_services()
    return StatsResponse(chunk_count=vs.count())


@router.get("/provider", response_model=ProviderInfo)
async def get_provider():
    _, _, _, _, _, _, prov = _get_services()
    return ProviderInfo(
        provider=getattr(prov, "provider", "unknown") if hasattr(prov, "provider") else type(prov).__name__,
        model=getattr(prov, "model", "unknown"),
    )


@router.post("/provider", response_model=ProviderInfo)
async def switch_provider(request: SwitchProviderRequest):
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
        raise HTTPException(status_code=500, detail=f"Provider-Wechsel fehlgeschlagen: {e}")

    return ProviderInfo(
        provider=request.provider,
        model=getattr(main_module.provider, "model", "unknown"),
    )
