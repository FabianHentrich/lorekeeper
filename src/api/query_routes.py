"""Chat queries: ``/query`` (synchronous) and ``/query/stream`` (SSE).

Condense → retrieval → prompt → LLM. Every query builds its filter with ``access.effective_filters`` —
player mode is enforced here, not in the UI. Services are module globals of ``src.main`` (set in the
lifespan), looked up per request via ``import src.main as main_module``. The generation counter drives
the "LLM is answering another question" notice (Ollama answers one question at a time, roadmap 8.0).
One INFO line carries the question, one the timings (``session``/``role``/``model`` tie them together).
"""
import asyncio
import json
import logging
import time
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.access import InvalidFilter, NoPlayerSources, effective_filters
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceReference,
)
from src.conversation.manager import needs_condense
from src.generation.providers.base import StreamResult
from src.ingestion.orchestrator import IMAGE_PATH_SEPARATOR

logger = logging.getLogger(__name__)

router = APIRouter()


_SOURCE_PREVIEW_CHARS = 100  # chunk text shown under a chat answer


# Answers the LLM is generating right now. Ollama runs with OLLAMA_NUM_PARALLEL=1 — a further question
# waits for these to finish (roadmap 8.0: up to ~17 s with four at once), so the chat says so.
_active_generations = 0


@contextmanager
def _generating():
    """Count one running generation; the count drops on finish, error and client abort alike."""
    global _active_generations
    _active_generations += 1
    try:
        yield
    finally:
        _active_generations -= 1


def _queue_notice(provider_name: str) -> str | None:
    """German chat notice when this answer has to wait for others (Ollama only — Gemini runs in parallel)."""
    ahead = _active_generations if provider_name == "ollama" else 0
    if not ahead:
        return None
    others = "eine andere Frage" if ahead == 1 else f"{ahead} andere Fragen"
    return f"Das LLM beantwortet gerade {others} — deine kommt danach."


def _ms(since: float) -> int:
    """Milliseconds elapsed since a ``time.perf_counter()`` reading."""
    return round((time.perf_counter() - since) * 1000)


def _who(session_id: str, role: str) -> str:
    """``session=<first 8 chars> role=<role>`` — ties the lines of one question together when several
    people ask at the same time."""
    return f"session={session_id[:8]} role={role}"


def _log_timings(timings: dict, who: str, model: str, usage: dict | None = None, generate_ms: int = 0) -> None:
    """One INFO line per answered query — the basis for spotting slow stages.

    With the provider's token counts it also logs the output rate: ``generate_ms`` is the whole LLM call for
    /query (prefill included) and the time after the first token for the stream. A rate far below the usual
    one with an unchanged prompt size points at the GPU, not at LoreKeeper — 17.10. the model sat partly in
    shared system memory after loading next to another program (106 → 28 characters/s).
    """
    stages = " ".join(f"{k}={v}" for k, v in timings.get("retrieval_stages_ms", {}).items())
    tokens = ""
    if usage and usage.get("tokens_out"):
        rate = f" {usage['tokens_out'] / (generate_ms / 1000):.1f}tok/s" if generate_ms > 0 else ""
        tokens = f" tokens_in={usage.get('tokens_in', 0)} tokens_out={usage['tokens_out']}{rate}"
    logger.info(
        f"Query timings: {who} model={model} condense={timings.get('condense_ms', 0)}ms "
        f"retrieval={timings.get('retrieval_ms', 0)}ms ({stages}) "
        f"first_token={timings.get('first_token_ms', 0)}ms total={timings.get('total_ms', 0)}ms "
        f"chunks={timings.get('chunks', 0)} prompt_chars={timings.get('prompt_chars', 0)}{tokens}"
    )


async def _prepare_query(request: QueryRequest, timings: dict):
    """Shared condense+retrieve step. Returns (session, chunks, system_prompt,
    qa_prompt, search_query) — qa_prompt is None when no chunks passed the
    threshold; search_query is the (possibly condensed) question used for retrieval.
    Condense/retrieval durations, chunk count and prompt size are written into ``timings``."""
    import src.main as main_module
    cm, gen, pm = main_module.conversation_manager, main_module.generator, main_module.prompt_manager

    try:
        filters = effective_filters(request.role, main_module.config.settings.ingestion.sources,
                                    request.metadata_filters)
    except NoPlayerSources:
        logger.warning("Refused %s query: no source is released for players", request.role)
        raise HTTPException(status_code=403, detail="Keine Quelle für Spieler freigegeben") from None
    except InvalidFilter as e:
        logger.warning("Refused %s query: invalid metadata_filters (%s)", request.role, e)
        raise HTTPException(status_code=422, detail=f"Ungültiger metadata_filters: {e}") from None

    session = cm.get_or_create_session(request.session_id, access_role=request.role)
    question = request.question

    history = cm.get_history_for_condense(session.session_id)
    if history and cm.config.condense_question:
        if needs_condense(question):
            condense_prompt = pm.render_condense(history=history, question=question)
            t0 = time.perf_counter()
            try:
                question = await gen.condense_question(condense_prompt, main_module.condense_provider)
            except Exception as e:
                logger.warning("Condense failed (%s: %s); retrieving with the original question",
                               type(e).__name__, e)
            timings["condense_ms"] = _ms(t0)
        else:
            logger.debug("Condense skipped (standalone question)")

    # The one line per query that carries the question — the anchor for the lines that follow
    # (decision roadmap 10.0 #11). A follow-up shows the rewritten form used for retrieval too.
    who = _who(session.session_id, request.role)
    if question != request.question:
        logger.info("Question (%s): %r → condensed %r", who, request.question, question)
    else:
        logger.info("Question (%s): %r", who, question)

    stages: dict[str, float] = {}
    t0 = time.perf_counter()
    chunks = await main_module.retriever.retrieve(
        query=question,
        top_k=request.top_k,
        metadata_filters=filters,
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
        # Condensed question for the sources; the last QA pair resolves references to the previous answer.
        # Sessions are role-bound, so a player only ever sees their own history here.
        qa_prompt = pm.render_qa(chunks=chunk_dicts, question=question,
                                 history=cm.get_answer_history(session.session_id))
    else:
        chunk_dicts = []
        system_prompt = ""
        qa_prompt = None

    timings["prompt_chars"] = len(system_prompt) + len(qa_prompt or "")
    return session, chunks, system_prompt, qa_prompt, question


def _chunk_to_source_dict(c) -> dict:
    """Format a retrieved chunk into a source dictionary for the API response (preview and full text)."""
    return {
        "file": c.source_file,
        "source_path": c.metadata.get("source_path", ""),
        "document_type": c.document_type,
        "heading": c.heading,
        "chunk_preview": c.content[:_SOURCE_PREVIEW_CHARS],
        "content": c.content,
        "images": [p for p in c.metadata.get("images", "").split(IMAGE_PATH_SEPARATOR) if p],
        # ChromaDB keeps the alias list comma-joined (flattened metadata)
        "aliases": [a.strip() for a in c.metadata.get("aliases", "").split(",") if a.strip()],
        "score": c.score,
    }


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute a query against the knowledge base and return the generated answer and sources."""
    start = time.time()
    t_start = time.perf_counter()
    import src.main as main_module
    gen, pm = main_module.generator, main_module.prompt_manager
    timings: dict = {}
    session, chunks, system_prompt, qa_prompt, search_query = await _prepare_query(request, timings)

    usage: dict = {}
    generate_ms = 0
    if qa_prompt is not None:
        try:
            t_gen = time.perf_counter()
            with _generating():
                response = await gen.generate(system_prompt=system_prompt, qa_prompt=qa_prompt)
            generate_ms = _ms(t_gen)
            answer = response.content
            model_used = response.model
            usage = response.usage
        except Exception as e:
            logger.exception("LLM generation failed (%s)", _who(session.session_id, request.role))
            raise HTTPException(status_code=503, detail=f"LLM nicht erreichbar: {e}") from e
    else:
        answer = pm.render_no_context(question=request.question)
        model_used = "none"

    session.add_message("user", request.question)
    session.add_message("assistant", answer)

    sources = [SourceReference(**_chunk_to_source_dict(c)) for c in chunks]
    latency_ms = (time.time() - start) * 1000
    timings["total_ms"] = _ms(t_start)
    _log_timings(timings, _who(session.session_id, request.role), model_used, usage, generate_ms)

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
    import src.main as main_module
    gen, pm, prov = main_module.generator, main_module.prompt_manager, main_module.provider
    timings: dict = {}
    session, chunks, system_prompt, qa_prompt, search_query = await _prepare_query(request, timings)
    sources = [_chunk_to_source_dict(c) for c in chunks]

    async def event_stream():
        """Yield SSE 'token' events during generation and a final 'done' event with metadata.

        If other answers are being generated, a 'status' event with a German notice comes first —
        the UI shows it until the first token arrives.

        A client that leaves mid-answer (page closed, UI read timeout) cancels the generator at a yield;
        that is logged — otherwise the log just stops after the retrieval lines.
        """
        full_answer = ""
        stream_ctx = StreamResult()

        try:
            if qa_prompt is not None:
                if notice := _queue_notice(prov.provider):
                    yield f"data: {json.dumps({'type': 'status', 'content': notice})}\n\n"
                with _generating():
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
        except (asyncio.CancelledError, GeneratorExit):
            logger.info("Stream aborted by the client after %dms (%d chars sent, %s)", _ms(t_start),
                        len(full_answer), _who(session.session_id, request.role))
            raise
        except Exception as e:
            logger.exception("Streaming generation failed (%s)", _who(session.session_id, request.role))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        session.add_message("user", request.question)
        session.add_message("assistant", full_answer)

        usage = dict(stream_ctx.usage)
        if usage:
            session.add_usage(usage)
        timings["total_ms"] = _ms(t_start)
        model_used = stream_ctx.model or prov.model
        _log_timings(timings, _who(session.session_id, request.role), model_used, usage,
                     timings["total_ms"] - timings.get("first_token_ms", timings["total_ms"]))

        done_event = json.dumps({
            "type": "done",
            "session_id": session.session_id,
            "sources": sources,
            "model_used": model_used,
            "search_query": search_query,
            "timings": timings,
            "usage": usage,
            "session_usage": session.usage_totals,
        })
        yield f"data: {done_event}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
