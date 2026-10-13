"""/eval/* API: golden set, single-question preview, chunk inspector, eval jobs and stored reports.

Runs in the backend process with the shared retriever — the eval CLI opens ``chroma_data`` itself and must
not run while the backend does. One eval job at a time (409, check and start under a lock). Retrieval evals
call the retriever directly; end-to-end evals send every question through this server's own ``/query``.
Only the newest ``_MAX_RESULTS`` reports per eval type are kept. ``/eval/preview`` takes ``role`` and gets
the same player filter as ``/query``.
"""
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from evaluation.evaluate_retrieval import load_qa_pairs, save_report
from src.api.access import InvalidFilter, NoPlayerSources, effective_filters
from src.api.eval_schemas import (
    ChunkListResponse,
    EvalJobRequest,
    EvalJobResponse,
    EvalJobStatus,
    EvalResultSummary,
    InspectedChunk,
    QAPair,
    QAPairList,
    RetrievalPreviewRequest,
    RetrievalPreviewResponse,
    RetrievedChunkResponse,
)
from src.api.routes import _spawn_background, refuse_during_ingest

logger = logging.getLogger(__name__)

eval_router = APIRouter(prefix="/eval", tags=["evaluation"])

_BASE_DIR = Path(__file__).resolve().parents[2] / "evaluation"
_QA_PATH = _BASE_DIR / "qa_pairs.yaml"
_RESULTS_DIR = _BASE_DIR / "results"

# In-memory eval job storage (single job at a time)
_eval_jobs: dict[str, EvalJobStatus] = {}
_MAX_RESULTS = 3
_PREVIEW_CHARS = 200  # chunk text per hit in the single-question preview

# Guards the check-then-create window in start_eval so two concurrent POSTs
# can't both slip past _has_running_job().
_eval_start_lock = asyncio.Lock()

# ─── QA Pairs CRUD ─────────────────────────────────────────────────────

@eval_router.get("/qa-pairs", response_model=QAPairList)
async def get_qa_pairs():
    """Load the current list of QA pairs from the golden set file."""
    return QAPairList(pairs=[QAPair(**p) for p in load_qa_pairs(_QA_PATH)])


@eval_router.put("/qa-pairs", response_model=QAPairList)
async def put_qa_pairs(body: QAPairList):
    """Overwrite the QA pairs golden set file with a new list."""
    _QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"pairs": [p.model_dump() for p in body.pairs]}
    _QA_PATH.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Golden Set saved: %d QA pairs", len(body.pairs))
    return body


# ─── Single-question preview ──────────────────────────────────────────

@eval_router.post("/preview", response_model=RetrievalPreviewResponse)
async def retrieval_preview(req: RetrievalPreviewRequest):
    """Run a single question against the retriever and return the raw top chunks without LLM generation."""
    import src.main as main_module

    try:
        filters = effective_filters(req.role, main_module.config.settings.ingestion.sources, req.metadata_filters)
    except NoPlayerSources:
        logger.warning("Refused %s preview: no source is released for players", req.role)
        raise HTTPException(status_code=403, detail="Keine Quelle für Spieler freigegeben") from None
    except InvalidFilter as e:
        logger.warning("Refused %s preview: invalid metadata_filters (%s)", req.role, e)
        raise HTTPException(status_code=422, detail=f"Ungültiger metadata_filters: {e}") from None

    retriever = main_module.retriever
    start = time.time()
    chunks = await retriever.retrieve(
        query=req.question,
        top_k=req.top_k,
        top_k_rerank=req.top_k_rerank,
        max_per_source=req.max_per_source,
        hybrid=req.hybrid,
        metadata_filters=filters,
    )
    latency_ms = (time.time() - start) * 1000

    logger.info("Preview: question=%r → %d chunks in %.0fms", req.question, len(chunks), latency_ms)

    return RetrievalPreviewResponse(
        chunks=[
            RetrievedChunkResponse(
                source_id=c.metadata.get("source_id", ""),
                source_file=c.source_file,
                document_type=c.document_type,
                heading=c.heading,
                score=round(c.score, 4),
                content_preview=c.content[:_PREVIEW_CHARS],
            )
            for c in chunks
        ],
        latency_ms=round(latency_ms, 1),
    )


# ─── Chunk inspector ─────────────────────────────────────────────────

@eval_router.get("/chunks/files")
async def chunk_files(source_id: str):
    """Indexed files of one source, sorted — the file list of the chunk inspector."""
    import src.main as main_module
    try:
        files = await run_in_threadpool(main_module.vectorstore.list_files, source_id)
    except Exception as e:
        logger.exception("Chunk inspector: listing files of %r failed", source_id)
        raise HTTPException(status_code=503, detail=f"ChromaDB-Abfrage fehlgeschlagen: {e}") from e
    return {"files": files}


@eval_router.get("/chunks", response_model=ChunkListResponse)
async def file_chunks(source_id: str, source_file: str):
    """Every stored chunk of one file in document order, with full text and metadata."""
    import src.main as main_module
    items = await run_in_threadpool(main_module.vectorstore.get_chunks_for_file, source_id, source_file)
    if not items:
        raise HTTPException(status_code=404, detail=f"Keine Chunks für {source_id}/{source_file}")
    items.sort(key=lambda i: (i["metadata"].get("chunk_index", 0), i["id"]))
    return ChunkListResponse(
        source_id=source_id,
        source_file=source_file,
        chunks=[
            InspectedChunk(
                id=i["id"],
                chunk_index=i["metadata"].get("chunk_index", 0),
                heading=i["metadata"].get("heading_hierarchy", ""),
                chars=len(i["content"]),
                content=i["content"],
                metadata=i["metadata"],
            )
            for i in items
        ],
    )


# ─── Eval job (retrieval + e2e) ──────────────────────────────────────

def _has_running_job() -> bool:
    """Check if any evaluation job is currently queued or running."""
    return any(j.status in ("queued", "running") for j in _eval_jobs.values())


def _cleanup_results(eval_type: str):
    """Keep only the last _MAX_RESULTS files per eval_type."""
    files = sorted(_RESULTS_DIR.glob(f"{eval_type}_*.json"), reverse=True)
    for old in files[_MAX_RESULTS:]:
        old.unlink(missing_ok=True)
        logger.info(f"Deleted old eval result: {old.name}")


@eval_router.post("/run", response_model=EvalJobResponse)
async def start_eval(req: EvalJobRequest, request: Request):
    """Start an asynchronous evaluation job processing the entire golden set."""
    if not _QA_PATH.exists():
        raise HTTPException(status_code=404, detail="qa_pairs.yaml nicht gefunden")

    qa_pairs = load_qa_pairs(_QA_PATH)
    if not qa_pairs:
        raise HTTPException(status_code=400, detail="Golden Set ist leer")

    refuse_during_ingest()  # numbers from a half-built index would be meaningless
    async with _eval_start_lock:
        if _has_running_job():
            raise HTTPException(status_code=409, detail="Es läuft bereits eine Evaluation")

        job_id = str(uuid.uuid4())
        job = EvalJobStatus(job_id=job_id, status="queued", total=len(qa_pairs))
        _eval_jobs[job_id] = job

        logger.info(
            "Eval job started: job_id=%s, type=%s, questions=%d, top_k=%s, top_k_rerank=%s, max_per_source=%s",
            job_id, req.eval_type, len(qa_pairs), req.top_k, req.top_k_rerank, req.max_per_source,
        )

        if req.eval_type == "retrieval":
            _spawn_background(_run_retrieval_eval(job, qa_pairs, req))
        else:
            # Same server that received this request — no hardcoded host/port
            _spawn_background(_run_e2e_eval(job, qa_pairs, str(request.base_url).rstrip("/")))

    return EvalJobResponse(job_id=job_id, status="queued")


async def _run_retrieval_eval(job: EvalJobStatus, qa_pairs: list[dict], req: EvalJobRequest):
    """Execute a retrieval-only evaluation and save the report."""
    import src.main as main_module
    from evaluation.evaluate_retrieval import run_evaluation_with_retriever
    job.status = "running"
    retriever = main_module.retriever

    def _progress(current, total):
        """Mirror evaluator progress onto the tracked job status."""
        job.progress = current
        job.total = total

    try:
        report = await run_evaluation_with_retriever(
            qa_pairs=qa_pairs,
            retriever=retriever,
            top_k=req.top_k,
            top_k_rerank=req.top_k_rerank,
            max_per_source=req.max_per_source,
            hybrid=req.hybrid,
            progress_callback=_progress,
        )

        filename = save_report(report, _RESULTS_DIR, "retrieval").name
        _cleanup_results("retrieval")

        job.result_file = filename
        job.status = "done"
        logger.info(f"Retrieval eval done: hit_rate={report['hit_rate']}, file={filename}")

    except Exception as e:
        logger.exception("Retrieval eval job %s failed", job.job_id)
        job.status = "error"
        job.error = str(e)


async def _run_e2e_eval(job: EvalJobStatus, qa_pairs: list[dict], api_url: str):
    """Execute an end-to-end evaluation including the LLM answer generation."""
    from evaluation.evaluate import evaluate

    job.status = "running"

    try:
        # Blocking HTTP calls per question; to_thread because the job outlives the starting request
        report = await asyncio.to_thread(evaluate, api_url, qa_pairs)

        filename = save_report(report, _RESULTS_DIR, "e2e").name
        _cleanup_results("e2e")

        job.result_file = filename
        job.progress = job.total
        job.status = "done"
        logger.info(f"E2E eval done: hit_rate={report['hit_rate']}, file={filename}")

    except Exception as e:
        logger.exception("E2E eval job %s failed", job.job_id)
        job.status = "error"
        job.error = str(e)


@eval_router.get("/status/{job_id}", response_model=EvalJobStatus)
async def eval_status(job_id: str):
    """Get the current state and progress of an active evaluation job."""
    if job_id not in _eval_jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    return _eval_jobs[job_id]


# ─── Results ──────────────────────────────────────────────────────────

@eval_router.get("/results", response_model=list[EvalResultSummary])
async def list_results():
    """List all available stored evaluation reports."""
    if not _RESULTS_DIR.exists():
        return []

    summaries = []
    for f in sorted(_RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            eval_type = "retrieval" if f.name.startswith("retrieval_") else "e2e"
            summaries.append(EvalResultSummary(
                filename=f.name,
                timestamp=data.get("timestamp", ""),
                eval_type=eval_type,
                hit_rate=data.get("hit_rate", 0),
                total_questions=data.get("total_questions", 0),
                config=data.get("config", {}),
            ))
        except Exception as e:
            logger.warning("Skipping unreadable eval result %s (%s: %s)", f.name, type(e).__name__, e)
            continue

    return summaries


def _result_path(filename: str) -> Path:
    """Resolve a result file, rejecting names that would escape the results dir (e.g. '..' + backslash on Windows)."""
    if not filename or filename in (".", "..") or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail=f"Ungültiger Ergebnis-Dateiname: {filename!r}")
    return _RESULTS_DIR / filename


@eval_router.get("/results/{filename}")
async def get_result(filename: str):
    """Download the full JSON report for a specific evaluation run."""
    path = _result_path(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Ergebnis nicht gefunden")
    return json.loads(path.read_text(encoding="utf-8"))


@eval_router.delete("/results/{filename}")
async def delete_result(filename: str):
    """Delete a specific evaluation report file."""
    path = _result_path(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Ergebnis nicht gefunden")
    path.unlink()
    return {"deleted": filename}
