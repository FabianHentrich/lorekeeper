import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from src.api.eval_schemas import (
    EvalJobRequest,
    EvalJobResponse,
    EvalJobStatus,
    EvalResultSummary,
    QAPair,
    QAPairList,
    RetrievalPreviewRequest,
    RetrievalPreviewResponse,
    RetrievedChunkResponse,
)

logger = logging.getLogger(__name__)

eval_router = APIRouter(prefix="/eval", tags=["evaluation"])

_BASE_DIR = Path(__file__).resolve().parents[2] / "evaluation"
_QA_PATH = _BASE_DIR / "qa_pairs.yaml"
_RESULTS_DIR = _BASE_DIR / "results"

# In-memory eval job storage (single job at a time)
_eval_jobs: dict[str, EvalJobStatus] = {}
_MAX_RESULTS = 3


def _get_retriever():
    from src.main import retriever
    return retriever


# ─── QA Pairs CRUD ─────────────────────────────────────────────────────

@eval_router.get("/qa-pairs", response_model=QAPairList)
async def get_qa_pairs():
    if not _QA_PATH.exists():
        return QAPairList(pairs=[])
    data = yaml.safe_load(_QA_PATH.read_text(encoding="utf-8")) or {}
    raw = data.get("pairs", [])
    pairs = [QAPair(**p) for p in raw]
    return QAPairList(pairs=pairs)


@eval_router.put("/qa-pairs", response_model=QAPairList)
async def put_qa_pairs(body: QAPairList):
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
    retriever = _get_retriever()
    start = time.time()
    chunks = await retriever.retrieve(
        query=req.question,
        top_k=req.top_k,
        top_k_rerank=req.top_k_rerank,
        max_per_source=req.max_per_source,
    )
    latency_ms = (time.time() - start) * 1000

    logger.info("Preview: question=%r → %d chunks in %.0fms", req.question, len(chunks), latency_ms)

    return RetrievalPreviewResponse(
        chunks=[
            RetrievedChunkResponse(
                source_file=c.source_file,
                document_type=c.document_type,
                heading=c.heading,
                score=round(c.score, 4),
                content_preview=c.content[:200],
            )
            for c in chunks
        ],
        latency_ms=round(latency_ms, 1),
    )


# ─── Eval job (retrieval + e2e) ──────────────────────────────────────

def _has_running_job() -> bool:
    return any(j.status in ("queued", "running") for j in _eval_jobs.values())


def _cleanup_results(eval_type: str):
    """Keep only the last _MAX_RESULTS files per eval_type."""
    prefix = "retrieval_" if eval_type == "retrieval" else "e2e_"
    files = sorted(_RESULTS_DIR.glob(f"{prefix}*.json"), reverse=True)
    for old in files[_MAX_RESULTS:]:
        old.unlink(missing_ok=True)
        logger.info(f"Deleted old eval result: {old.name}")


@eval_router.post("/run", response_model=EvalJobResponse)
async def start_eval(req: EvalJobRequest):
    if _has_running_job():
        raise HTTPException(status_code=409, detail="An evaluation is already running")

    if not _QA_PATH.exists():
        raise HTTPException(status_code=404, detail="No qa_pairs.yaml found")

    data = yaml.safe_load(_QA_PATH.read_text(encoding="utf-8")) or {}
    qa_pairs = data.get("pairs", [])
    if not qa_pairs:
        raise HTTPException(status_code=400, detail="Golden Set is empty")

    job_id = str(uuid.uuid4())
    job = EvalJobStatus(job_id=job_id, status="queued", total=len(qa_pairs))
    _eval_jobs[job_id] = job

    logger.info(
        "Eval job started: job_id=%s, type=%s, questions=%d, top_k=%d, top_k_rerank=%d, max_per_source=%d",
        job_id, req.eval_type, len(qa_pairs), req.top_k, req.top_k_rerank, req.max_per_source,
    )

    if req.eval_type == "retrieval":
        asyncio.create_task(_run_retrieval_eval(job, qa_pairs, req))
    elif req.eval_type == "e2e":
        asyncio.create_task(_run_e2e_eval(job, qa_pairs))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown eval_type: {req.eval_type}")

    return EvalJobResponse(job_id=job_id, status="queued")


async def _run_retrieval_eval(job: EvalJobStatus, qa_pairs: list[dict], req: EvalJobRequest):
    from evaluation.evaluate_retrieval import run_evaluation_with_retriever

    job.status = "running"
    retriever = _get_retriever()

    def _progress(current, total):
        job.progress = current
        job.total = total

    try:
        report = await run_evaluation_with_retriever(
            qa_pairs=qa_pairs,
            retriever=retriever,
            top_k=req.top_k,
            top_k_rerank=req.top_k_rerank,
            max_per_source=req.max_per_source,
            progress_callback=_progress,
        )

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = _RESULTS_DIR / filename
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        _cleanup_results("retrieval")

        job.result_file = filename
        job.status = "done"
        logger.info(f"Retrieval eval done: hit_rate={report['hit_rate']}, file={filename}")

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        logger.error(f"Retrieval eval failed: {e}")


async def _run_e2e_eval(job: EvalJobStatus, qa_pairs: list[dict]):
    from evaluation.evaluate import evaluate

    job.status = "running"

    def _run_sync():
        return evaluate("http://localhost:8000", qa_pairs)

    try:
        report = await asyncio.get_event_loop().run_in_executor(None, _run_sync)

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = _RESULTS_DIR / filename
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        _cleanup_results("e2e")

        job.result_file = filename
        job.progress = job.total
        job.status = "done"
        logger.info(f"E2E eval done: hit_rate={report['hit_rate']}, file={filename}")

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        logger.error(f"E2E eval failed: {e}")


@eval_router.get("/status/{job_id}", response_model=EvalJobStatus)
async def eval_status(job_id: str):
    if job_id not in _eval_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _eval_jobs[job_id]


# ─── Results ──────────────────────────────────────────────────────────

@eval_router.get("/results", response_model=list[EvalResultSummary])
async def list_results():
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
        except Exception:
            continue

    return summaries


@eval_router.get("/results/{filename}")
async def get_result(filename: str):
    path = _RESULTS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    return json.loads(path.read_text(encoding="utf-8"))


@eval_router.delete("/results/{filename}")
async def delete_result(filename: str):
    path = _RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    path.unlink()
    return {"deleted": filename}
