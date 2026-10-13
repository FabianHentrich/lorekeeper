"""Sources and the index: ingest jobs, source list, folders, scan, reindex, delete, recategorize, wipe, stats.

Ingestion runs as a background job in a thread, one at a time (409 while one runs); anything that
writes or reads the whole index (delete, wipe, recategorize, eval runs) is refused during a job — a
source deleted mid-run came back as orphans. Every change of the collection invalidates the BM25 index.
The job registry lives in memory: after a restart the log is the only record of a failed run.
"""
import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.api.background import spawn_background
from src.api.schemas import (
    IngestJobResponse,
    IngestStatusResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Ingestion job storage (in-memory)
# IngestionResult fields mirrored onto the job status while a job runs
_PROGRESS_FIELDS = ("phase", "documents_processed", "documents_total", "documents_skipped",
                    "chunks_created", "chunks_updated", "chunks_deleted", "duration_seconds")


_ingest_jobs: dict[str, IngestStatusResponse] = {}


_INGEST_BUSY = "Eine Indexierung läuft bereits"


def ingest_running() -> bool:
    """True while an ingest/reindex job is queued or running."""
    return any(j.status in ("queued", "running") for j in _ingest_jobs.values())


def refuse_during_ingest() -> None:
    """409 for anything that writes or reads the whole index while a job is filling it: a source
    deleted mid-run came back as orphans (the job upserted it again), a wipe left a silent partial
    index, an eval measured a half-built index (roadmap 11.0 #5)."""
    if ingest_running():
        raise HTTPException(status_code=409, detail=_INGEST_BUSY)


def _start_ingest_job(run_sync, label: str) -> str:
    """Create a job, launch the sync orchestrator run in a thread, and return the job_id.

    ``run_sync(progress_cb)`` returns an IngestionResult; ``label`` names the job in the log
    ("incremental", "full", "reindex <id>"). Start, end and failure are logged — the job registry is
    in memory, so after a restart the log is the only record of why a run failed."""
    import src.main as main_module

    # Parallel runs race on delete/upsert of the same chunks
    refuse_during_ingest()

    job_id = str(uuid.uuid4())
    _ingest_jobs[job_id] = IngestStatusResponse(job_id=job_id, status="queued")
    logger.info("Ingest job %s started (%s)", job_id, label)

    def _update_progress(result):
        """Copy orchestrator progress fields onto the tracked job status."""
        job = _ingest_jobs[job_id]
        for name in _PROGRESS_FIELDS:
            setattr(job, name, getattr(result, name))

    async def _runner():
        """Run the sync orchestrator off-thread and update job state on success/error."""
        _ingest_jobs[job_id].status = "running"
        try:
            # asyncio.to_thread, not run_in_threadpool: the job outlives the request that started it
            result = await asyncio.to_thread(run_sync, _update_progress)
            _update_progress(result)
            _ingest_jobs[job_id].status = "done"
            _ingest_jobs[job_id].errors = result.errors
            main_module.bm25_index.invalidate()
            logger.info("Ingest job %s done (%s): %d file errors", job_id, label, len(result.errors))
        except Exception as e:
            logger.exception("Ingest job %s failed (%s)", job_id, label)
            _ingest_jobs[job_id].status = "error"
            _ingest_jobs[job_id].errors.append(str(e))

    spawn_background(_runner())
    return job_id


@router.post("/ingest", response_model=IngestJobResponse)
async def ingest(full: bool = False):
    """Start an asynchronous ingestion job to process and embed all configured sources.

    Incremental by default (unchanged files are skipped); ``?full=true`` re-chunks and
    re-embeds every file, e.g. after chunking settings changed.
    """
    import src.main as main_module
    from src.ingestion.orchestrator import IngestionOrchestrator
    vs = main_module.vectorstore

    def _run(progress_cb):
        """Run ingestion across all configured sources."""
        return IngestionOrchestrator(config=main_module.config).run(
            vectorstore=vs, progress_callback=progress_cb, full=full,
        )

    return IngestJobResponse(job_id=_start_ingest_job(_run, "full" if full else "incremental"), status="queued")


@router.get("/ingest/active", response_model=IngestStatusResponse | None)
async def ingest_active():
    """The queued or running ingest job, or null — lets a client attach to it after a 409."""
    return next((j for j in _ingest_jobs.values() if j.status in ("queued", "running")), None)


@router.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
async def ingest_status(job_id: str):
    """Get the current progress and status of an ingestion job."""
    if job_id not in _ingest_jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    return _ingest_jobs[job_id]


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """Get statistics about the vector store, like total chunk count."""
    import src.main as main_module
    return StatsResponse(chunk_count=main_module.vectorstore.count())


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
        raise HTTPException(status_code=404, detail=f"Unbekannte Quelle: {source_id}")

    src_path = Path(source.path).resolve()
    if not src_path.exists():
        logger.warning(f"list_source_folders: path does not exist: {src_path} (source={source_id})")
        return {"folders": [], "error": f"Pfad existiert nicht: {src_path}"}
    if src_path.is_file():
        return {"folders": [], "note": "Dateiquelle — keine Unterordner"}

    supported = set(main_module.config.settings.ingestion.supported_formats)
    return {"folders": _scan_path(src_path, supported)}


@router.post("/sources/scan")
async def scan_path(payload: dict):
    """Scan a freeform path (not yet a configured source). Body: {"path": "..."}."""
    import src.main as main_module
    raw_path = (payload or {}).get("path", "")
    if not raw_path:
        raise HTTPException(status_code=400, detail="Feld 'path' fehlt")

    p = Path(raw_path).resolve()
    logger.info(f"Scanning path: {p}")
    if not p.exists():
        logger.warning(f"Scan: path does not exist: {p}")
        return {"folders": [], "is_file": False, "error": f"Pfad existiert nicht: {p}"}
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
        raise HTTPException(status_code=400, detail="Feld 'sources' fehlt")
    try:
        new_sources = [SourceConfig(**s) for s in raw_sources]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ungültige Quellen-Konfiguration: {e}") from e
    ids = [s.id for s in new_sources]
    # Only new ids are checked — an existing (legacy) id must never block saving the list.
    # '.'/'..' get normalized away by HTTP clients, so /sources/{id}/... could not reach them.
    existing = {s.id for s in main_module.config.settings.ingestion.sources}
    bad = [i for i in ids if i not in existing
           and (not i.strip() or i != i.strip() or i in (".", ".."))]
    if bad:
        raise HTTPException(status_code=400, detail=f"Ungültige Quellen-ID(s): {bad} — nicht leer, nicht "
                            "'.' oder '..', ohne Leerzeichen am Anfang oder Ende")
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=400, detail="Doppelte Quellen-IDs")
    main_module.config.settings.ingestion.sources = new_sources
    main_module.config.save_sources()
    # player_visible decides what players can search — the log must show when it changed
    logger.info("Sources saved: %d sources, player_visible=%s", len(new_sources),
                [s.id for s in new_sources if s.player_visible])
    return {"sources": [s.model_dump() for s in new_sources]}


@router.post("/sources/{source_id:path}/reindex", response_model=IngestJobResponse)
async def reindex_source(source_id: str):
    """Re-ingest a single source: every file re-parsed and re-embedded (``full=True``).

    No delete up front — the source stays searchable during the run: each file's old chunks are
    replaced when its new ones are written, files gone from disk are removed at the end. (Deleting
    first left the source missing from every answer for the whole reindex, roadmap 11.0 #10.)
    """
    import src.main as main_module
    from src.ingestion.orchestrator import IngestionOrchestrator
    vs = main_module.vectorstore

    if not any(s.id == source_id for s in main_module.config.settings.ingestion.sources):
        raise HTTPException(status_code=404, detail=f"Unbekannte Quelle: {source_id}")

    def _run(progress_cb):
        """Re-ingest just this source's files, unchanged ones included."""
        return IngestionOrchestrator(config=main_module.config).run(
            vectorstore=vs, only_source_id=source_id, progress_callback=progress_cb, full=True,
        )

    return IngestJobResponse(job_id=_start_ingest_job(_run, f"reindex {source_id}"), status="queued")


@router.delete("/sources/{source_id:path}")
async def delete_source(source_id: str):
    """Remove a source from config and delete all its chunks."""
    import src.main as main_module
    vs = main_module.vectorstore

    sources = main_module.config.settings.ingestion.sources
    if not any(s.id == source_id for s in sources):
        raise HTTPException(status_code=404, detail=f"Unbekannte Quelle: {source_id}")
    refuse_during_ingest()

    deleted = vs.delete_by_source_id(source_id)
    main_module.bm25_index.invalidate()  # else hybrid search keeps finding the deleted chunks
    main_module.config.settings.ingestion.sources = [s for s in sources if s.id != source_id]
    main_module.config.save_sources()
    logger.info("Source removed: %s (%d chunks deleted)", source_id, deleted)
    return {"deleted_chunks": deleted}


@router.post("/sources/recategorize")
async def recategorize_endpoint():
    """Trigger a recategorization task to update metadata for existing configurations."""
    import src.main as main_module
    from src.ingestion.recategorize import recategorize
    refuse_during_ingest()
    stats = await run_in_threadpool(recategorize, config=main_module.config, vectorstore=main_module.vectorstore)
    main_module.bm25_index.invalidate()  # BM25 holds its own copy of the metadata the filters check
    return stats


@router.post("/admin/wipe")
async def wipe_collection(payload: dict):
    """Drop the entire ChromaDB collection. Requires confirm='DELETE' in body."""
    if payload.get("confirm") != "DELETE":
        raise HTTPException(status_code=400, detail="Zum Löschen confirm='DELETE' mitschicken")
    import src.main as main_module
    refuse_during_ingest()
    main_module.vectorstore.wipe_collection()
    main_module.bm25_index.invalidate()
    return {"status": "wiped"}
