"""Ingestion pipeline: discover files → parse → chunk → embed → store.

Runs as a background job inside the backend (``/ingest``, ``/sources/{id}/reindex``) or as a CLI
(``python -m src.ingestion.orchestrator``) — the CLI opens ``chroma_data`` itself, so only with the backend
stopped. Incremental via the SHA-256 content hash, not timestamps (unreliable on Windows mounts and external
repos); ``full=True`` re-processes everything. Chunks are upserted in batches of ``UPSERT_BATCH_CHUNKS``
(one file never spans two batches); files gone from disk are deleted from the index, and a run over all
sources also deletes the chunks of sources no longer in sources.yaml. Every chunk carries
``source_id``, ``group`` and ``content_category`` from its source config.
"""
import fnmatch
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.config.manager import ConfigManager, SourceConfig
from src.ingestion.chunking import Chunk, chunk_documents
from src.ingestion.parsers.base import BaseParser
from src.ingestion.parsers.image import ImageMetaParser
from src.ingestion.parsers.markdown import MarkdownParser
from src.ingestion.parsers.pdf import PDFParser

logger = logging.getLogger(__name__)

# Pending chunks are embedded + upserted once this many have accumulated — gives
# progress during embedding and keeps finished work if a later file crashes the run.
UPSERT_BATCH_CHUNKS = 500


def _resolve_category(file_path: Path, source: SourceConfig) -> tuple[str, str]:
    """Resolve (content_category, group) for a file based on its source config.

    File source → always (default_category, source.group).
    Folder source → category_map[top_folder] if matched, else fallback.
    category_map values can be a plain string (inherits source.group) or a dict
    with 'category' and optional 'group' override.
    """
    src_path = Path(source.path).resolve()
    fallback_cat = source.default_category or "misc"
    fallback_group = source.group

    if src_path.is_file():
        return fallback_cat, fallback_group

    try:
        relative = file_path.resolve().relative_to(src_path)
    except ValueError:
        return fallback_cat, fallback_group

    if len(relative.parts) > 1:
        top = relative.parts[0]
        # Case-insensitive lookup against category_map keys
        for key, entry in source.category_map.items():
            if key.lower() == top.lower():
                if isinstance(entry, dict):
                    return entry["category"], entry.get("group", fallback_group)
                return entry, fallback_group

    return fallback_cat, fallback_group


def source_base(source: SourceConfig) -> Path:
    """Base path that stored ``source_file`` values are relative to: the folder, or a file source's parent.
    Shared with recategorize, which resolves stored chunks back to files."""
    p = Path(source.path).resolve()
    return p.parent if p.is_file() else p


def _compute_content_hash(file_path: Path) -> str:
    """SHA-256 of file bytes; used to skip re-embedding unchanged files across runs."""
    content = file_path.read_bytes()
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _is_excluded(file_path: Path, base_path: Path, exclude_patterns: list[str]) -> bool:
    """Check if a file matches any of the configured exclude patterns.

    The patterns are evaluated against both the file's exact name
    and its relative path from the base directory, supporting glob
    syntax like '*.md' or 'folder/*'.
    """
    try:
        relative = str(file_path.relative_to(base_path))
    except ValueError:
        return False

    # Normalize path separators
    relative = relative.replace("\\", "/")

    for pattern in exclude_patterns:
        if fnmatch.fnmatch(relative, pattern):
            return True
        if fnmatch.fnmatch(file_path.name, pattern):
            return True

    return False


@dataclass
class IngestionResult:
    """Tracks the progress and statistics of an ongoing or completed
    ingestion job, including document and chunk counts and timing.
    """
    documents_processed: int = 0
    documents_total: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    phase: str = "starting"
    errors: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Elapsed seconds since start; live until end_time is set."""
        end = self.end_time or datetime.now(UTC)
        return (end - self.start_time).total_seconds()


class IngestionOrchestrator:
    """Coordinates the entire ingestion pipeline: discovering files,
    parsing formats, chunking text, computing metadata/hashes, and
    syncing the results (upserts and deletes) with the vector store.
    """
    def __init__(self, config: ConfigManager):
        """Register the built-in parsers (Markdown, PDF, image metadata).

        ``config`` is required: the backend passes its own ``main_module.config`` — a second
        ConfigManager would never see source edits made via ``PUT /sources``. Only the CLI uses the
        default instance.
        """
        self.config = config
        self.parsers: list[BaseParser] = [
            MarkdownParser(skip_sections=self.config.settings.ingestion.skip_sections),
            PDFParser(pdf_config=self.config.settings.ingestion.pdf),
            ImageMetaParser(),
        ]

    def _get_parser(self, file_path: Path) -> BaseParser | None:
        """Return the first registered parser that accepts this file, or None."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def _discover_files(self, only_source_id: str | None = None,
                        sources: list[SourceConfig] | None = None) -> list[tuple[Path, SourceConfig]]:
        """Returns list of (file_path, source) tuples.

        If only_source_id is given, restricts discovery to that single source —
        used by per-source reindex from the UI. ``sources`` defaults to the configured list.
        """
        settings = self.config.settings
        files: list[tuple[Path, SourceConfig]] = []

        for source in (settings.ingestion.sources if sources is None else sources):
            if only_source_id and source.id != only_source_id:
                continue

            src_path = Path(source.path).resolve()
            if not src_path.exists():
                logger.warning(f"Source path does not exist: {src_path} (id={source.id})")
                continue

            exclude = settings.ingestion.exclude_patterns + source.exclude_patterns

            if src_path.is_file():
                if src_path.suffix.lower() in settings.ingestion.supported_formats:
                    files.append((src_path, source))
                else:
                    logger.warning(f"Unsupported file extension for source {source.id}: {src_path}")
                continue

            # Folder source — suffix compared lowercased; rglob(f"*{ext}") is case-sensitive on Linux
            supported = {ext.lower() for ext in settings.ingestion.supported_formats}
            for file_path in src_path.rglob("*"):
                if (file_path.suffix.lower() in supported and file_path.is_file()
                        and not _is_excluded(file_path, src_path, exclude)):
                    files.append((file_path, source))

        return files

    def run(self, vectorstore=None, only_source_id: str | None = None,
            progress_callback=None, full: bool = False) -> IngestionResult:
        """Execute the ingestion pipeline.

        This method discovers relevant files, parses them into chunks, assigns metadata
        (e.g. content hashes and categories), and synchronizes them with the vector store.
        It correctly handles insertions of new chunks, updates of modified documents,
        and deletions of files that no longer exist on disk (orphans).

        ``full=True`` re-processes files whose content hash is unchanged — needed after
        chunking settings or parser logic changed.
        """
        result = IngestionResult()
        settings = self.config.settings

        def _report():
            """Push the live IngestionResult to the caller, if a callback was provided."""
            if progress_callback:
                progress_callback(result)

        # One snapshot of the source list for the whole run — discovery and the hash/orphan scope must
        # agree even if the list is edited on the Sources page meanwhile
        sources = list(settings.ingestion.sources)
        files = self._discover_files(only_source_id=only_source_id, sources=sources)
        logger.info(f"Discovered {len(files)} files for ingestion")

        result.documents_total = len(files)
        result.phase = "parsing"
        _report()

        # Source-scoped existing-hash lookup — unchanged files are skipped before parsing
        existing: dict[tuple[str, str], str] = {}
        if vectorstore is not None:
            scope_ids = [only_source_id] if only_source_id else [s.id for s in sources]
            for sid in scope_ids:
                for src_file, h in vectorstore.get_content_hashes_for_source(sid).items():
                    existing[(sid, src_file)] = h
            logger.info(f"Existing files in scope: {len(existing)}")

        # Track which (source_id, source_file) pairs we process (orphan detection)
        processed: set[tuple[str, str]] = set()
        # Files of the pending batch already in the store — old chunks are dropped before upsert
        changed: set[tuple[str, str]] = set()
        pending: list[Chunk] = []

        def _flush():
            """Embed + upsert the pending batch. A file's chunks are always in one batch."""
            if vectorstore is None or not pending:
                return
            result.phase = "embedding"
            _report()
            # Drop old chunks of changed files first — otherwise a file that shrinks
            # keeps its trailing chunks from the previous version.
            for (sid, src_file) in changed:
                vectorstore.delete_by_source_file(sid, src_file)
            logger.info(f"Upserting batch of {len(pending)} chunks...")
            vectorstore.upsert_chunks(pending)
            pending.clear()
            changed.clear()
            result.phase = "parsing"
            _report()

        total_files = len(files)
        for idx, (file_path, source) in enumerate(files, 1):
            parser = self._get_parser(file_path)
            if parser is None:
                continue

            base = source_base(source)
            relative_path = str(file_path.relative_to(base))
            key = (source.id, relative_path)

            try:
                content_hash = _compute_content_hash(file_path)
                if not full and existing.get(key) == content_hash:
                    processed.add(key)
                    result.documents_processed += 1
                    result.documents_skipped += 1
                    _report()
                    continue

                if idx % 10 == 1 or idx == total_files:
                    logger.info(f"Parsing [{idx}/{total_files}]: {file_path.name}")
                documents = parser.parse(file_path, base)
                chunks = chunk_documents(documents, settings.chunking)

                content_category, group = _resolve_category(file_path, source)
                last_modified = datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=UTC
                ).isoformat()

                for chunk in chunks:
                    chunk.source_file = relative_path
                    chunk.metadata.update({
                        "content_hash": content_hash,
                        "content_category": content_category,
                        "source_id": source.id,
                        "group": group,
                        "last_modified": last_modified,
                    })

                pending.extend(chunks)
                processed.add(key)
                if key in existing:
                    changed.add(key)
                    result.chunks_updated += len(chunks)
                else:
                    result.chunks_created += len(chunks)
                result.documents_processed += 1
                _report()

                if len(pending) >= UPSERT_BATCH_CHUNKS:
                    _flush()

            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.exception(error_msg)
                result.errors.append(error_msg)
                # Keep the previously indexed version instead of treating the file as deleted
                processed.add(key)

        _flush()

        if vectorstore is not None:
            # Orphans: anything in existing that wasn't processed this run
            for (sid, src_file) in set(existing.keys()) - processed:
                result.chunks_deleted += vectorstore.delete_by_source_file(sid, src_file)
            # Chunks of sources that are no longer configured — an entry removed from sources.yaml by hand
            # instead of DELETE /sources/{id} stayed searchable (13.10.: 36 chunks of two sim sources).
            # Only on runs over all sources, and never with an empty list: a missing or unreadable
            # sources.yaml must not wipe the index.
            if only_source_id is None and sources:
                for sid in sorted(vectorstore.list_source_ids() - {s.id for s in sources}):
                    deleted = vectorstore.delete_by_source_id(sid)
                    result.chunks_deleted += deleted
                    logger.warning("Deleted %d chunks of source '%s' — it is no longer in sources.yaml",
                                   deleted, sid)

        result.end_time = datetime.now(UTC)
        logger.info(
            f"Ingestion complete: {result.documents_processed} docs, "
            f"{result.chunks_created} created, {result.chunks_updated} updated, "
            f"{result.chunks_deleted} deleted, {len(result.errors)} errors, "
            f"{result.duration_seconds:.1f}s"
        )

        return result


if __name__ == "__main__":
    from src.config.manager import config_manager
    from src.logging_setup import setup_logging
    setup_logging(config_manager.settings.logging)  # same file, levels and suppress list as the backend
    orchestrator = IngestionOrchestrator(config=config_manager)

    # Lazy imports to avoid circular dependency at module level
    from src.retrieval.embeddings import EmbeddingService
    from src.retrieval.vectorstore import VectorStoreService

    embedding_service = EmbeddingService(config_manager.settings.embeddings)
    vs = VectorStoreService(config_manager.settings.vectorstore, embedding_service)

    result = orchestrator.run(vectorstore=vs)
    print(f"Done: {result.documents_processed} docs, {result.chunks_created} chunks created")
