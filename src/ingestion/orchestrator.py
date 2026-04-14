import fnmatch
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config.manager import ConfigManager, SourceConfig, config_manager
from src.ingestion.chunking import Chunk, chunk_documents
from src.ingestion.parsers.base import BaseParser
from src.ingestion.parsers.image import ImageMetaParser
from src.ingestion.parsers.markdown import MarkdownParser
from src.ingestion.parsers.pdf import PDFParser

logger = logging.getLogger(__name__)


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


class IngestionResult:
    """Tracks the progress and statistics of an ongoing or completed
    ingestion job, including document and chunk counts and timing.
    """
    def __init__(self):
        """Start an empty result bag with the current UTC start timestamp."""
        self.documents_processed: int = 0
        self.documents_total: int = 0
        self.chunks_created: int = 0
        self.chunks_updated: int = 0
        self.chunks_deleted: int = 0
        self.phase: str = "starting"
        self.errors: list[str] = []
        self.start_time: datetime = datetime.now(timezone.utc)
        self.end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Elapsed seconds since start; live until end_time is set."""
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()


class IngestionOrchestrator:
    """Coordinates the entire ingestion pipeline: discovering files,
    parsing formats, chunking text, computing metadata/hashes, and
    syncing the results (upserts and deletes) with the vector store.
    """
    def __init__(self, config: ConfigManager | None = None):
        """Register the built-in parsers (Markdown, PDF, image metadata)."""
        self.config = config or config_manager
        self.parsers: list[BaseParser] = [
            MarkdownParser(),
            PDFParser(pdf_config=self.config.settings.ingestion.pdf),
            ImageMetaParser(),
        ]

    def _get_parser(self, file_path: Path) -> BaseParser | None:
        """Return the first registered parser that accepts this file, or None."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def _discover_files(self, only_source_id: str | None = None) -> list[tuple[Path, SourceConfig]]:
        """Returns list of (file_path, source) tuples.

        If only_source_id is given, restricts discovery to that single source —
        used by per-source reindex from the UI.
        """
        settings = self.config.settings
        files: list[tuple[Path, SourceConfig]] = []

        for source in settings.ingestion.sources:
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

            # Folder source
            for ext in settings.ingestion.supported_formats:
                for file_path in src_path.rglob(f"*{ext}"):
                    if not _is_excluded(file_path, src_path, exclude):
                        files.append((file_path, source))

        return files

    @staticmethod
    def _source_base(source: SourceConfig) -> Path:
        """Effective base path for relative_to(): file's parent for file sources."""
        p = Path(source.path).resolve()
        return p.parent if p.is_file() else p

    def run(self, vectorstore=None, only_source_id: str | None = None,
            progress_callback=None) -> IngestionResult:
        """Execute the ingestion pipeline.

        This method discovers relevant files, parses them into chunks, assigns metadata
        (e.g. content hashes and categories), and synchronizes them with the vector store.
        It correctly handles insertions of new chunks, updates of modified documents,
        and deletions of files that no longer exist on disk (orphans).
        """
        result = IngestionResult()
        settings = self.config.settings

        def _report():
            """Push the live IngestionResult to the caller, if a callback was provided."""
            if progress_callback:
                progress_callback(result)

        files = self._discover_files(only_source_id=only_source_id)
        logger.info(f"Discovered {len(files)} files for ingestion")

        result.documents_total = len(files)
        result.phase = "parsing"
        _report()

        # Track which (source_id, source_file) pairs we process (orphan detection)
        processed: set[tuple[str, str]] = set()
        all_chunks: list[Chunk] = []

        total_files = len(files)
        for idx, (file_path, source) in enumerate(files, 1):
            parser = self._get_parser(file_path)
            if parser is None:
                continue

            try:
                base = self._source_base(source)
                if idx % 10 == 1 or idx == total_files:
                    logger.info(f"Parsing [{idx}/{total_files}]: {file_path.name}")
                documents = parser.parse(file_path, base)
                chunks = chunk_documents(documents, settings.chunking)

                content_hash = _compute_content_hash(file_path)
                content_category, group = _resolve_category(file_path, source)
                last_modified = datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()

                relative_path = str(file_path.relative_to(base))

                for chunk in chunks:
                    chunk.source_file = relative_path
                    chunk.metadata.update({
                        "content_hash": content_hash,
                        "content_category": content_category,
                        "source_id": source.id,
                        "group": group,
                        "source_collection": base.name,
                        "last_modified": last_modified,
                    })

                all_chunks.extend(chunks)
                processed.add((source.id, relative_path))
                result.documents_processed += 1
                _report()

            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        logger.info(f"Parsing complete: {result.documents_processed} docs, {len(all_chunks)} chunks. Starting vectorstore sync...")

        if vectorstore is not None:
            result.phase = "embedding"
            _report()

            # Source-scoped existing-hash lookup
            scope_ids = [only_source_id] if only_source_id else [s.id for s in settings.ingestion.sources]
            existing: dict[tuple[str, str], str] = {}
            for sid in scope_ids:
                for src_file, h in vectorstore.get_content_hashes_for_source(sid).items():
                    existing[(sid, src_file)] = h
            logger.info(f"Existing chunks in scope: {len(existing)}")

            for chunk in all_chunks:
                key = (chunk.metadata.get("source_id", ""), chunk.source_file)
                chunk_hash = chunk.metadata.get("content_hash", "")
                if existing.get(key) == chunk_hash:
                    continue
                elif key in existing:
                    result.chunks_updated += 1
                else:
                    result.chunks_created += 1

            # Orphans: anything in existing that wasn't processed this run
            for (sid, src_file) in set(existing.keys()) - processed:
                vectorstore.delete_by_source_file(sid, src_file)
                result.chunks_deleted += 1

            logger.info(f"Upserting {len(all_chunks)} chunks (new={result.chunks_created}, updated={result.chunks_updated}, deleted={result.chunks_deleted})...")
            vectorstore.upsert_chunks(all_chunks)
            logger.info("Upsert complete")
        else:
            result.chunks_created = len(all_chunks)

        result.end_time = datetime.now(timezone.utc)
        logger.info(
            f"Ingestion complete: {result.documents_processed} docs, "
            f"{result.chunks_created} created, {result.chunks_updated} updated, "
            f"{result.chunks_deleted} deleted, {len(result.errors)} errors, "
            f"{result.duration_seconds:.1f}s"
        )

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orchestrator = IngestionOrchestrator()

    # Lazy imports to avoid circular dependency at module level
    from src.retrieval.embeddings import EmbeddingService
    from src.retrieval.vectorstore import VectorStoreService

    embedding_service = EmbeddingService(config_manager.settings.embeddings)
    vs = VectorStoreService(config_manager.settings.vectorstore, embedding_service)

    result = orchestrator.run(vectorstore=vs)
    print(f"Done: {result.documents_processed} docs, {result.chunks_created} chunks created")
