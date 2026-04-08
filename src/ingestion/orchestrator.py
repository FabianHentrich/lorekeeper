import fnmatch
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config.manager import ConfigManager, config_manager
from src.ingestion.chunking import Chunk, chunk_documents
from src.ingestion.parsers.base import BaseParser
from src.ingestion.parsers.image import ImageMetaParser
from src.ingestion.parsers.markdown import MarkdownParser
from src.ingestion.parsers.pdf import PDFParser

logger = logging.getLogger(__name__)

# content_category mapping from top-level folder name
CATEGORY_MAPPING = {
    "npcs": "npc",
    "orte": "location",
    "gegner": "enemy",
    "gegenstände": "item",
    "organisationen": "organization",
    "dämonen": "daemon",
    "götter": "god",
    "backstorys": "backstory",
    "spielleiter-tools": "tool",
    "regelwerk": "rules",
    "rules": "rules",
}


def _get_content_category(file_path: Path, base_path: Path) -> str:
    try:
        relative = file_path.relative_to(base_path)
    except ValueError:
        return "misc"

    # If the file lives in a sub-folder, use that folder name as the category
    # key. If the file sits directly in the document_path root (e.g. a single
    # rulebook PDF in `data/rules/`), fall back to the document_path's own
    # name — otherwise root-level files would silently land in "misc".
    if len(relative.parts) > 1:
        top_folder = relative.parts[0].lower()
    else:
        top_folder = base_path.name.lower()

    for prefix, category in CATEGORY_MAPPING.items():
        if top_folder.startswith(prefix):
            return category

    if top_folder.startswith("geschichte"):
        return "story"

    return "misc"


def _compute_content_hash(file_path: Path) -> str:
    content = file_path.read_bytes()
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _is_excluded(file_path: Path, base_path: Path, exclude_patterns: list[str]) -> bool:
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
    def __init__(self):
        self.documents_processed: int = 0
        self.chunks_created: int = 0
        self.chunks_updated: int = 0
        self.chunks_deleted: int = 0
        self.errors: list[str] = []
        self.start_time: datetime = datetime.now(timezone.utc)
        self.end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()


class IngestionOrchestrator:
    def __init__(self, config: ConfigManager | None = None):
        self.config = config or config_manager
        self.parsers: list[BaseParser] = [
            MarkdownParser(),
            PDFParser(),
            ImageMetaParser(),
        ]

    def _get_parser(self, file_path: Path) -> BaseParser | None:
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def _discover_files(self) -> list[tuple[Path, Path]]:
        """Returns list of (file_path, base_path) tuples."""
        settings = self.config.settings
        files = []

        for doc_path_str in settings.ingestion.document_paths:
            base_path = Path(doc_path_str).resolve()
            if not base_path.exists():
                logger.warning(f"Document path does not exist: {base_path}")
                continue

            for ext in settings.ingestion.supported_formats:
                for file_path in base_path.rglob(f"*{ext}"):
                    if not _is_excluded(file_path, base_path, settings.ingestion.exclude_patterns):
                        files.append((file_path, base_path))

        return files

    def run(self, vectorstore=None) -> IngestionResult:
        result = IngestionResult()
        settings = self.config.settings

        files = self._discover_files()
        logger.info(f"Discovered {len(files)} files for ingestion")

        # Track which source files we process (for orphan detection)
        processed_sources = set()

        all_chunks: list[Chunk] = []

        for file_path, base_path in files:
            parser = self._get_parser(file_path)
            if parser is None:
                continue

            try:
                documents = parser.parse(file_path, base_path)
                chunks = chunk_documents(documents, settings.chunking)

                content_hash = _compute_content_hash(file_path)
                content_category = _get_content_category(file_path, base_path)
                source_collection = base_path.name
                last_modified = datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()

                relative_path = str(file_path.relative_to(base_path))

                for chunk in chunks:
                    chunk.source_file = relative_path
                    chunk.metadata.update({
                        "content_hash": content_hash,
                        "content_category": content_category,
                        "source_collection": source_collection,
                        "last_modified": last_modified,
                    })

                all_chunks.extend(chunks)
                processed_sources.add(relative_path)
                result.documents_processed += 1

            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        if vectorstore is not None:
            existing_hashes = vectorstore.get_all_content_hashes()

            # Determine which chunks are new/updated
            for chunk in all_chunks:
                chunk_hash = chunk.metadata.get("content_hash", "")
                source = chunk.source_file
                existing = existing_hashes.get(source)

                if existing and existing == chunk_hash:
                    # Unchanged — skip
                    continue
                elif existing:
                    result.chunks_updated += 1
                else:
                    result.chunks_created += 1

            # Delete chunks for files that no longer exist
            orphaned = set(existing_hashes.keys()) - processed_sources
            for orphan in orphaned:
                vectorstore.delete_by_source(orphan)
                result.chunks_deleted += 1

            # Upsert all chunks (vectorstore handles dedup by ID)
            vectorstore.upsert_chunks(all_chunks)
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
