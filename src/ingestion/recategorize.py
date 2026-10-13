"""Recategorize stored chunks against the current sources config.

Use case: you changed config/sources.yaml (added a category_map entry, changed
group, etc.) and want existing ChromaDB chunks to reflect the new mapping
without paying for a full re-embed. This rewrites only metadata.
"""

import logging

from src.config.manager import ConfigManager, SourceConfig, config_manager
from src.ingestion.orchestrator import _resolve_category, source_base
from src.retrieval.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)


def _match_source(meta: dict, sources: list[SourceConfig]) -> SourceConfig | None:
    """The configured source a stored chunk belongs to, by its ``source_id`` (None if unknown).

    Every chunk written by the orchestrator carries ``source_id``; chunks of a removed or renamed
    source stay unmatched and are counted as such.
    """
    return next((s for s in sources if s.id == meta.get("source_id")), None)


def recategorize(config: ConfigManager | None = None, vectorstore: VectorStoreService | None = None) -> dict:
    """Execute a metadata-only update across the entire vector store to apply new source configurations.

    This function reads all currently stored chunks, recalculates their matching category and group
    based on the current rules in sources.yaml, and pushes the updated metadata back to the vector store.
    This avoids the high cost of re-embedding the text content.
    """
    config = config or config_manager
    if vectorstore is None:
        from src.retrieval.embeddings import EmbeddingService
        embed = EmbeddingService(config.settings.embeddings)
        vectorstore = VectorStoreService(config.settings.vectorstore, embed)

    sources = list(config.settings.ingestion.sources)
    stats = {"chunks_updated": 0, "chunks_skipped": 0, "chunks_unmatched": 0}

    all_data = vectorstore.get_all(include=["metadatas"])
    ids = all_data.get("ids") or []
    metas = all_data.get("metadatas") or []

    new_ids: list[str] = []
    new_metas: list[dict] = []

    for cid, meta in zip(ids, metas, strict=True):
        source = _match_source(meta, sources)
        if source is None:
            stats["chunks_unmatched"] += 1
            continue

        base = source_base(source)
        try:
            file_path = (base / meta.get("source_file", "")).resolve()
        except Exception:
            file_path = base
        new_category, new_group = _resolve_category(file_path, source)

        # Check if the metadata actually changed. If identical, skip the update to save database ops.
        if (meta.get("content_category") == new_category
                and meta.get("group") == new_group
                and meta.get("source_id") == source.id):
            stats["chunks_skipped"] += 1
            continue

        updated = dict(meta)
        updated["content_category"] = new_category
        updated["group"] = new_group
        updated["source_id"] = source.id
        new_ids.append(cid)
        new_metas.append(updated)

    if new_ids:
        vectorstore.update_metadata_batch(new_ids, new_metas)
        stats["chunks_updated"] = len(new_ids)

    logger.info(f"Recategorize complete: {stats}")
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = recategorize()
    print(f"Done: {result}")
