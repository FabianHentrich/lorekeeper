"""ChromaDB access — embedded ``PersistentClient``, only ever opened by one process.

Stored document = the chunk text the LLM sees; embedded text = filename stem + aliases + chunk text
(``_build_embed_text``). Chunk ids are stable per (source_id, file, chunk index), so re-ingesting a file
overwrites its chunks. Scores are ``1 - distance`` — only right for cosine, so the collection's space is set
explicitly and a mismatch with an existing collection is logged.
"""
import hashlib
import logging
import re
from typing import Any

import chromadb
import numpy as np
from chromadb.errors import NotFoundError

from src.config.manager import VectorStoreConfig
from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# Chunks per upsert/update call — ChromaDB rejects batches above the client's max_batch_size
_CHROMA_BATCH = 5000
# Cosine scores closer than this count as tied (float32 noise) in the exactness check
_TIE_EPS = 1e-6
_TABLE_SEPARATOR = re.compile(r"^\|?(\s*:?-{2,}:?\s*\|)+\s*:?-*:?\s*\|?$")


def _table_row_names(content: str) -> list[str]:
    """First cell of every body row of the Markdown tables in ``content`` (header and separator rows skipped).

    Rows with an empty first cell are continuations of the row above (PDF tables wrap long cells) and are
    left out. ``**``/``<br>`` are removed so the names read as plain words.
    """
    names: list[str] = []
    header_pending = True
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            header_pending = True   # a new table starts with its own header row
            continue
        if _TABLE_SEPARATOR.match(stripped):
            continue
        if header_pending:
            header_pending = False
            continue
        first = stripped[1:].split("|", 1)[0]
        first = re.sub(r"<br\s*/?>", " ", first).replace("**", "").strip()
        if first:
            names.append(first)
    return names


class VectorStoreService:
    """ChromaDB wrapper — embedded (``PersistentClient``) inside the backend process.

    No second process may open ``persist_directory`` while the backend runs (file locks).

    Covers the chunk lifecycle — upsert, similarity query, metadata-only update, delete, wipe.
    """
    def __init__(self, config: VectorStoreConfig, embedding_service: EmbeddingService):
        """Retain config + embedding service; the ChromaDB client is connected lazily."""
        self.config = config
        self.embedding_service = embedding_service
        self._client: chromadb.ClientAPI | None = None
        self._collection = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Return the embedded ChromaDB client, connecting on first call."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.config.persist_directory)
            logger.info(f"ChromaDB connected (embedded, {self.config.persist_directory})")
        return self._client

    def _get_collection(self):
        """Return (or create on first call) the configured collection with the right distance metric
        and search breadth.

        Everything goes through ``configuration`` — combined with the old ``metadata={"hnsw:space": …}``
        Chroma 1.5 silently created an **l2** collection. The space is fixed at creation (a mismatch is
        logged with the fix); ``ef_search`` can change and is applied to an existing collection here,
        because ``get_or_create_collection`` ignores the configuration of a collection that exists.
        """
        if self._collection is None:
            client = self._get_client()
            # Always set the space explicitly — Chroma defaults to l2, which turns
            # ``1 - distance`` into ``2*cos - 1`` and silently tightens score_threshold.
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                configuration={"hnsw": {"space": self.config.distance_metric,
                                        "ef_search": self.config.hnsw_ef_search}},
            )
            hnsw = (self._collection.configuration_json or {}).get("hnsw") or {}
            space = hnsw.get("space", "l2")
            if space != self.config.distance_metric:
                logger.warning(
                    f"Collection '{self.config.collection_name}' uses distance '{space}' but config "
                    f"expects '{self.config.distance_metric}' — scores are wrong until the collection "
                    "is wiped (POST /admin/wipe) and re-ingested."
                )
            if hnsw.get("ef_search") != self.config.hnsw_ef_search:
                self._collection.modify(configuration={"hnsw": {"ef_search": self.config.hnsw_ef_search}})
                logger.info("Collection '%s': hnsw ef_search %s → %s", self.config.collection_name,
                            hnsw.get("ef_search"), self.config.hnsw_ef_search)
        return self._collection

    def _chunk_id(self, chunk: Chunk) -> str:
        """Stable MD5 ID from ``source_id::source_file::chunk_index``.

        ``source_id`` keeps two sources with the same relative path (e.g. ``Index.md``) from overwriting each other.
        """
        raw = f"{chunk.metadata.get('source_id', '')}::{chunk.source_file}::{chunk.chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _build_embed_text(chunk: Chunk) -> str:
        """Construct the text that gets embedded.

        Strategy:
          1. Filename stem + aliases  →  document identity (who/what is this about)
          2. Heading path             →  section context (already in chunk.content after chunking fix)
          3. Content                  →  actual text

        Aliases and filename stem are prepended so the embedding vector captures
        the entity name even when the content itself never repeats it (e.g. a
        "Steckbrief" table that describes Arkenfeld without saying "Arkenfeld").

        No explicit weighting — sentence-transformers has none. Prominence is
        achieved by position (beginning of text) and repetition.

        Table chunks get a line with their row names after the identity ("Einträge: Dolch, Kurzschwert,
        Langschwert, …"): a table is one vector, the average of all rows, so a single row ("Wie viel Schaden
        macht ein Langschwert?") matched it weakly. Golden set 13.10. (roadmap 3.10): weapon questions
        p086/p087 solved, vector rank 22 → 14 and 47 → 3, nothing lost; rewriting the table as plain text
        was worse.
        """
        parts = []

        # 1. Identity line: filename stem + aliases from metadata
        stem = chunk.source_file.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
        aliases_raw = chunk.metadata.get("aliases", "")
        if isinstance(aliases_raw, list):
            aliases = [a for a in aliases_raw if a and a != stem]
        elif aliases_raw:
            aliases = [a.strip() for a in aliases_raw.split(",") if a.strip() and a.strip() != stem]
        else:
            aliases = []

        identity = stem
        if aliases:
            identity += " | " + " | ".join(aliases)
        parts.append(identity)

        # 1b. Row names of tables — makes single rows findable (see docstring)
        if names := _table_row_names(chunk.content):
            parts.append("Einträge: " + ", ".join(names))

        # 2+3. Heading + content (already prepended in chunk.content by _doc_to_chunk)
        parts.append(chunk.content)

        return "\n\n".join(parts)

    def upsert_chunks(self, chunks: list[Chunk]):
        """Embed the enriched text of each chunk and insert/update it in the collection.

        The raw chunk content is stored as the document body (what the LLM later sees),
        while ``_build_embed_text`` augments the *embedded* text with filename and aliases.
        Chunk metadata is flattened — ChromaDB only accepts str/int/float/bool values.
        """
        if not chunks:
            return

        collection = self._get_collection()

        # Embed enriched text; store original content for LLM context
        texts = [self._build_embed_text(c) for c in chunks]
        embeddings = self.embedding_service.embed_texts_sync(texts)

        ids = []
        documents = []
        metadatas = []
        embedding_list = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_id = self._chunk_id(chunk)
            ids.append(chunk_id)
            documents.append(chunk.content)
            embedding_list.append(embedding)

            meta = {
                "source_file": chunk.source_file,
                "source_path": chunk.source_path,
                "document_type": chunk.document_type,
                "heading_hierarchy": " > ".join(chunk.heading_hierarchy),
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            }
            # Flatten chunk metadata (ChromaDB only supports str/int/float/bool)
            for k, v in chunk.metadata.items():
                if isinstance(v, list):
                    meta[k] = ", ".join(str(item) for item in v)
                elif isinstance(v, (str, int, float, bool)):
                    meta[k] = v

            metadatas.append(meta)

        for i in range(0, len(ids), _CHROMA_BATCH):
            end = i + _CHROMA_BATCH
            collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embedding_list[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(f"Upserted {len(ids)} chunks to ChromaDB")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Similarity search; returns hits with ``score = 1 - distance`` (assumes cosine)."""
        collection = self._get_collection()

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = collection.query(**query_params)

        items = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances; for cosine, similarity = 1 - distance
                distance = results["distances"][0][i]
                score = 1.0 - distance

                items.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score,
                })

        return items

    def get_table_entries(self, names: list[str], query_embedding: list[float], where: dict | None = None,
                          limit: int = 3) -> list[dict[str, Any]]:
        """Table chunks with a row whose first cell is one of ``names`` (entry slot, roadmap 3.10).

        A table is one vector — the average of all its rows — so "Wie viel Schaden macht ein Dolch-Angriff?"
        misses the weapon table although "Dolch" is a row of it. Matches the bold cell ``|**Name**|`` in the
        stored text, then keeps only chunks where the name really is a row's first cell (so "Dolch" doesn't
        match "Dolch der Schatten"). Chunks naming more of ``names`` come first. ``score`` is the real cosine
        similarity to ``query_embedding``, so these hits read like any other. ``where`` is AND-ed in — the
        request's filter (player restriction included) applies like on every other search path.
        """
        if not names or limit <= 0:
            return []
        patterns: list[dict[str, Any]] = [{"$contains": f"|**{n}**|"} for n in names]
        params: dict[str, Any] = {
            "where_document": patterns[0] if len(patterns) == 1 else {"$or": patterns},
            "include": ["documents", "metadatas", "embeddings"],
        }
        if where:
            params["where"] = where
        data = self._get_collection().get(**params)

        wanted = set(names)
        q = np.asarray(query_embedding, dtype=np.float32)
        hits = []
        for i, doc_id in enumerate(data.get("ids") or []):
            content = data["documents"][i]
            matched = wanted & set(_table_row_names(content))
            if not matched:
                continue
            e = np.asarray(data["embeddings"][i], dtype=np.float32)
            score = float(q @ e / (np.linalg.norm(q) * np.linalg.norm(e)))
            hits.append({"id": doc_id, "content": content, "metadata": data["metadatas"][i], "score": score,
                         "_matched": len(matched)})
        hits.sort(key=lambda h: (h["_matched"], h["score"]), reverse=True)
        return [{k: v for k, v in h.items() if k != "_matched"} for h in hits[:limit]]

    def get_content_hashes_for_source(self, source_id: str) -> dict[str, str]:
        """Return ``{source_file: content_hash}`` for one source, used by ingestion to skip unchanged files.

        A database error is raised, not turned into ``{}``: an empty result would make the ingestion treat
        every file as new — old chunks of changed files and orphaned chunks would silently stay.
        """
        data = self._get_collection().get(where={"source_id": source_id}, include=["metadatas"])

        hashes: dict[str, str] = {}
        if data and data["metadatas"]:
            for meta in data["metadatas"]:
                src = meta.get("source_file", "")
                h = meta.get("content_hash", "")
                if src and h:
                    hashes[src] = h
        return hashes

    def list_files(self, source_id: str) -> list[str]:
        """Sorted indexed files of one source (chunk inspector). Every ingested chunk carries a
        ``content_hash``, so the hash lookup lists them all; database errors are raised."""
        return sorted(self.get_content_hashes_for_source(source_id))

    def list_source_ids(self) -> set[str]:
        """Every ``source_id`` that has chunks in the index — the ingest compares it with sources.yaml."""
        data = self._get_collection().get(include=["metadatas"])
        return {sid for m in data.get("metadatas") or [] if (sid := (m or {}).get("source_id"))}

    def get_all(self, include: list[str]) -> dict[str, Any]:
        """Every stored chunk as ChromaDB's column dict (``ids`` plus the ``include`` fields) —
        for the full scans of the BM25 build and recategorize."""
        return self._get_collection().get(include=include)

    def compare_with_exact_search(self, query_embeddings: list[list[float]], top_k: int,
                                  where: dict | None) -> list[int]:
        """For each query: how many of the exact top-``top_k`` chunks the HNSW search misses (0 = exact).

        The exact side is a brute-force cosine over every stored vector that passes ``where`` (ChromaDB
        applies the filter for both sides). Used by ``/eval/search-exactness`` to show whether
        ``hnsw_ef_search`` is still large enough for the index size (roadmap 5.5). Reads only.

        Only top-K chunks scoring clearly above the best chunk outside the top K must be found: chunks
        tied across the cut-off (equal embeddings, e.g. the same note in two sources) are interchangeable —
        counting them reported deviations that no ``ef_search`` could fix.
        """
        collection = self._get_collection()
        hnsw = collection.query(query_embeddings=query_embeddings, n_results=top_k, where=where, include=[])
        data = collection.get(where=where, include=["embeddings"])
        ids = data.get("ids") or []
        if not ids:
            return [0] * len(query_embeddings)
        stored = np.asarray(data["embeddings"], dtype=np.float32)
        stored /= np.linalg.norm(stored, axis=1, keepdims=True)
        queries = np.asarray(query_embeddings, dtype=np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        sims = queries @ stored.T
        missed = []
        for row, found in zip(sims, hnsw["ids"], strict=True):
            order = np.argsort(-row)
            # Score of the best chunk just outside the top K; a member tied with it may be swapped for it
            outside = row[order[top_k]] if len(order) > top_k else -np.inf
            must = {ids[i] for i in order[:top_k] if row[i] > outside + _TIE_EPS}
            missed.append(len(must - set(found)))
        return missed

    def get_chunks_for_file(self, source_id: str, source_file: str,
                            where: dict | None = None) -> list[dict[str, Any]]:
        """Return every chunk of one file as ``{id, content, metadata}`` (section expansion, chunk inspector).

        ``where`` is AND-ed in: section expansion passes the request's filter, so it can never load more
        than the search itself may see — today the player filter is per source and expansion stays in
        the hit's source, but a finer filter (or a new expansion type copying this) must not leak.
        """
        collection = self._get_collection()
        conditions: list[dict] = [{"source_id": source_id}, {"source_file": source_file}]
        if where:
            conditions.append(where)
        data = collection.get(
            where={"$and": conditions},
            include=["documents", "metadatas"],
        )
        return [
            {"id": doc_id, "content": data["documents"][i], "metadata": data["metadatas"][i]}
            for i, doc_id in enumerate(data.get("ids") or [])
        ]

    def delete_by_source_file(self, source_id: str, source_file: str) -> int:
        """Delete chunks scoped to (source_id, source_file). Avoids cross-source collisions.
        Returns count before deletion. Raises on DB failure so callers can abort instead
        of reporting false success."""
        collection = self._get_collection()
        where = {"$and": [
            {"source_id": source_id},
            {"source_file": source_file},
        ]}
        existing = collection.get(where=where, include=[])
        count = len(existing["ids"]) if existing and existing.get("ids") else 0
        if count:
            collection.delete(where=where)
        logger.info(f"Deleted {count} chunks: source_id={source_id} file={source_file}")
        return count

    def delete_by_source_id(self, source_id: str) -> int:
        """Delete every chunk that belongs to a source. Returns count before deletion.
        Raises on DB failure."""
        collection = self._get_collection()
        existing = collection.get(where={"source_id": source_id}, include=[])
        count = len(existing["ids"]) if existing and existing.get("ids") else 0
        if count:
            collection.delete(where={"source_id": source_id})
        logger.info(f"Deleted {count} chunks for source_id={source_id}")
        return count

    def update_metadata_batch(self, ids: list[str], metadatas: list[dict]) -> None:
        """Update metadata for existing chunks in batches, without re-embedding."""
        if not ids:
            return
        collection = self._get_collection()
        for i in range(0, len(ids), _CHROMA_BATCH):
            collection.update(ids=ids[i:i + _CHROMA_BATCH], metadatas=metadatas[i:i + _CHROMA_BATCH])
        logger.info(f"Updated metadata for {len(ids)} chunks")

    def wipe_collection(self) -> None:
        """Delete the active collection entirely. The next call recreates it empty."""
        client = self._get_client()
        try:
            client.delete_collection(name=self.config.collection_name)
        except NotFoundError:
            pass  # nothing to wipe — any other error must reach the caller, not end in "Wiped"
        self._collection = None
        logger.warning(f"Wiped collection: {self.config.collection_name}")

    def count(self) -> int:
        """Return the total number of chunks currently stored in the active collection."""
        collection = self._get_collection()
        return collection.count()

    def health_check(self) -> bool:
        """Ping the ChromaDB instance to verify connectivity and readiness."""
        try:
            self._get_client().heartbeat()
            return True
        except Exception:
            return False
