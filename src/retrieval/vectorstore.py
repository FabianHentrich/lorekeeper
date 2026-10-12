import hashlib
import logging
from typing import Any

import chromadb

from src.config.manager import VectorStoreConfig
from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreService:
    """ChromaDB wrapper handling both embedded (PersistentClient) and client (HttpClient) modes.

    Covers the chunk lifecycle — upsert, similarity query, metadata-only update, delete, wipe.
    """
    def __init__(self, config: VectorStoreConfig, embedding_service: EmbeddingService):
        """Retain config + embedding service; the ChromaDB client is connected lazily."""
        self.config = config
        self.embedding_service = embedding_service
        self._client: chromadb.ClientAPI | None = None
        self._collection = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Return the ChromaDB client, connecting on first call.

        HttpClient in client mode, otherwise PersistentClient."""
        if self._client is None:
            if self.config.mode == "client":
                self._client = chromadb.HttpClient(
                    host=self.config.chroma_host,
                    port=self.config.chroma_port,
                )
            else:
                self._client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                )
            logger.info(f"ChromaDB connected (mode={self.config.mode})")
        return self._client

    def _get_collection(self):
        """Return (or create on first call) the configured collection with the right distance metric."""
        if self._collection is None:
            client = self._get_client()
            # Always set the space explicitly — Chroma defaults to l2, which turns
            # ``1 - distance`` into ``2*cos - 1`` and silently tightens score_threshold.
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
            # The space is fixed at creation; an existing collection keeps its old one.
            space = (self._collection.metadata or {}).get("hnsw:space", "l2")
            if space != self.config.distance_metric:
                logger.warning(
                    f"Collection '{self.config.collection_name}' uses distance '{space}' but config "
                    f"expects '{self.config.distance_metric}' — scores are wrong until the collection "
                    "is wiped (POST /admin/wipe) and re-ingested."
                )
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

        # ChromaDB has a batch limit, process in batches of 5000
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
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

    def get_content_hashes_for_source(self, source_id: str) -> dict[str, str]:
        """Return ``{source_file: content_hash}`` for one source, used by ingestion to skip unchanged files."""
        collection = self._get_collection()
        try:
            data = collection.get(where={"source_id": source_id}, include=["metadatas"])
        except Exception:
            return {}

        hashes: dict[str, str] = {}
        if data and data["metadatas"]:
            for meta in data["metadatas"]:
                src = meta.get("source_file", "")
                h = meta.get("content_hash", "")
                if src and h:
                    hashes[src] = h
        return hashes

    def list_files(self, source_id: str) -> list[str]:
        """Sorted indexed files of one source. Unlike ``get_content_hashes_for_source`` a database
        error is raised, not turned into an empty list."""
        data = self._get_collection().get(where={"source_id": source_id}, include=["metadatas"])
        return sorted({m.get("source_file", "") for m in (data.get("metadatas") or []) if m.get("source_file")})

    def get_chunks_for_file(self, source_id: str, source_file: str) -> list[dict[str, Any]]:
        """Return every chunk of one file as ``{id, content, metadata}`` (used for section expansion)."""
        collection = self._get_collection()
        data = collection.get(
            where={"$and": [{"source_id": source_id}, {"source_file": source_file}]},
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
        """Update metadata for existing chunks in batches of 5000, without re-embedding."""
        if not ids:
            return
        collection = self._get_collection()
        batch = 5000
        for i in range(0, len(ids), batch):
            collection.update(ids=ids[i:i+batch], metadatas=metadatas[i:i+batch])
        logger.info(f"Updated metadata for {len(ids)} chunks")

    def wipe_collection(self) -> None:
        """Delete the active collection entirely. The next call recreates it empty."""
        client = self._get_client()
        try:
            client.delete_collection(name=self.config.collection_name)
        except Exception:
            pass
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
