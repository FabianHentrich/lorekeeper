import hashlib
import logging
from typing import Any

import chromadb

from src.config.manager import VectorStoreConfig
from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages the connection and interactions with the ChromaDB vector store.

    Handles both embedded (local SQLite) and client (HTTP container) modes, mapping
    concept chunks (from ingestion) to embedded vectors, performing similarity searches,
    and managing the chunk lifecycle (upserts, metadata updates, wipes, and orphaned file deletions).
    """
    def __init__(self, config: VectorStoreConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self._client: chromadb.ClientAPI | None = None
        self._collection = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Lazily initialize and return the correct ChromaDB client based on configuration.
        Switches between HttpClient and PersistentClient depending on `mode`."""
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
        """Get or create the target collection within ChromaDB context, setting up
        the necessary distance metric (e.g. cosine similarity)."""
        if self._collection is None:
            client = self._get_client()
            metadata = {}
            if self.config.distance_metric != "cosine":
                metadata["hnsw:space"] = self.config.distance_metric
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata=metadata or None,
            )
        return self._collection

    def _chunk_id(self, chunk: Chunk) -> str:
        """Deterministically compute a unique chunk ID derived from the source
        file path and the chunk's sequential slice index."""
        raw = f"{chunk.source_file}::{chunk.chunk_index}"
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
        """Embed and insert/update a batch of processed document chunks into the database.

        The text passed to the embedding model is enriched with file names and aliases
        (_build_embed_text), while the actual raw 'content' is saved for context window delivery.
        It flattens metadata dictionaries since ChromaDB only accepts flat primitives.
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

        for chunk, embedding in zip(chunks, embeddings):
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
        """Perform a semantic similarity search using the embedded query.

        Converts the returned distances into standard similarity scores
        (assuming cosine config: similarity = 1 - distance).
        Matches can be filtered pre-search using Chroma metadata tags via `where`.
        """
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
        """Returns {source_file: content_hash} scoped to a single source_id.

        Used heavily during ingestion to identify if a file has changed. By scoping to source_id,
        it prevents collisions if the user maps multiple sources to visually identical structural folders.
        """
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

    def delete_by_source_file(self, source_id: str, source_file: str) -> None:
        """Delete chunks scoped to (source_id, source_file). Avoids cross-source collisions.
        Raises on DB failure so callers can abort instead of reporting false success."""
        collection = self._get_collection()
        collection.delete(where={"$and": [
            {"source_id": source_id},
            {"source_file": source_file},
        ]})
        logger.info(f"Deleted chunks: source_id={source_id} file={source_file}")

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
        """Update metadata for existing chunks without re-embedding.

        This is a lightweight operation (used for things like mass tag/category updates)
        that modifies database objects without wasting CPU on embedding calls.
        """
        if not ids:
            return
        collection = self._get_collection()
        batch = 5000
        for i in range(0, len(ids), batch):
            collection.update(ids=ids[i:i+batch], metadatas=metadatas[i:i+batch])
        logger.info(f"Updated metadata for {len(ids)} chunks")

    def wipe_collection(self) -> None:
        """Drop and recreate the collection. Works in both embedded and client mode.
        This wipes the entire database context for the active collection."""
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
