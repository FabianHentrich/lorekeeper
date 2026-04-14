import logging
from dataclasses import dataclass
from typing import Any

from fastapi.concurrency import run_in_threadpool

from src.config.manager import RetrievalConfig
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)


def _rrf_merge(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    bm25_weight: float,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Combine results from vector search and keyword search using Reciprocal Rank Fusion (RRF).

    RRF assigns a score to each document based on its rank in both the vector and BM25 result lists.
    The `bm25_weight` parameter controls the balance between the two methods; a higher value
    gives keyword matches more influence over the final ranking. The constant `k` mitigates the
    impact of extreme outliers by softening the rank decay curve.
    """
    vector_weight = 1.0 - bm25_weight

    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(vector_results):
        doc_id = item["id"]
        scores[doc_id] = vector_weight / (k + rank + 1)
        items[doc_id] = item

    for rank, item in enumerate(bm25_results):
        doc_id = item["id"]
        scores.setdefault(doc_id, 0.0)
        scores[doc_id] += bm25_weight / (k + rank + 1)
        if doc_id not in items:
            items[doc_id] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {**items[doc_id], "score": fused_score}
        for doc_id, fused_score in ranked
    ]


@dataclass
class RetrievedChunk:
    content: str
    source_file: str
    document_type: str
    heading: str
    score: float
    metadata: dict


class Retriever:
    """Orchestrates the semantic retrieval process by combining dense vector
    search with optional BM25 keyword matching and cross-encoder reranking.

    This class is responsible for filtering out unsupported document types (like images), applying
    metadata constraints, executing hybrid search (if enabled), applying configured score
    thresholds, and finally re-ordering the context chunks to maximize relevance and diversity.
    """
    def __init__(
        self,
        config: RetrievalConfig,
        embedding_service: EmbeddingService,
        vectorstore: VectorStoreService,
        bm25_index: BM25Index | None = None,
    ):
        self.config = config
        self.embedding_service = embedding_service
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index or BM25Index()
        self._reranker = None

    def get_reranker(self):
        """Lazy-load and cache the cross-encoder reranking model."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.config.reranking.model}")
            self._reranker = CrossEncoder(self.config.reranking.model, max_length=512)
            logger.info("Reranker loaded")
        return self._reranker

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: dict | None = None,
        top_k_rerank: int | None = None,
        max_per_source: int | None = None,
        hybrid: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Search the vector store for chunks most relevant to the given query.

        If hybrid search is enabled (either globally or via the request flag), it combines dense embeddings
        with BM25 keyword matching via Reciprocal Rank Fusion. It then drops any chunk falling below the
        configured score threshold, and optionally reranks the surviving candidates via a cross-encoder model.
        """
        top_k = top_k or self.config.top_k
        query_embedding = await self.embedding_service.embed_text(query)

        # Build where filter — images are excluded from LLM context (their content
        # is just a filename stub and pollutes ranking). They appear as UI sources
        # only when they happen to be the cited document in a text chunk.
        text_filter: dict = {"document_type": {"$ne": "image"}}
        if metadata_filters:
            conditions = [text_filter] + [{k: v} for k, v in metadata_filters.items()]
            where: dict | None = {"$and": conditions}
        else:
            where = text_filter

        # Resolve hybrid flag: per-request override > config default
        use_hybrid = hybrid if hybrid is not None else self.config.hybrid.enabled

        logger.debug(
            f"Retrieval query: {query!r} | top_k={top_k} | hybrid={use_hybrid} | "
            f"score_threshold={self.config.score_threshold} | where={where}"
        )

        # 1. Vector search (always)
        vector_results = self.vectorstore.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Vectorstore returned {len(vector_results)} candidates (pre-threshold):")
            for i, item in enumerate(vector_results):
                logger.debug(
                    f"  [{i}] score={item['score']:.4f} "
                    f"file={item['metadata'].get('source_file', '?')} "
                    f"heading={item['metadata'].get('heading_hierarchy', '')[:60]}"
                )

        # 2. BM25 search + RRF fusion (if hybrid enabled)
        if use_hybrid:
            if not self.bm25_index.is_built:
                logger.info("Building BM25 index from vectorstore (lazy init)...")
                await run_in_threadpool(
                    self.bm25_index.build_from_vectorstore, self.vectorstore
                )

            bm25_results = self.bm25_index.query(
                query_text=query,
                top_k=self.config.hybrid.bm25_top_k,
                where=where,
            )

            bm25_only = len([r for r in bm25_results if r["id"] not in {v["id"] for v in vector_results}])
            logger.info(
                f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} bm25 "
                f"candidates ({bm25_only} bm25-only)"
            )

            results = _rrf_merge(
                vector_results, bm25_results,
                bm25_weight=self.config.hybrid.bm25_weight,
            )
        else:
            results = vector_results

        chunks = [
            RetrievedChunk(
                content=item["content"],
                source_file=item["metadata"].get("source_file", ""),
                document_type=item["metadata"].get("document_type", ""),
                heading=item["metadata"].get("heading_hierarchy", ""),
                score=item["score"],
                metadata=item["metadata"],
            )
            for item in results
            if item["score"] >= self.config.score_threshold
        ]

        logger.debug(
            f"After score_threshold ({self.config.score_threshold}): "
            f"{len(chunks)}/{len(results)} chunks kept"
        )

        if self.config.reranking.enabled and chunks:
            chunks = await self._rerank(query, chunks, top_k_rerank=top_k_rerank, max_per_source=max_per_source)

        logger.info(f"Retrieved {len(chunks)} chunks (query: {query[:60]}...)")
        return chunks

    async def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k_rerank: int | None = None,
        max_per_source: int | None = None,
    ) -> list[RetrievedChunk]:
        """Apply a cross-encoder model to re-score and re-order the retrieved chunks.

        Unlike the bi-encoder used for initial retrieval, a cross-encoder evaluates the query and chunk text
        simultaneously, yielding a highly accurate relevance score at the cost of higher latency.

        The method also applies a diversity-aware selection strategy: it prefers to select chunks from different
        source files up to a `max_per_source` limit. If the pool of candidates above the initial score threshold
        is small, it will optionally backfill from cap-overflow candidates rather than return fewer chunks than requested.
        """
        reranker = self.get_reranker()
        pairs = [[query, c.content] for c in chunks]

        scores = await run_in_threadpool(reranker.predict, pairs)

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top_k = top_k_rerank or self.config.reranking.top_k_rerank

        # Diversity-aware selection in two passes:
        #
        #   Pass 1 — diverse fill: walk the sorted list and accept chunks
        #     until each source hits `max_per_source`. Cap-blocked chunks are
        #     parked in `overflow` (still in score order).
        #
        #   Pass 2 — backfill: if we did not reach `top_k` after pass 1
        #     (because the post-threshold pool was thin and dominated by one
        #     source), fill the remaining slots from `overflow` in score
        #     order. This makes the cap a *preference* for diversity, not a
        #     hard slot-killer — we never return fewer chunks than we could.
        #
        # Without this, a query like "Was ist Arkenfeld?" can yield 5 chunks
        # instead of top_k_rerank=8, because the cap drops 3 Arkenfeld chunks
        # and there are no other above-threshold candidates to take their place.
        if max_per_source is None:
            max_per_source = self.config.reranking.max_per_source
        per_source_count: dict[str, int] = {}
        selected: list[tuple[float, RetrievedChunk]] = []
        overflow: list[tuple[float, RetrievedChunk]] = []

        for score, chunk in ranked:
            if len(selected) >= top_k:
                break
            if max_per_source and per_source_count.get(chunk.source_file, 0) >= max_per_source:
                overflow.append((score, chunk))
                continue
            per_source_count[chunk.source_file] = per_source_count.get(chunk.source_file, 0) + 1
            selected.append((score, chunk))

        backfilled = 0
        if len(selected) < top_k and overflow:
            for entry in overflow:
                if len(selected) >= top_k:
                    break
                selected.append(entry)
                backfilled += 1

        # Re-sort after backfill so the LLM still sees the most relevant chunks first.
        selected.sort(key=lambda x: x[0], reverse=True)

        reranked = [chunk for _, chunk in selected]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Rerank scores (top {len(selected)} of {len(chunks)}; "
                f"cap-overflow={len(overflow)}, backfilled={backfilled}):"
            )
            for i, (score, chunk) in enumerate(selected):
                logger.debug(
                    f"  [{i}] rerank={score:.4f} "
                    f"bi_score={chunk.score:.4f} "
                    f"file={chunk.source_file} heading={chunk.heading[:60]}"
                )

        logger.info(
            f"Reranked {len(chunks)} → {len(reranked)} chunks "
            f"(cap-overflow={len(overflow)}, backfilled={backfilled}); "
            f"top score: {ranked[0][0]:.3f} ({ranked[0][1].source_file})"
        )
        return reranked
