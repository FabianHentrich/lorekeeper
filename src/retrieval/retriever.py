import logging
from dataclasses import dataclass

from fastapi.concurrency import run_in_threadpool

from src.config.manager import RetrievalConfig
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    content: str
    source_file: str
    document_type: str
    heading: str
    score: float
    metadata: dict


class Retriever:
    def __init__(
        self,
        config: RetrievalConfig,
        embedding_service: EmbeddingService,
        vectorstore: VectorStoreService,
    ):
        self.config = config
        self.embedding_service = embedding_service
        self.vectorstore = vectorstore
        self._reranker = None

    def _get_reranker(self):
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
    ) -> list[RetrievedChunk]:
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

        logger.debug(
            f"Retrieval query: {query!r} | top_k={top_k} | "
            f"score_threshold={self.config.score_threshold} | where={where}"
        )

        results = self.vectorstore.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Vectorstore returned {len(results)} candidates (pre-threshold):")
            for i, item in enumerate(results):
                logger.debug(
                    f"  [{i}] score={item['score']:.4f} "
                    f"file={item['metadata'].get('source_file', '?')} "
                    f"heading={item['metadata'].get('heading_hierarchy', '')[:60]}"
                )

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
        reranker = self._get_reranker()
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
