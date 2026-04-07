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
            chunks = await self._rerank(query, chunks)

        logger.info(f"Retrieved {len(chunks)} chunks (query: {query[:60]}...)")
        return chunks

    async def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        reranker = self._get_reranker()
        pairs = [[query, c.content] for c in chunks]

        scores = await run_in_threadpool(reranker.predict, pairs)

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top_k = self.config.reranking.top_k_rerank
        reranked = [chunk for _, chunk in ranked[:top_k]]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Rerank scores (top {top_k} of {len(chunks)}):")
            for i, (score, chunk) in enumerate(ranked[:top_k]):
                logger.debug(
                    f"  [{i}] rerank={score:.4f} "
                    f"bi_score={chunk.score:.4f} "
                    f"file={chunk.source_file} heading={chunk.heading[:60]}"
                )

        logger.info(
            f"Reranked {len(chunks)} → {len(reranked)} chunks; "
            f"top score: {ranked[0][0]:.3f} ({ranked[0][1].source_file})"
        )
        return reranked
