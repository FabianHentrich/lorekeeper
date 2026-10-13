"""Retrieval: embed → vector search (+ BM25 and RRF) → score threshold → rerank with per-file cap and
rules slot → section expansion.

Player-mode safety: every extra search (rules slot) keeps the request's ``where`` filter, and expansion only
loads chunks of files that already passed it (tests/integration/test_player_mode_security.py). Image chunks
are excluded (filename stubs). The score threshold applies to cosine scores before fusion. The cross-encoder
runs on the GPU via DirectML when configured and drops to the CPU for the rest of the run if the GPU fails.
Design decisions and measurements: docs/embedding-strategy.md.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from fastapi.concurrency import run_in_threadpool

from src.config.manager import RetrievalConfig
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)

RULES_GROUP = "rules"
_RERANK_MAX_TOKENS = 512  # cross-encoder input limit (query + chunk); longer pairs are truncated
_SLOT_MIN_PLACES = 3  # with fewer final chunks one rules chunk would push out most of the answer's context


def _to_chunk(item: dict[str, Any]) -> "RetrievedChunk":
    """Vector/BM25 hit dict → RetrievedChunk."""
    return RetrievedChunk(
        content=item["content"],
        source_file=item["metadata"].get("source_file", ""),
        document_type=item["metadata"].get("document_type", ""),
        heading=item["metadata"].get("heading_hierarchy", ""),
        score=item["score"],
        metadata=item["metadata"],
    )


def _rrf_merge(
    vector_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    bm25_weight: float,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse vector and BM25 hit lists via Reciprocal Rank Fusion.

    Each document's fused score sums ``weight / (k + rank + 1)`` contributions from
    the two lists. ``bm25_weight`` (0..1) tilts the balance; ``k`` dampens the
    advantage of the top ranks so lower-ranked but commonly-retrieved docs can
    still surface.
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


def _section_key(chunk) -> tuple[str, str, str] | None:
    """(source_id, source_file, parent heading) — None for chunks without a parent heading."""
    parts = chunk.heading.split(" > ") if chunk.heading else []
    if len(parts) < 2:
        return None
    return (chunk.metadata.get("source_id", ""), chunk.source_file, " > ".join(parts[:-1]))


@dataclass
class RetrievedChunk:
    """A hit as the API and the prompt use it; ``heading`` is the joined heading path ("A > B")."""
    content: str
    source_file: str
    document_type: str
    heading: str
    score: float
    metadata: dict


class Retriever:
    """Runs vector search (optionally fused with BM25 via RRF) and an optional cross-encoder rerank."""
    def __init__(
        self,
        config: RetrievalConfig,
        embedding_service: EmbeddingService,
        vectorstore: VectorStoreService,
        bm25_index: BM25Index | None = None,
    ):
        """Wire up the embedding, vector, and BM25 dependencies; reranker is lazy-loaded."""
        self.config = config
        self.embedding_service = embedding_service
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index or BM25Index()
        self._reranker: Any = None  # CrossEncoder (CPU) or DirectMLCrossEncoder (GPU), loaded lazily
        self._reranker_on_gpu = False
        self._fallback_lock = asyncio.Lock()  # one GPU→CPU switch even if several queries fail at once

    def get_reranker(self):
        """Lazy-load and cache the cross-encoder reranking model.

        ``reranking.backend: directml`` runs it on the GPU; if DirectML can't run it, a warning
        is logged and the torch model on the CPU is used — same scores, only slower.
        """
        if self._reranker is None:
            model = self.config.reranking.model
            if self.config.reranking.backend == "directml":
                from src.retrieval.onnx_reranker import DirectMLCrossEncoder
                try:
                    self._reranker = DirectMLCrossEncoder(model, max_length=_RERANK_MAX_TOKENS)
                    self._reranker_on_gpu = True
                    logger.info(f"Reranker loaded on the GPU (DirectML): {model}")
                except Exception as e:
                    logger.warning(f"DirectML reranker unavailable ({e}) — falling back to the CPU (torch)")
            if self._reranker is None:
                self._reranker = self._load_cpu_reranker()
        return self._reranker

    def _load_cpu_reranker(self):
        """The torch cross-encoder on the CPU — default backend and the fallback for DirectML."""
        from sentence_transformers import CrossEncoder
        model = self.config.reranking.model
        logger.info(f"Loading reranker model: {model}")
        reranker = CrossEncoder(model, max_length=_RERANK_MAX_TOKENS)
        logger.info("Reranker loaded (CPU, torch)")
        return reranker

    async def _predict(self, pairs: list[list[str]]) -> list[float]:
        """Score the pairs; if the GPU fails at runtime (driver reset, out of VRAM), switch to the CPU
        model for the rest of the run instead of failing every query until a restart.

        Whether to fall back is decided by the model *this* call used, not the shared flag: a second
        query still running on the GPU when the first one already switched retries on the CPU too.
        The lock makes concurrent failures load the CPU model once.
        """
        reranker = await run_in_threadpool(self.get_reranker)  # first load may take seconds — not on the event loop
        used_gpu = self._reranker_on_gpu and reranker is self._reranker
        try:
            return list(await run_in_threadpool(reranker.predict, pairs))
        except Exception as e:
            if not used_gpu:
                raise
            async with self._fallback_lock:
                if self._reranker is reranker:   # nobody switched yet
                    logger.warning(f"DirectML reranker failed ({e}) — switching to the CPU (torch) for this run")
                    self._reranker = await run_in_threadpool(self._load_cpu_reranker)
                    self._reranker_on_gpu = False
                cpu = self._reranker
            return list(await run_in_threadpool(cpu.predict, pairs))

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: dict | None = None,
        top_k_rerank: int | None = None,
        max_per_source: int | None = None,
        hybrid: bool | None = None,
        timings: dict[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        """Search the vector store for chunks most relevant to the given query.

        Vector hits below the score threshold are dropped first; with hybrid search (config or request flag)
        BM25 hits are fused in via Reciprocal Rank Fusion. The rerank picks ``top_k_rerank`` chunks with the
        per-file cap and the rules slot, then section expansion may replace fragments with their whole
        section. Parameters left at None take the config value. ``metadata_filters`` is applied to every
        search path. When ``timings`` is given, per-stage durations in ms are written into it (embed,
        search, bm25, rerank, expand).
        """
        stage_ms = timings if timings is not None else {}
        top_k = top_k or self.config.top_k
        t0 = time.perf_counter()
        query_embedding = await self.embedding_service.embed_text(query)
        stage_ms["embed"] = (time.perf_counter() - t0) * 1000

        # Build where filter — image chunks are excluded entirely: their content is just a
        # filename stub that pollutes ranking. So no image ever becomes a source today; real
        # image descriptions and their retrieval are roadmap item 2.
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
        t0 = time.perf_counter()
        vector_results = await run_in_threadpool(
            self.vectorstore.query,
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )
        stage_ms["search"] = (time.perf_counter() - t0) * 1000

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Vectorstore returned {len(vector_results)} candidates (pre-threshold):")
            for i, item in enumerate(vector_results):
                logger.debug(
                    f"  [{i}] score={item['score']:.4f} "
                    f"file={item['metadata'].get('source_file', '?')} "
                    f"heading={item['metadata'].get('heading_hierarchy', '')[:60]}"
                )

        # Apply cosine score_threshold to vector hits up front. RRF fuses ranks,
        # not absolute scores, so we cannot threshold the fused list (RRF scores
        # max out around 1/(60+1) ≈ 0.016 and would be wiped by any cosine-scale
        # threshold). BM25 scores live on their own scale and are not thresholded
        # here — we trust bm25_top_k to cap noise on the keyword side.
        vector_results = [
            r for r in vector_results
            if r["score"] >= self.config.score_threshold
        ]

        # 1b. Rules slot candidates: a separate search inside the rulebook, same filters (player mode
        # included). Kept apart from the main pool — at most one of them enters the result (see _rerank).
        slot = self.config.rules_slot
        rules_extra: list[dict[str, Any]] = []
        final_k = top_k_rerank or self.config.reranking.top_k_rerank
        if slot.enabled and slot.candidates and self.config.reranking.enabled and final_k >= _SLOT_MIN_PLACES:
            rules_hits = await run_in_threadpool(
                self.vectorstore.query,
                query_embedding=query_embedding,
                top_k=slot.candidates,
                where={"$and": [where, {"group": RULES_GROUP}]},
            )
            known = {r["id"] for r in vector_results}
            rules_extra = [r for r in rules_hits if r["id"] not in known and r["score"] >= self.config.score_threshold]

        # 2. BM25 search + RRF fusion (if hybrid enabled)
        if use_hybrid:
            t0 = time.perf_counter()
            if not self.bm25_index.is_built:
                logger.info("Building BM25 index from vectorstore (lazy init)...")
                await run_in_threadpool(
                    self.bm25_index.build_from_vectorstore, self.vectorstore
                )

            bm25_results = await run_in_threadpool(
                self.bm25_index.query,
                query_text=query,
                top_k=self.config.hybrid.bm25_top_k,
                where=where,
            )

            vector_ids = {v["id"] for v in vector_results}
            bm25_only = sum(1 for r in bm25_results if r["id"] not in vector_ids)
            logger.info(
                f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} bm25 "
                f"candidates ({bm25_only} bm25-only)"
            )

            results = _rrf_merge(
                vector_results, bm25_results,
                bm25_weight=self.config.hybrid.bm25_weight,
            )
            stage_ms["bm25"] = (time.perf_counter() - t0) * 1000
        else:
            results = vector_results

        chunks = [_to_chunk(item) for item in results]

        if self.config.reranking.enabled and chunks:
            t0 = time.perf_counter()
            chunks = await self._rerank(query, chunks, top_k_rerank=top_k_rerank, max_per_source=max_per_source,
                                        rules_candidates=[_to_chunk(item) for item in rules_extra])
            stage_ms["rerank"] = (time.perf_counter() - t0) * 1000

        if self.config.section_expansion.enabled and chunks:
            t0 = time.perf_counter()
            chunks = await self._expand_sections(chunks)
            stage_ms["expand"] = (time.perf_counter() - t0) * 1000

        logger.info(f"Retrieved {len(chunks)} chunks (query: {query[:60]}...)")
        return chunks

    async def _expand_sections(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Replace sibling fragments with their complete parent section.

        Questions like "which abilities does the ranger have?" match a few of the
        ~20 sub-heading chunks of one section; the LLM then answers from fragments.
        When >= 2 hits of one file share a parent heading with *different* child
        headings, every chunk under that parent is loaded and inserted in document
        order at the position of the best hit. Sections above ``max_chunks`` stay as is.

        Only sections with a hit among the first ``top_rank`` chunks are expanded —
        two stray mid-ranking hits in an unrelated section (e.g. the Krieger section
        for "was ist überschaden?") would otherwise flood the context with ~20 chunks.
        """
        top_rank = self.config.section_expansion.top_rank
        siblings: dict[tuple[str, str, str], set[str]] = {}
        best_rank: dict[tuple[str, str, str], int] = {}
        for rank, c in enumerate(chunks):
            key = _section_key(c)
            if key is not None:
                siblings.setdefault(key, set()).add(c.heading)
                best_rank.setdefault(key, rank)

        expanded: dict[tuple[str, str, str], list[RetrievedChunk]] = {}
        for (source_id, source_file, parent), headings in siblings.items():
            if len(headings) < 2:
                continue
            if top_rank and best_rank[(source_id, source_file, parent)] >= top_rank:
                logger.info(
                    f"Section '{parent}' ({source_file}) not expanded — best hit at rank "
                    f"{best_rank[(source_id, source_file, parent)] + 1} > {top_rank}"
                )
                continue
            items = await run_in_threadpool(self.vectorstore.get_chunks_for_file, source_id, source_file)
            section = [
                it for it in items
                if (h := it["metadata"].get("heading_hierarchy", "")) == parent or h.startswith(parent + " > ")
            ]
            if len(section) > self.config.section_expansion.max_chunks:
                logger.info(f"Section '{parent}' ({source_file}) has {len(section)} chunks — not expanded")
                continue
            section.sort(key=lambda it: it["metadata"].get("chunk_index", 0))
            score = max(c.score for c in chunks if c.source_file == source_file and c.heading in headings)
            # The section's chunks inherit the score of its best hit
            expanded[(source_id, source_file, parent)] = [_to_chunk({**it, "score": score}) for it in section]
            logger.info(f"Expanded section '{parent}' ({source_file}): {len(headings)} hits → {len(section)} chunks")

        if not expanded:
            return chunks

        # The first (best-ranked) hit of a section inserts the whole section; dedupe by
        # chunk identity so siblings and deeper hits already inside it aren't repeated.
        result: list[RetrievedChunk] = []
        seen: set = set()
        for c in chunks:
            key = _section_key(c)
            for item in (expanded.get(key, [c]) if key is not None else [c]):
                idx = item.metadata.get("chunk_index")
                # source_id too: two sources may contain the same relative file name
                ident = (item.metadata.get("source_id"), item.source_file, idx) if idx is not None else id(item)
                if ident not in seen:
                    seen.add(ident)
                    result.append(item)
        return result

    async def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k_rerank: int | None = None,
        max_per_source: int | None = None,
        rules_candidates: list[RetrievedChunk] | None = None,
    ) -> list[RetrievedChunk]:
        """Apply a cross-encoder model to re-score and re-order the retrieved chunks.

        Unlike the bi-encoder used for initial retrieval, a cross-encoder evaluates the query and chunk text
        simultaneously, yielding a highly accurate relevance score at the cost of higher latency.

        The method also applies a diversity-aware selection strategy: it prefers to select chunks from different
        source files up to a `max_per_source` limit. If the pool of candidates above the initial score threshold
        is small, it will optionally backfill from cap-overflow candidates rather than return fewer chunks than
        requested.
        """
        extra = rules_candidates or []
        pairs = [[query, c.content] for c in chunks + extra]

        all_scores = await self._predict(pairs)
        scores, extra_scores = all_scores[:len(chunks)], all_scores[len(chunks):]

        ranked = sorted(zip(scores, chunks, strict=True), key=lambda x: x[0], reverse=True)
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
        # Keyed by (source_id, file): equally named files in two sources get their own cap
        per_source_count: dict[tuple[str, str], int] = {}
        selected: list[tuple[float, RetrievedChunk]] = []
        overflow: list[tuple[float, RetrievedChunk]] = []

        for score, chunk in ranked:
            if len(selected) >= top_k:
                break
            key = (chunk.metadata.get("source_id", ""), chunk.source_file)
            if max_per_source and per_source_count.get(key, 0) >= max_per_source:
                overflow.append((score, chunk))
                continue
            per_source_count[key] = per_source_count.get(key, 0) + 1
            selected.append((score, chunk))

        fill = overflow[:max(0, top_k - len(selected))]
        selected += fill
        backfilled = len(fill)

        # Rules slot: no rulebook chunk made it — the best one (main pool or the separate rules
        # search) takes the last place if it scores ≥ min_score. Never more than this one chunk.
        slot = self.config.rules_slot
        slot_used = False
        if (slot.enabled and top_k >= _SLOT_MIN_PLACES
                and not any(c.metadata.get("group") == RULES_GROUP for _, c in selected)):
            chosen = {id(c) for _, c in selected}
            pool = [(s, c) for s, c in ranked if c.metadata.get("group") == RULES_GROUP and id(c) not in chosen]
            pool += list(zip(extra_scores, extra, strict=True))
            best_rule = max(pool, key=lambda x: x[0], default=None)
            if best_rule and best_rule[0] >= slot.min_score:
                if len(selected) >= top_k:
                    selected.sort(key=lambda x: x[0], reverse=True)
                    selected.pop()
                selected.append(best_rule)
                slot_used = True

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
            f"(cap-overflow={len(overflow)}, backfilled={backfilled}{', rules slot' if slot_used else ''}); "
            f"top score: {ranked[0][0]:.3f} ({ranked[0][1].source_file})"
        )
        return reranked
