from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.manager import HybridSearchConfig, RerankingConfig, RetrievalConfig
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import Retriever, RetrievedChunk, _rrf_merge


@pytest.fixture
def config():
    """Provides a baseline RetrievalConfig matching Chroma parameters with reranking mocked out."""
    return RetrievalConfig(top_k=5, score_threshold=0.5, reranking=RerankingConfig(enabled=False))


@pytest.fixture
def embed_service():
    """Provides a sterile mock of the embedding service returning arbitrary constants."""
    svc = AsyncMock()
    svc.embed_text.return_value = [0.1, 0.2, 0.3]
    return svc


@pytest.fixture
def vectorstore():
    """Provides a Mock vectorstore configured to return synthetic result sets representing typical Chroma hits."""
    vs = MagicMock()
    vs.query.return_value = [
        {
            "id": "1",
            "content": "Arkenfeld is a city.",
            "metadata": {"source_file": "Orte/Arkenfeld.md", "document_type": "markdown", "heading_hierarchy": "Orte > Arkenfeld"},
            "score": 0.9,
        },
        {
            "id": "2",
            "content": "A low-relevance chunk.",
            "metadata": {"source_file": "misc.md", "document_type": "markdown", "heading_hierarchy": ""},
            "score": 0.3,  # below threshold
        },
    ]
    return vs


@pytest.fixture
def retriever(config, embed_service, vectorstore):
    """Provides a completely integrated test Retriever combining all mocked subsystems."""
    return Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)


class TestRetrieve:
    """Test suite targeting simple retrieval evaluations absent complex reranking loops."""

    @pytest.mark.asyncio
    async def test_returns_chunks_above_threshold(self, retriever):
        """Verify queries successfully return mapped dictionaries filtered against the threshold."""
        results = await retriever.retrieve("where is arkenfeld")
        assert len(results) == 1
        assert results[0].content == "Arkenfeld is a city."

    @pytest.mark.asyncio
    async def test_filters_out_low_scores(self, retriever):
        """Verify vectorstore hits dipping beneath `score_threshold` are stripped proactively."""
        results = await retriever.retrieve("something")
        scores = [r.score for r in results]
        assert all(s >= 0.5 for s in scores)

    @pytest.mark.asyncio
    async def test_chunk_fields_populated(self, retriever):
        """Verify schema projection extracts fundamental metadata cleanly from the unrolled DB representation."""
        results = await retriever.retrieve("query")
        chunk = results[0]
        assert chunk.source_file == "Orte/Arkenfeld.md"
        assert chunk.document_type == "markdown"
        assert chunk.heading == "Orte > Arkenfeld"
        assert chunk.score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_image_type_excluded_via_filter(self, retriever, vectorstore):
        """Verify standard query constraints forcefully eject inline image references from standard text search."""
        await retriever.retrieve("query")
        kwargs = vectorstore.query.call_args[1]
        where = kwargs["where"]
        # must contain the image exclusion filter
        assert where is not None
        assert "document_type" in str(where)

    @pytest.mark.asyncio
    async def test_metadata_filters_combined(self, retriever, vectorstore):
        """Verify extra constraints stack gracefully via the `$and` implicit merger."""
        vectorstore.query.return_value = []
        await retriever.retrieve("query", metadata_filters={"category": "npc"})
        kwargs = vectorstore.query.call_args[1]
        where = kwargs["where"]
        assert "$and" in where

    @pytest.mark.asyncio
    async def test_custom_top_k(self, retriever, vectorstore):
        """Verify specific overrides map cleanly bypassing config default constraints."""
        vectorstore.query.return_value = []
        await retriever.retrieve("query", top_k=10)
        kwargs = vectorstore.query.call_args[1]
        assert kwargs["top_k"] == 10

    @pytest.mark.asyncio
    async def test_empty_vectorstore_returns_empty(self, retriever, vectorstore):
        """Verify missing results yield clean empty lists devoid of parsing crashes."""
        vectorstore.query.return_value = []
        results = await retriever.retrieve("nothing here")
        assert results == []


class TestRerank:
    """Test suite validating cross-encoder interactions enforcing relevance scoring over dense space assumptions."""

    @pytest.mark.asyncio
    async def test_reranking_reorders_chunks(self, config, embed_service, vectorstore):
        """Verify the cross-encoder correctly manipulates list indexing according to deep contextual evaluations."""
        config.reranking = RerankingConfig(enabled=True, top_k_rerank=2)
        vectorstore.query.return_value = [
            {"id": "1", "content": "chunk A", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.8},
            {"id": "2", "content": "chunk B", "metadata": {"source_file": "b.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.7},
            {"id": "3", "content": "chunk C", "metadata": {"source_file": "c.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.6},
        ]
        retriever = Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)

        mock_reranker = MagicMock()
        # reranker scores: C > A > B
        mock_reranker.predict.return_value = [0.5, 0.9, 0.3]
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("query")
        assert len(results) == 2
        assert results[0].content == "chunk B"  # highest reranker score (0.9 at index 1)
        assert results[1].content == "chunk A"

    @pytest.mark.asyncio
    async def test_max_per_source_backfills_when_pool_thin(self, config, embed_service, vectorstore):
        """Verify reranking caps gracefully relax mapping remaining items when alternative file pools run dry."""
        # Pool: 5 chunks from a.md, 1 from b.md. top_k_rerank=5, cap=2.
        # Pass 1: 2× a.md + 1× b.md = 3 selected, 3 a.md chunks parked.
        # Pass 2: backfill from overflow → 5 total. The cap must NOT reduce
        # the result count below top_k_rerank when alternatives are exhausted.
        config.reranking = RerankingConfig(
            enabled=True, top_k_rerank=5, max_per_source=2,
        )
        vectorstore.query.return_value = [
            {"id": "1", "content": "A1", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h1"}, "score": 0.95},
            {"id": "2", "content": "A2", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h2"}, "score": 0.9},
            {"id": "3", "content": "A3", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h3"}, "score": 0.85},
            {"id": "4", "content": "A4", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h4"}, "score": 0.8},
            {"id": "5", "content": "A5", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h5"}, "score": 0.75},
            {"id": "6", "content": "B1", "metadata": {"source_file": "b.md", "document_type": "markdown", "heading_hierarchy": "x"}, "score": 0.7},
        ]
        retriever = Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("query")
        # Must reach the requested count, not silently drop slots
        assert len(results) == 5
        sources = [r.source_file for r in results]
        assert sources.count("a.md") == 4   # 2 from cap + 2 backfilled
        assert sources.count("b.md") == 1
        # Re-sorted by score: highest first
        scores_descending = [r.score for r in results]
        assert scores_descending == sorted(scores_descending, reverse=True)

    @pytest.mark.asyncio
    async def test_max_per_source_caps_single_file_when_diversity_available(
        self, config, embed_service, vectorstore,
    ):
        """Verify distinct file enforcement rigidly rejects clustered hits provided diversity offsets exist in the pool."""
        # Pool: 3× a.md, 2× b.md, 1× c.md. top_k=5, cap=2.
        # With enough diversity, the cap stays binding (no backfill needed).
        config.reranking = RerankingConfig(
            enabled=True, top_k_rerank=5, max_per_source=2,
        )
        vectorstore.query.return_value = [
            {"id": "1", "content": "A1", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h1"}, "score": 0.95},
            {"id": "2", "content": "A2", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h2"}, "score": 0.9},
            {"id": "3", "content": "A3", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h3"}, "score": 0.85},
            {"id": "4", "content": "B1", "metadata": {"source_file": "b.md", "document_type": "markdown", "heading_hierarchy": "x"}, "score": 0.8},
            {"id": "5", "content": "B2", "metadata": {"source_file": "b.md", "document_type": "markdown", "heading_hierarchy": "y"}, "score": 0.75},
            {"id": "6", "content": "C1", "metadata": {"source_file": "c.md", "document_type": "markdown", "heading_hierarchy": "z"}, "score": 0.7},
        ]
        retriever = Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("query")
        sources = [r.source_file for r in results]
        # Cap binding: A3 must be skipped because b.md and c.md provide diversity.
        assert sources.count("a.md") == 2
        assert "A3" not in [r.content for r in results]
        assert "b.md" in sources
        assert "c.md" in sources


class TestRRFMerge:
    """Test suite validating the Reciprocal Rank Fusion primitive."""

    def test_empty_inputs(self):
        """Verify fusion of two empty lists yields an empty list."""
        assert _rrf_merge([], [], bm25_weight=0.3) == []

    def test_vector_only(self):
        """Verify a vector-only input preserves rank order and returns all items."""
        vector = [
            {"id": "a", "content": "A", "metadata": {}, "score": 0.9},
            {"id": "b", "content": "B", "metadata": {}, "score": 0.8},
        ]
        merged = _rrf_merge(vector, [], bm25_weight=0.3)
        assert [m["id"] for m in merged] == ["a", "b"]
        assert merged[0]["score"] > merged[1]["score"]

    def test_overlap_boosts_shared_item(self):
        """Verify items appearing in both lists rank above singletons of equal standalone rank."""
        vector = [
            {"id": "a", "content": "A", "metadata": {}, "score": 0.9},
            {"id": "b", "content": "B", "metadata": {}, "score": 0.8},
        ]
        bm25 = [
            {"id": "b", "content": "B", "metadata": {}, "score": 5.0},
            {"id": "c", "content": "C", "metadata": {}, "score": 3.0},
        ]
        merged = _rrf_merge(vector, bm25, bm25_weight=0.5)
        ids = [m["id"] for m in merged]
        # b appears in both lists → fused score > singleton a or c at same rank.
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_weight_shifts_balance(self):
        """Verify bm25_weight close to 1 lets the BM25 list dominate ordering."""
        vector = [{"id": "a", "content": "A", "metadata": {}, "score": 0.9}]
        bm25 = [{"id": "b", "content": "B", "metadata": {}, "score": 5.0}]
        merged_bm_heavy = _rrf_merge(vector, bm25, bm25_weight=0.9)
        assert merged_bm_heavy[0]["id"] == "b"
        merged_vec_heavy = _rrf_merge(vector, bm25, bm25_weight=0.1)
        assert merged_vec_heavy[0]["id"] == "a"


class TestHybridRetrieve:
    """Test suite verifying BM25 fusion wiring, thresholding and toggle semantics in retrieve()."""

    @pytest.fixture
    def hybrid_config(self):
        """Retrieval config with hybrid enabled by default and reranking off."""
        return RetrievalConfig(
            top_k=5,
            score_threshold=0.5,
            reranking=RerankingConfig(enabled=False),
            hybrid=HybridSearchConfig(enabled=True, bm25_weight=0.3, bm25_top_k=5),
        )

    def _make_bm25(self, docs):
        """Build an in-memory BM25 index from inline dicts, skipping the vectorstore."""
        from rank_bm25 import BM25Okapi
        from src.retrieval.bm25_index import _tokenize
        idx = BM25Index()
        idx._documents = docs
        corpus = [_tokenize(d["content"]) for d in docs]
        idx._index = BM25Okapi(corpus) if corpus else None
        idx._doc_token_sets = [set(toks) for toks in corpus]
        idx._is_built = True
        return idx

    @pytest.mark.asyncio
    async def test_hybrid_merges_vector_and_bm25(self, hybrid_config, embed_service):
        """Verify BM25-only candidates reach the output when hybrid is active."""
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "semantic hit about the city",
             "metadata": {"source_file": "v.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
        ]
        bm25 = self._make_bm25([
            {"id": "v1", "content": "semantic hit about the city",
             "metadata": {"source_file": "v.md", "document_type": "markdown"}},
            {"id": "b1", "content": "Langschwert 15 Gold exact match",
             "metadata": {"source_file": "weapons.md", "document_type": "markdown"}},
        ])
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        results = await retriever.retrieve("Langschwert Gold")
        sources = [r.source_file for r in results]
        assert "v.md" in sources
        assert "weapons.md" in sources  # BM25-only hit survived fusion

    @pytest.mark.asyncio
    async def test_hybrid_disabled_skips_bm25(self, hybrid_config, embed_service):
        """Verify hybrid=False bypasses BM25 completely, never calling query()."""
        hybrid_config.hybrid.enabled = False
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "hit",
             "metadata": {"source_file": "v.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
        ]
        bm25 = MagicMock(spec=BM25Index)
        bm25.is_built = True
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        await retriever.retrieve("anything")
        bm25.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_flag_override_forces_bm25(self, hybrid_config, embed_service):
        """Verify per-request hybrid=True overrides a disabled config default."""
        hybrid_config.hybrid.enabled = False
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "hit",
             "metadata": {"source_file": "v.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
        ]
        bm25 = self._make_bm25([
            {"id": "b1", "content": "exact keyword match",
             "metadata": {"source_file": "kw.md", "document_type": "markdown"}},
        ])
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        results = await retriever.retrieve("exact keyword", hybrid=True)
        sources = [r.source_file for r in results]
        assert "kw.md" in sources

    @pytest.mark.asyncio
    async def test_hybrid_does_not_nullify_threshold(self, hybrid_config, embed_service):
        """Regression: score_threshold=0.5 must not wipe hybrid results despite tiny RRF scores."""
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "Arkenfeld details",
             "metadata": {"source_file": "Arkenfeld.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
            {"id": "v2", "content": "low score noise",
             "metadata": {"source_file": "noise.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.2},
        ]
        bm25 = self._make_bm25([
            {"id": "b1", "content": "Arkenfeld keyword",
             "metadata": {"source_file": "kw.md", "document_type": "markdown"}},
        ])
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        results = await retriever.retrieve("Arkenfeld")
        # High-score vector hit + BM25 hit must survive; low-score vector noise drops.
        sources = [r.source_file for r in results]
        assert "Arkenfeld.md" in sources
        assert "noise.md" not in sources
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_hybrid_lazy_builds_index(self, hybrid_config, embed_service):
        """Verify the BM25 index is built on first hybrid call if not already built."""
        vs = MagicMock()
        vs.query.return_value = []
        # Minimal fake collection for build_from_vectorstore.
        collection = MagicMock()
        collection.get.return_value = {
            "ids": ["b1"],
            "documents": ["keyword content"],
            "metadatas": [{"source_file": "kw.md", "document_type": "markdown"}],
        }
        vs._get_collection.return_value = collection

        bm25 = BM25Index()
        assert bm25.is_built is False
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        await retriever.retrieve("anything")
        assert bm25.is_built is True

    @pytest.mark.asyncio
    async def test_hybrid_results_reach_reranker(self, hybrid_config, embed_service):
        """Verify BM25-only candidates are passed into the cross-encoder reranker."""
        hybrid_config.reranking = RerankingConfig(enabled=True, top_k_rerank=3)
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "vector hit",
             "metadata": {"source_file": "v.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
        ]
        bm25 = self._make_bm25([
            {"id": "b1", "content": "BM25-only exact keyword",
             "metadata": {"source_file": "kw.md", "document_type": "markdown"}},
        ])
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.6, 0.8]  # rerank passes both
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("exact keyword")
        # The reranker was called with both vector and BM25 candidates.
        pairs = mock_reranker.predict.call_args[0][0]
        contents = [p[1] for p in pairs]
        assert "vector hit" in contents
        assert "BM25-only exact keyword" in contents
        assert len(results) == 2
