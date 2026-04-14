from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.manager import RerankingConfig, RetrievalConfig
from src.retrieval.retriever import Retriever, RetrievedChunk


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
