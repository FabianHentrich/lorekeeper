from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.manager import RerankingConfig, RetrievalConfig
from src.retrieval.retriever import Retriever, RetrievedChunk


@pytest.fixture
def config():
    return RetrievalConfig(top_k=5, score_threshold=0.5, reranking=RerankingConfig(enabled=False))


@pytest.fixture
def embed_service():
    svc = AsyncMock()
    svc.embed_text.return_value = [0.1, 0.2, 0.3]
    return svc


@pytest.fixture
def vectorstore():
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
    return Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_chunks_above_threshold(self, retriever):
        results = await retriever.retrieve("where is arkenfeld")
        assert len(results) == 1
        assert results[0].content == "Arkenfeld is a city."

    @pytest.mark.asyncio
    async def test_filters_out_low_scores(self, retriever):
        results = await retriever.retrieve("something")
        scores = [r.score for r in results]
        assert all(s >= 0.5 for s in scores)

    @pytest.mark.asyncio
    async def test_chunk_fields_populated(self, retriever):
        results = await retriever.retrieve("query")
        chunk = results[0]
        assert chunk.source_file == "Orte/Arkenfeld.md"
        assert chunk.document_type == "markdown"
        assert chunk.heading == "Orte > Arkenfeld"
        assert chunk.score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_image_type_excluded_via_filter(self, retriever, vectorstore):
        await retriever.retrieve("query")
        kwargs = vectorstore.query.call_args[1]
        where = kwargs["where"]
        # must contain the image exclusion filter
        assert where is not None
        assert "document_type" in str(where)

    @pytest.mark.asyncio
    async def test_metadata_filters_combined(self, retriever, vectorstore):
        vectorstore.query.return_value = []
        await retriever.retrieve("query", metadata_filters={"category": "npc"})
        kwargs = vectorstore.query.call_args[1]
        where = kwargs["where"]
        assert "$and" in where

    @pytest.mark.asyncio
    async def test_custom_top_k(self, retriever, vectorstore):
        vectorstore.query.return_value = []
        await retriever.retrieve("query", top_k=10)
        kwargs = vectorstore.query.call_args[1]
        assert kwargs["top_k"] == 10

    @pytest.mark.asyncio
    async def test_empty_vectorstore_returns_empty(self, retriever, vectorstore):
        vectorstore.query.return_value = []
        results = await retriever.retrieve("nothing here")
        assert results == []


class TestRerank:
    @pytest.mark.asyncio
    async def test_reranking_reorders_chunks(self, config, embed_service, vectorstore):
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
