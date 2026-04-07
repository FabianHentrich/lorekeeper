"""
Integration test for the RAG pipeline.

Uses real ChromaDB (in-memory via EphemeralClient) and real sentence-transformer
embeddings, but mocks the LLM call. No Ollama or Gemini required.
"""
from unittest.mock import AsyncMock, MagicMock

import chromadb
import pytest

from src.config.manager import (
    EmbeddingsConfig,
    RerankingConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from src.generation.generator import Generator
from src.generation.providers.base import LLMResponse
from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import Retriever
from src.retrieval.vectorstore import VectorStoreService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embed_service():
    """Real embedding service — loads the multilingual MiniLM model once."""
    config = EmbeddingsConfig(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        batch_size=32,
        normalize=True,
    )
    return EmbeddingService(config)


@pytest.fixture(scope="module")
def vectorstore(embed_service):
    """Real ChromaDB running in-memory (no disk I/O)."""
    config = VectorStoreConfig(
        mode="embedded",
        collection_name="integration_test",
        distance_metric="cosine",
    )
    svc = VectorStoreService(config, embed_service)
    # Override the client with an ephemeral in-memory instance
    svc._client = chromadb.EphemeralClient()
    return svc


@pytest.fixture(scope="module")
def populated_vectorstore(vectorstore, embed_service):
    """Insert a small set of test chunks once for the whole module."""
    chunks = [
        Chunk(
            content="Arkenfeld is a fortified city in the eastern provinces.",
            source_file="Orte/Arkenfeld.md",
            source_path="/vault/Orte/Arkenfeld.md",
            document_type="markdown",
            heading_hierarchy=["Orte", "Arkenfeld"],
            chunk_index=0,
            total_chunks=1,
            metadata={"aliases": ["Arken"], "content_hash": "sha256:aaa"},
        ),
        Chunk(
            content="Aldric is a veteran NPC blacksmith living in Arkenfeld.",
            source_file="NPCs/Aldric.md",
            source_path="/vault/NPCs/Aldric.md",
            document_type="markdown",
            heading_hierarchy=["NPCs", "Aldric"],
            chunk_index=0,
            total_chunks=1,
            metadata={"content_hash": "sha256:bbb"},
        ),
        Chunk(
            content="The Dragon of the North terrorises the mountain passes.",
            source_file="Gegner/Drache.md",
            source_path="/vault/Gegner/Drache.md",
            document_type="markdown",
            heading_hierarchy=["Gegner", "Drache"],
            chunk_index=0,
            total_chunks=1,
            metadata={"content_hash": "sha256:ccc"},
        ),
        Chunk(
            content="portrait_aldric.png",
            source_file="images/portrait_aldric.png",
            source_path="/vault/images/portrait_aldric.png",
            document_type="image",
            chunk_index=0,
            total_chunks=1,
            metadata={"content_hash": "sha256:ddd"},
        ),
    ]
    vectorstore.upsert_chunks(chunks)
    return vectorstore


@pytest.fixture(scope="module")
def retriever(populated_vectorstore, embed_service):
    config = RetrievalConfig(
        top_k=5,
        score_threshold=0.0,  # accept all results so we can test filtering logic
        reranking=RerankingConfig(enabled=False),
    )
    return Retriever(
        config=config,
        embedding_service=embed_service,
        vectorstore=populated_vectorstore,
    )


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    provider.generate.return_value = LLMResponse(
        content="Arkenfeld is a fortified city in the east.",
        model="mock-model",
        provider="mock",
    )
    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUpsertAndCount:
    def test_chunks_stored(self, populated_vectorstore):
        assert populated_vectorstore.count() >= 4


class TestRetrieval:
    @pytest.mark.asyncio
    async def test_relevant_chunk_returned(self, retriever):
        results = await retriever.retrieve("Tell me about Arkenfeld")
        contents = [r.content for r in results]
        assert any("Arkenfeld" in c for c in contents)

    @pytest.mark.asyncio
    async def test_images_excluded(self, retriever):
        results = await retriever.retrieve("portrait aldric")
        for r in results:
            assert r.document_type != "image"

    @pytest.mark.asyncio
    async def test_npc_query(self, retriever):
        results = await retriever.retrieve("Aldric NPC blacksmith")
        assert any("Aldric" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_score_threshold_filters(self, embed_service, populated_vectorstore):
        """Chunks below threshold must be excluded."""
        strict_config = RetrievalConfig(
            top_k=5,
            score_threshold=0.99,  # only near-perfect matches pass
            reranking=RerankingConfig(enabled=False),
        )
        strict_retriever = Retriever(
            config=strict_config,
            embedding_service=embed_service,
            vectorstore=populated_vectorstore,
        )
        results = await strict_retriever.retrieve("completely unrelated query xyz123")
        assert results == []


class TestContentHashLookup:
    def test_hash_returned_for_stored_file(self, populated_vectorstore):
        hashes = populated_vectorstore.get_all_content_hashes()
        assert "Orte/Arkenfeld.md" in hashes
        assert hashes["Orte/Arkenfeld.md"] == "sha256:aaa"


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_retrieve_then_generate(self, retriever, mock_llm_provider):
        """End-to-end: retrieve context, pass to generator, get response."""
        chunks = await retriever.retrieve("What is Arkenfeld?")
        assert chunks, "Expected at least one retrieved chunk"

        context = "\n\n".join(c.content for c in chunks)
        system_prompt = "You are a lore assistant."
        qa_prompt = f"Context:\n{context}\n\nQuestion: What is Arkenfeld?"

        generator = Generator(provider=mock_llm_provider)
        response = await generator.generate(system_prompt=system_prompt, qa_prompt=qa_prompt)

        assert response.content
        mock_llm_provider.generate.assert_called_once()
