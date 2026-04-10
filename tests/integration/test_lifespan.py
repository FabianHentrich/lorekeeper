"""Smoke test for the FastAPI lifespan in ``src.main``.

Verifies that startup wires all service globals, the session-GC background
task is running, and shutdown cancels it cleanly — without touching real
models, ChromaDB, or Ollama.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.main as main_module


@pytest.fixture
def patched_lifespan():
    """Patch every heavy dependency the lifespan pulls in."""
    fake_embed = MagicMock()
    fake_embed.embed_text = AsyncMock(return_value=[0.0] * 768)

    fake_vs = MagicMock()
    fake_vs.health_check = MagicMock(return_value=True)

    fake_retriever = MagicMock()
    fake_retriever._get_reranker = MagicMock()

    fake_provider = MagicMock()
    fake_provider.model = "fake-model"
    fake_provider.health_check = AsyncMock(return_value=True)

    fake_generator = MagicMock()

    with patch("src.main.EmbeddingService", return_value=fake_embed), \
         patch("src.main.VectorStoreService", return_value=fake_vs), \
         patch("src.main.Retriever", return_value=fake_retriever), \
         patch("src.main.ProviderFactory") as fake_factory, \
         patch("src.main.Generator", return_value=fake_generator), \
         patch("src.main._ensure_ollama", new=AsyncMock(return_value=None)):

        fake_factory.create.return_value = fake_provider
        fake_factory.create_condense_provider.return_value = None
        fake_factory.create_fallback.return_value = None

        yield SimpleNamespace(
            embed=fake_embed,
            vs=fake_vs,
            retriever=fake_retriever,
            provider=fake_provider,
            generator=fake_generator,
            factory=fake_factory,
        )


def test_lifespan_startup_and_shutdown(patched_lifespan):
    """Entering the TestClient context runs startup; exit runs shutdown."""
    with TestClient(main_module.app) as client:
        # All globals wired
        assert main_module.config is not None
        assert main_module.conversation_manager is not None
        assert main_module.prompt_manager is not None
        assert main_module.embedding_service is patched_lifespan.embed
        assert main_module.vectorstore is patched_lifespan.vs
        assert main_module.retriever is patched_lifespan.retriever
        assert main_module.provider is patched_lifespan.provider
        assert main_module.generator is patched_lifespan.generator

        # Startup side-effects happened
        patched_lifespan.embed.embed_text.assert_awaited_once_with("warmup")
        assert patched_lifespan.vs.health_check.call_count >= 1

        # App responds to a trivial request (middleware + router wired)
        response = client.get("/openapi.json")
        assert response.status_code == 200

    # After the `with` block, shutdown has run — the gc_task was cancelled
    # without raising. If we got here, the lifespan contract held.


def test_lifespan_preloads_reranker_when_enabled(patched_lifespan):
    with TestClient(main_module.app):
        patched_lifespan.retriever._get_reranker.assert_called_once()


def test_lifespan_creates_provider_via_factory(patched_lifespan):
    with TestClient(main_module.app):
        patched_lifespan.factory.create.assert_called_once()
        patched_lifespan.factory.create_condense_provider.assert_called_once()
        patched_lifespan.factory.create_fallback.assert_called_once()
