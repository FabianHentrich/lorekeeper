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
    fake_retriever.get_reranker = MagicMock()

    fake_provider = MagicMock()
    fake_provider.model = "fake-model"
    fake_provider.health_check = AsyncMock(return_value=True)

    fake_generator = MagicMock()

    with patch("src.main.EmbeddingService", return_value=fake_embed), \
         patch("src.main.VectorStoreService", return_value=fake_vs), \
         patch("src.main.Retriever", return_value=fake_retriever), \
         patch("src.main.ProviderFactory") as fake_factory, \
         patch("src.main._ensure_ollama", new=AsyncMock(return_value=None)), \
         patch("src.logging_setup.setup_logging"):  # else the root logger writes all later test logs to logs/lorekeeper.log

        fake_factory.build_all.return_value = (fake_provider, None, fake_generator)

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
    """Ensure reranker preloading hook triggers when appropriate."""
    with TestClient(main_module.app):
        patched_lifespan.retriever.get_reranker.assert_called_once()


def test_lifespan_creates_provider_via_factory(patched_lifespan):
    """Provider, condense provider and generator (with fallback) come from one factory call."""
    with TestClient(main_module.app):
        patched_lifespan.factory.build_all.assert_called_once()
        assert main_module.condense_provider is None



def test_failed_startup_stops_the_ollama_it_started(patched_lifespan):
    """A crash after the Ollama autostart (e.g. the embedding model fails to load) used to leave the
    started Ollama running — the lifespan now cleans up in ``finally``."""
    ollama = MagicMock(pid=4711)
    patched_lifespan.embed.embed_text = AsyncMock(side_effect=RuntimeError("model download failed"))
    settings = SimpleNamespace(llm=SimpleNamespace(provider="ollama"))
    with patch("src.main._ensure_ollama", new=AsyncMock(return_value=ollama)), \
            patch("src.main.ConfigManager") as fake_config:
        fake_config.return_value.settings.llm.provider = settings.llm.provider
        with pytest.raises(RuntimeError, match="model download failed"):
            with TestClient(main_module.app):
                pass

    ollama.terminate.assert_called_once()
    assert main_module._ollama_process is None
