"""Shared integration-test helpers: one real embedding model per test run, SSE parsing."""
import json

import pytest

from src.config.manager import EmbeddingsConfig
from src.retrieval.embeddings import EmbeddingService


@pytest.fixture(scope="session")
def embed_service():
    """Real embedding model (small multilingual MiniLM), loaded once for all integration tests."""
    return EmbeddingService(EmbeddingsConfig(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu", batch_size=32, normalize=True,
    ))


def parse_sse(raw: str) -> list[dict]:
    """The JSON events of a /query/stream response body."""
    return [json.loads(line[len("data: "):]) for line in raw.splitlines() if line.startswith("data: ")]
