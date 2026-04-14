import logging

import numpy as np
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer

from src.config.manager import EmbeddingsConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Provides semantic embedding capabilities via local sentence-transformers models.

    This service is responsible for converting raw text (both queries and document chunks)
    into fixed-size dense vectors. It supports dynamic prompt prefixing required by models
    like intfloat/multilingual-e5-large (e.g. 'query:' vs 'passage:'). It also ensures CPU-bound
    encoding tasks are properly offloaded to the asyncio threadpool to avoid blocking FastAPI.
    """

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self._model: SentenceTransformer | None = None
        self._is_e5 = "e5" in config.model.lower()

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the underlying SentenceTransformer model into memory."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.config.model}")
            self._model = SentenceTransformer(
                self.config.model,
                device=self.config.device if self.config.device != "auto" else None,
            )
            logger.info(f"Embedding model loaded (dim={self._model.get_sentence_embedding_dimension()})")
        return self._model

    def _with_query_prefix(self, text: str) -> str:
        """Prepend 'query:' to the string if the active model requires E5-style prompting."""
        return f"query: {text}" if self._is_e5 else text

    def _with_passage_prefix(self, text: str) -> str:
        """Prepend 'passage:' to the string if the active model requires E5-style prompting."""
        return f"passage: {text}" if self._is_e5 else text

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single query string asynchronously.

        Always wrapped in `run_in_threadpool` because `model.encode()` is highly CPU-bound
        and would otherwise halt the entire FastAPI event loop for concurrent users.
        """
        model = self._get_model()
        embedding = await run_in_threadpool(
            model.encode,
            self._with_query_prefix(text),
            normalize_embeddings=self.config.normalize,
        )
        return embedding.tolist()

    def embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of passage strings synchronously.

        Typically used in background tasks like the document Ingestion Orchestrator,
        where blocking a specific thread is perfectly acceptable. Applies the
        configured batch size processing.
        """
        model = self._get_model()
        prefixed = [self._with_passage_prefix(t) for t in texts]
        embeddings = model.encode(
            prefixed,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
        )
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [e.tolist() for e in embeddings]
