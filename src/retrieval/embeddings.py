import logging

import numpy as np
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer

from src.config.manager import EmbeddingsConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self._model: SentenceTransformer | None = None
        self._is_e5 = "e5" in config.model.lower()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.config.model}")
            self._model = SentenceTransformer(
                self.config.model,
                device=self.config.device if self.config.device != "auto" else None,
            )
            logger.info(f"Embedding model loaded (dim={self._model.get_sentence_embedding_dimension()})")
        return self._model

    def _with_query_prefix(self, text: str) -> str:
        return f"query: {text}" if self._is_e5 else text

    def _with_passage_prefix(self, text: str) -> str:
        return f"passage: {text}" if self._is_e5 else text

    async def embed_text(self, text: str) -> list[float]:
        """Embed a query string."""
        model = self._get_model()
        embedding = await run_in_threadpool(
            model.encode,
            self._with_query_prefix(text),
            normalize_embeddings=self.config.normalize,
        )
        return embedding.tolist()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of passage strings."""
        model = self._get_model()
        prefixed = [self._with_passage_prefix(t) for t in texts]
        embeddings = await run_in_threadpool(
            model.encode,
            prefixed,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
        )
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [e.tolist() for e in embeddings]

    def embed_text_sync(self, text: str) -> list[float]:
        """Embed a query string (synchronous)."""
        model = self._get_model()
        embedding = model.encode(
            self._with_query_prefix(text),
            normalize_embeddings=self.config.normalize,
        )
        return embedding.tolist()

    def embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of passage strings (synchronous)."""
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
