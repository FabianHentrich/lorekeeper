from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config.manager import EmbeddingsConfig
from src.retrieval.embeddings import EmbeddingService


@pytest.fixture
def config():
    """Provides a sterile baseline EmbeddingsConfig payload."""
    return EmbeddingsConfig(model="test-model", device="cpu", batch_size=32, normalize=True)


@pytest.fixture
def mock_model():
    """Provides a mocked SentenceTransformer avoiding GPU resource lockups during testing."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 4
    return model


@pytest.fixture
def service(config, mock_model):
    """Provides a pre-hydrated EmbeddingService skipping the slow model download phase."""
    svc = EmbeddingService(config)
    svc._model = mock_model
    return svc


class TestEmbedText:
    """Test suite validating standard text encoding payload formats mapping to ChromaDB specs."""

    @pytest.mark.asyncio
    async def test_returns_list_of_floats(self, service, mock_model):
        """Verify that underlying numpy array embeddings are cast reliably to raw python bounds."""
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        result = await service.embed_text("hello")
        assert isinstance(result, list)
        assert result == pytest.approx([0.1, 0.2, 0.3, 0.4])

    @pytest.mark.asyncio
    async def test_passes_normalize_flag(self, service, mock_model):
        """Verify normalization options cascade through the framework calls to the ML model."""
        mock_model.encode.return_value = np.array([1.0, 0.0])
        await service.embed_text("test")
        _, kwargs = mock_model.encode.call_args
        assert kwargs["normalize_embeddings"] is True


class TestEmbedSync:
    """Test suite validating the blocking synchronous vector pipelines utilized during mass ingestion."""

    def test_embed_texts_sync_ndarray(self, service, mock_model):
        """Verify batched multi-dimensional numpy inputs unroll correctly into document mappings."""
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = service.embed_texts_sync(["a", "b"])
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_embed_texts_sync_list_of_arrays(self, service, mock_model):
        """Verify batched outputs presenting as an array of arrays still decode properly."""
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        result = service.embed_texts_sync(["a", "b"])
        assert result[1] == pytest.approx([0.3, 0.4])


class TestLazyModelLoading:
    """Test suite validating memory lifecycle constraints deferring loading until immediately needed."""

    def test_model_not_loaded_on_init(self, config):
        """Verify instances do not initiate heavy HF model pulls just during init."""
        svc = EmbeddingService(config)
        assert svc._model is None

    def test_model_loaded_on_first_call(self, config):
        """Verify the first actual embedding attempt seamlessly blocks and hydrates the model reference."""
        svc = EmbeddingService(config)
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([0.0])
        fake_model.get_sentence_embedding_dimension.return_value = 1
        with patch("src.retrieval.embeddings.SentenceTransformer", return_value=fake_model):
            result = svc.embed_texts_sync(["x"])
        assert svc._model is fake_model
        assert isinstance(result, list)
