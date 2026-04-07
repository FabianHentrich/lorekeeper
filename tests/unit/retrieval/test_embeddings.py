from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config.manager import EmbeddingsConfig
from src.retrieval.embeddings import EmbeddingService


@pytest.fixture
def config():
    return EmbeddingsConfig(model="test-model", device="cpu", batch_size=32, normalize=True)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 4
    return model


@pytest.fixture
def service(config, mock_model):
    svc = EmbeddingService(config)
    svc._model = mock_model
    return svc


class TestEmbedText:
    @pytest.mark.asyncio
    async def test_returns_list_of_floats(self, service, mock_model):
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        result = await service.embed_text("hello")
        assert isinstance(result, list)
        assert result == pytest.approx([0.1, 0.2, 0.3, 0.4])

    @pytest.mark.asyncio
    async def test_passes_normalize_flag(self, service, mock_model):
        mock_model.encode.return_value = np.array([1.0, 0.0])
        await service.embed_text("test")
        _, kwargs = mock_model.encode.call_args
        assert kwargs["normalize_embeddings"] is True


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_returns_list_of_lists(self, service, mock_model):
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = await service.embed_texts(["a", "b"])
        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])

    @pytest.mark.asyncio
    async def test_passes_batch_size(self, service, mock_model):
        mock_model.encode.return_value = np.array([[0.0, 0.0]])
        await service.embed_texts(["x"])
        _, kwargs = mock_model.encode.call_args
        assert kwargs["batch_size"] == 32


class TestEmbedSync:
    def test_embed_text_sync(self, service, mock_model):
        mock_model.encode.return_value = np.array([0.5, 0.5])
        result = service.embed_text_sync("hello")
        assert result == pytest.approx([0.5, 0.5])

    def test_embed_texts_sync_ndarray(self, service, mock_model):
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = service.embed_texts_sync(["a", "b"])
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_embed_texts_sync_list_of_arrays(self, service, mock_model):
        mock_model.encode.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        result = service.embed_texts_sync(["a", "b"])
        assert result[1] == pytest.approx([0.3, 0.4])


class TestLazyModelLoading:
    def test_model_not_loaded_on_init(self, config):
        svc = EmbeddingService(config)
        assert svc._model is None

    def test_model_loaded_on_first_call(self, config):
        svc = EmbeddingService(config)
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([0.0])
        fake_model.get_sentence_embedding_dimension.return_value = 1
        with patch("src.retrieval.embeddings.SentenceTransformer", return_value=fake_model):
            result = svc.embed_text_sync("x")
        assert svc._model is fake_model
        assert isinstance(result, list)
