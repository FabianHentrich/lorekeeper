from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.config.manager import OllamaConfig
from src.generation.providers.ollama import OllamaProvider


class TestOllamaProvider:
    def test_init_with_config(self):
        config = OllamaConfig(base_url="http://test:11434", model="test-model")
        provider = OllamaProvider(config=config)
        assert provider.model == "test-model"
        assert provider.base_url == "http://test:11434"

    def test_init_with_explicit_params(self):
        provider = OllamaProvider(base_url="http://custom:1234", model="custom-model")
        assert provider.model == "custom-model"
        assert provider.base_url == "http://custom:1234"

    @pytest.mark.asyncio
    async def test_generate(self):
        config = OllamaConfig()
        provider = OllamaProvider(config=config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {}

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.generate("Test prompt", system_prompt="System")
        assert result.content == "Test response"
        assert result.provider == "ollama"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        provider = OllamaProvider(config=OllamaConfig())
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_resp = MagicMock(status_code=200)
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            assert await provider.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        provider = OllamaProvider(config=OllamaConfig())
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            assert await provider.health_check() is False
