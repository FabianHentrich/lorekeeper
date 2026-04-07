from unittest.mock import AsyncMock, MagicMock, patch
import os

import pytest

from src.config.manager import GeminiConfig


class TestGeminiProvider:
    def test_init_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        config = GeminiConfig(api_key_env="GEMINI_API_KEY")

        from src.generation.providers.gemini import GeminiProvider
        with patch("src.generation.providers.gemini.dotenv_values", return_value={}):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiProvider(config=config)

    def test_init_with_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        config = GeminiConfig(api_key_env="GEMINI_API_KEY")

        from src.generation.providers.gemini import GeminiProvider
        provider = GeminiProvider(config=config)
        assert provider.model == "gemini-2.5-flash"
