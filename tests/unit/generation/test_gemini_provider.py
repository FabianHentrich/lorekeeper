from unittest.mock import AsyncMock, MagicMock, patch
import os

import pytest

from src.config.manager import GeminiConfig


class TestGeminiProvider:
    """Test suite for the GeminiProvider, focusing on initialization and key requirements."""

    def test_init_requires_api_key(self, monkeypatch):
        """
        Verify that instantiating GeminiProvider without specifying an API key in the
        environment variables raises a ValueError.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        config = GeminiConfig(api_key_env="GEMINI_API_KEY")

        from src.generation.providers.gemini import GeminiProvider
        with patch("src.generation.providers.gemini.dotenv_values", return_value={}):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiProvider(config=config)

    def test_init_with_api_key(self, monkeypatch):
        """
        Verify that instantiating GeminiProvider with a valid API key configures the
        provider successfully with the default model.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        config = GeminiConfig(api_key_env="GEMINI_API_KEY")

        from src.generation.providers.gemini import GeminiProvider
        provider = GeminiProvider(config=config)
        assert provider.model == "gemini-2.5-flash"
