from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation.generator import Generator
from src.generation.providers.base import LLMResponse


@pytest.fixture
def mock_provider():
    """
    Provide a functionally mocked primary LLM provider capable of generating standard responses.

    Returns:
        AsyncMock: A mock provider configured to return a successful LLMResponse.
    """
    provider = AsyncMock()
    provider.generate.return_value = LLMResponse(
        content="Antwort",
        model="test-model",
        provider="test",
    )
    return provider


@pytest.fixture
def mock_fallback():
    """
    Provide a functionally mocked fallback LLM provider.

    Returns:
        AsyncMock: A mock provider returning a distinct fallback response for verification.
    """
    provider = AsyncMock()
    provider.generate.return_value = LLMResponse(
        content="Fallback-Antwort",
        model="fallback-model",
        provider="fallback",
    )
    return provider


class TestGenerator:
    """Test suite for the Generator wrapper, validating primary and fallback LLM orchestration."""

    @pytest.mark.asyncio
    async def test_generate(self, mock_provider):
        """
        Verify that the Generator successfully routes the prompt through the
        primary provider and returns its response.
        """
        gen = Generator(provider=mock_provider)
        result = await gen.generate(system_prompt="sys", qa_prompt="frage")
        assert result.content == "Antwort"
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_fallback(self, mock_provider, mock_fallback):
        """
        Verify that if the primary provider raises an exception, the Generator
        successfully delegates to the fallback provider.
        """
        mock_provider.generate.side_effect = Exception("Primary down")
        gen = Generator(provider=mock_provider, fallback_provider=mock_fallback)

        result = await gen.generate(system_prompt="sys", qa_prompt="frage")
        assert result.content == "Fallback-Antwort"

    @pytest.mark.asyncio
    async def test_generate_no_fallback_raises(self, mock_provider):
        """
        Verify that if the primary provider fails and no fallback is configured,
        the exception is propagated to the caller.
        """
        mock_provider.generate.side_effect = Exception("Down")
        gen = Generator(provider=mock_provider)

        with pytest.raises(Exception, match="Down"):
            await gen.generate(system_prompt="sys", qa_prompt="frage")

    @pytest.mark.asyncio
    async def test_condense_question(self, mock_provider):
        """
        Verify that condensing a conversation history correctly strips leading/trailing
        whitespace from the resulting standalone question string using the primary provider.
        """
        mock_provider.generate.return_value = LLMResponse(
            content="  Standalone Frage  ",
            model="test",
            provider="test",
        )
        gen = Generator(provider=mock_provider)
        result = await gen.condense_question("condensed prompt")
        assert result == "Standalone Frage"

    @pytest.mark.asyncio
    async def test_condense_with_separate_provider(self, mock_provider):
        """
        Verify that question condensation correctly utilizes a distinct provider
        when one is provided via `condense_provider`.
        """
        condense = AsyncMock()
        condense.generate.return_value = LLMResponse(
            content="Condensed",
            model="small",
            provider="test",
        )
        gen = Generator(provider=mock_provider)
        result = await gen.condense_question("prompt", condense_provider=condense)
        assert result == "Condensed"
        mock_provider.generate.assert_not_called()
