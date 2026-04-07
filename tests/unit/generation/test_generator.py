from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation.generator import Generator
from src.generation.providers.base import LLMResponse


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.generate.return_value = LLMResponse(
        content="Antwort",
        model="test-model",
        provider="test",
    )
    return provider


@pytest.fixture
def mock_fallback():
    provider = AsyncMock()
    provider.generate.return_value = LLMResponse(
        content="Fallback-Antwort",
        model="fallback-model",
        provider="fallback",
    )
    return provider


class TestGenerator:
    @pytest.mark.asyncio
    async def test_generate(self, mock_provider):
        gen = Generator(provider=mock_provider)
        result = await gen.generate(system_prompt="sys", qa_prompt="frage")
        assert result.content == "Antwort"
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_fallback(self, mock_provider, mock_fallback):
        mock_provider.generate.side_effect = Exception("Primary down")
        gen = Generator(provider=mock_provider, fallback_provider=mock_fallback)

        result = await gen.generate(system_prompt="sys", qa_prompt="frage")
        assert result.content == "Fallback-Antwort"

    @pytest.mark.asyncio
    async def test_generate_no_fallback_raises(self, mock_provider):
        mock_provider.generate.side_effect = Exception("Down")
        gen = Generator(provider=mock_provider)

        with pytest.raises(Exception, match="Down"):
            await gen.generate(system_prompt="sys", qa_prompt="frage")

    @pytest.mark.asyncio
    async def test_condense_question(self, mock_provider):
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
