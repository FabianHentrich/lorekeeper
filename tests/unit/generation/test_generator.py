from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation.generator import Generator
from src.generation.providers.base import LLMResponse, StreamResult


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


def _stream_provider(tokens: list[str], fail_after: int | None = None, model: str = "primary-model"):
    """Provider whose generate_stream yields `tokens`, raising after `fail_after` of them."""
    provider = MagicMock()
    provider.model = model

    async def _gen(*args, **kwargs):
        for i, tok in enumerate(tokens):
            if fail_after is not None and i == fail_after:
                raise RuntimeError("stream broke")
            yield tok
        if fail_after is not None and fail_after >= len(tokens):
            raise RuntimeError("stream broke")

    provider.generate_stream = MagicMock(side_effect=_gen)
    return provider


async def _collect(agen) -> list[str]:
    return [tok async for tok in agen]


class TestGenerateStream:
    """Test suite for streaming generation and its fallback rules."""

    async def test_streams_primary_tokens(self):
        """Verify tokens of the primary provider are passed through unchanged."""
        gen = Generator(provider=_stream_provider(["a", "b"]))
        result = StreamResult()
        assert await _collect(gen.generate_stream("sys", "q", stream_result=result)) == ["a", "b"]
        assert result.model == "primary-model"

    async def test_falls_back_when_primary_fails_before_first_token(self):
        """Verify the fallback takes over if nothing has been emitted yet."""
        fallback = _stream_provider(["x", "y"], model="fallback-model")
        gen = Generator(provider=_stream_provider(["a"], fail_after=0), fallback_provider=fallback)
        result = StreamResult()
        assert await _collect(gen.generate_stream("sys", "q", stream_result=result)) == ["x", "y"]
        fallback.generate_stream.assert_called_once()
        # The model that actually answered is reported, not the failed primary
        assert result.model == "fallback-model"

    async def test_no_fallback_after_tokens_were_emitted(self):
        """Verify a mid-stream failure re-raises instead of duplicating output via the fallback."""
        fallback = _stream_provider(["x"])
        gen = Generator(provider=_stream_provider(["a", "b"], fail_after=1), fallback_provider=fallback)
        received = []
        with pytest.raises(RuntimeError, match="stream broke"):
            async for tok in gen.generate_stream("sys", "q"):
                received.append(tok)
        assert received == ["a"]
        fallback.generate_stream.assert_not_called()

    async def test_raises_without_fallback(self):
        """Verify a failure before the first token propagates when no fallback is configured."""
        gen = Generator(provider=_stream_provider(["a"], fail_after=0))
        with pytest.raises(RuntimeError):
            await _collect(gen.generate_stream("sys", "q"))

    async def test_passes_prompts_and_stream_result(self):
        """Verify system prompt and stream_result reach the provider."""
        primary = _stream_provider(["a"])
        sentinel = StreamResult()
        gen = Generator(provider=primary)
        await _collect(gen.generate_stream("sys", "q", stream_result=sentinel))
        args, kwargs = primary.generate_stream.call_args
        assert args == ("q",)
        assert kwargs == {"system_prompt": "sys", "stream_result": sentinel}
