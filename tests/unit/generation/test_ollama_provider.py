from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.manager import OllamaConfig
from src.generation.providers.base import StreamResult
from src.generation.providers.ollama import OllamaProvider


class TestOllamaProvider:
    """Test suite for validating the capabilities, configuration overrides, and token streaming
    mechanics of the integrated Ollama provider.
    """

    def test_init_with_config(self):
        """
        Verify that configuring Ollama Provider directly passes default or specified values.
        """
        config = OllamaConfig(base_url="http://test:11434", model="test-model")
        provider = OllamaProvider(config=config)
        assert provider.model == "test-model"
        assert provider.base_url == "http://test:11434"

    def test_init_applies_config_values(self):
        """
        Verify that detailed configuration values (e.g. temperature overrides) correctly map
        to the client initialization phase.
        """
        config = OllamaConfig(base_url="http://custom:1234", model="custom-model", temperature=0.7)
        provider = OllamaProvider(config=config)
        assert provider.model == "custom-model"
        assert provider.base_url == "http://custom:1234"
        assert provider.config.temperature == 0.7

    @pytest.mark.asyncio
    async def test_generate(self):
        """
        Verify standard blocking response generation parses deeply nested choices and
        correctly handles empty token returns while populating the primary content payload.
        """
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
    @pytest.mark.parametrize("model, extra_body", [
        ("qwen3:14b", {"reasoning_effort": "none"}),
        ("llama3:8b", {}),
    ])
    async def test_qwen3_thinking_disabled_via_reasoning_effort(self, model, extra_body):
        """
        Verify Qwen3 thinking is switched off via reasoning_effort — Ollama ignores the
        old "/no_think" prompt suffix, which must no longer be appended.
        """
        provider = OllamaProvider(config=OllamaConfig(model=model))
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = None
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.generate("Frage")

        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == extra_body
        assert kwargs["messages"][-1]["content"] == "Frage"

    @pytest.mark.asyncio
    async def test_generate_uses_live_config_timeout(self):
        """
        Verify the request timeout comes from the config on each call, so Settings-page
        changes apply without rebuilding the provider.
        """
        config = OllamaConfig(timeout=42)
        provider = OllamaProvider(config=config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = None
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.generate("p")
        config.timeout = 99
        await provider.generate("p")

        timeouts = [c.kwargs["timeout"] for c in provider._client.chat.completions.create.call_args_list]
        assert timeouts == [42, 99]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model, pulled, expected", [
        ("qwen3:8b", ["qwen3:8b", "llama3:latest"], True),
        ("qwen3", ["qwen3:latest"], True),           # untagged name resolves to :latest
        ("qwen3:8b", [], False),                     # daemon up, model not pulled
        ("qwen3:8b", ["qwen3:14b"], False),
    ])
    async def test_health_check_requires_pulled_model(self, model, pulled, expected):
        """
        Verify the healthcheck is only truthy when the daemon responds AND the configured
        model is pulled — a running daemon without the model would fail every query.
        """
        provider = OllamaProvider(config=OllamaConfig(model=model))
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {"models": [{"name": n} for n in pulled]}
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            assert await provider.health_check() is expected

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """
        Verify that receiving underlying connection failures via httpx gracefully
        translates to a falsy provider healthcheck state rather than outright exceptions.
        """
        provider = OllamaProvider(config=OllamaConfig())
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            assert await provider.health_check() is False

    @pytest.mark.asyncio
    async def test_stream_strips_think_blocks_across_chunks(self):
        """
        Verify that <think> blocks spanning multiple token chunks must still be
        filtered accurately without corrupting trailing messages.
        """
        provider = OllamaProvider(config=OllamaConfig(model="qwen3:8b"))

        def make_chunk(content, usage=None):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
            chunk.usage = usage
            return chunk

        # Simulate tag split across chunks: "<th" + "ink>secret</thi" + "nk>hello"
        pieces = ["Hi ", "<th", "ink>", "secret ", "</thi", "nk>", "world"]

        class FakeStream:
            def __init__(self, items):
                self._iter = iter(items)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration from None

        fake_stream = FakeStream([make_chunk(p) for p in pieces])
        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=fake_stream)

        out = []
        async for tok in provider.generate_stream("prompt"):
            out.append(tok)

        full = "".join(out)
        assert "secret" not in full
        assert "<think>" not in full and "</think>" not in full
        assert "Hi" in full and "world" in full

    @pytest.mark.asyncio
    async def test_stream_populates_stream_result_usage(self):
        """
        Verify that final chunks in asynchronous streaming correctly enrich
        usage statistic references provided ahead of processing.
        """
        provider = OllamaProvider(config=OllamaConfig())

        def make_chunk(content=None, usage=None):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
            chunk.usage = usage
            return chunk

        usage = MagicMock(prompt_tokens=7, completion_tokens=11)
        pieces = [make_chunk("hello "), make_chunk("world"), make_chunk(None, usage=usage)]

        class FakeStream:
            def __init__(self, items):
                self._iter = iter(items)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration from None

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=FakeStream(pieces))

        ctx = StreamResult()
        out = []
        async for tok in provider.generate_stream("p", stream_result=ctx):
            out.append(tok)

        assert "".join(out) == "hello world"
        assert ctx.usage == {"tokens_in": 7, "tokens_out": 11, "tokens_thinking": 0}

    @pytest.mark.asyncio
    async def test_stream_counts_reasoning_field_without_emitting_it(self):
        """
        Verify reasoning Ollama sends in the separate field is counted as thinking
        tokens but never reaches the answer text.
        """
        from openai.types.chat.chat_completion_chunk import ChoiceDelta

        provider = OllamaProvider(config=OllamaConfig(model="qwen3:14b"))

        def make_chunk(delta=None, usage=None):
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=ChoiceDelta.model_validate(delta))] if delta else []
            chunk.usage = usage
            return chunk

        pieces = [
            make_chunk({"content": "", "reasoning": "Hmm, "}),
            make_chunk({"content": "", "reasoning": "überlegen"}),
            make_chunk({"content": "Antwort"}),
            make_chunk(usage=MagicMock(prompt_tokens=3, completion_tokens=4)),
        ]

        class FakeStream:
            def __init__(self, items):
                self._iter = iter(items)
            def __aiter__(self):
                return self
            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration from None

        provider._client = AsyncMock()
        provider._client.chat.completions.create = AsyncMock(return_value=FakeStream(pieces))

        ctx = StreamResult()
        out = [tok async for tok in provider.generate_stream("p", stream_result=ctx)]

        assert "".join(out) == "Antwort"
        assert ctx.usage["tokens_thinking"] == 2
