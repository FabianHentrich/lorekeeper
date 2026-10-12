from unittest.mock import patch

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

    def test_build_config_timeout_in_milliseconds(self, monkeypatch):
        """
        Verify the configured timeout (seconds) reaches the request as HttpOptions milliseconds.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        config = GeminiConfig(api_key_env="GEMINI_API_KEY", timeout=30)

        from src.generation.providers.gemini import GeminiProvider
        provider = GeminiProvider(config=config)
        assert provider._build_config().http_options.timeout == 30_000


from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from google.genai.errors import ClientError  # noqa: E402

from src.generation.providers import gemini  # noqa: E402
from src.generation.providers.base import StreamResult  # noqa: E402


def _quota_error():
    return ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})


def _bad_request():
    return ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})


@pytest.fixture
def no_runtime_key():
    gemini.set_runtime_api_key(None)
    yield
    gemini.set_runtime_api_key(None)


@pytest.fixture
def provider(monkeypatch, no_runtime_key):
    """Provider with a fake genai client; sleeps (rate limit, backoff) are recorded, not waited."""
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    client = MagicMock()
    monkeypatch.setattr(gemini.genai, "Client", MagicMock(return_value=client))
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(gemini.asyncio, "sleep", fake_sleep)
    p = gemini.GeminiProvider(GeminiConfig(api_key_env="GEMINI_API_KEY"))
    p.sleeps = sleeps
    return p


class TestApiKeyStatus:
    """Where the key comes from — the key itself is never returned."""

    def test_runtime_key_wins_over_env(self, monkeypatch, no_runtime_key):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        gemini.set_runtime_api_key("ui-key")
        assert gemini.get_api_key_status(GeminiConfig(api_key_env="GEMINI_API_KEY")) == \
            {"has_key": True, "source": "runtime"}

    def test_env_key(self, monkeypatch, no_runtime_key):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        assert gemini.get_api_key_status(GeminiConfig(api_key_env="GEMINI_API_KEY"))["source"] == "env"

    def test_no_key(self, monkeypatch, no_runtime_key):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("src.generation.providers.gemini.dotenv_values", return_value={}):
            assert gemini.get_api_key_status(GeminiConfig(api_key_env="GEMINI_API_KEY")) == \
                {"has_key": False, "source": "none"}

    def test_empty_runtime_key_clears_the_override(self, monkeypatch, no_runtime_key):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        gemini.set_runtime_api_key("ui-key")
        gemini.set_runtime_api_key("")
        assert gemini.get_api_key_status(GeminiConfig(api_key_env="GEMINI_API_KEY"))["source"] == "env"

    def test_runtime_key_reaches_the_client(self, monkeypatch, no_runtime_key):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        client_cls = MagicMock()
        monkeypatch.setattr(gemini.genai, "Client", client_cls)
        gemini.set_runtime_api_key("ui-key")
        gemini.GeminiProvider(GeminiConfig(api_key_env="GEMINI_API_KEY"))
        client_cls.assert_called_once_with(api_key="ui-key")


class TestGenerate:
    async def test_returns_text_and_token_usage(self, provider):
        provider._client.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(
            text="Antwort", usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=3),
        ))

        resp = await provider.generate("Frage", system_prompt="System")

        assert (resp.content, resp.provider, resp.model) == ("Antwort", "gemini", "gemini-2.5-flash")
        assert (resp.usage["tokens_in"], resp.usage["tokens_out"]) == (12, 3)
        cfg = provider._client.aio.models.generate_content.call_args.kwargs["config"]
        assert cfg.system_instruction == "System"

    async def test_empty_response_without_usage(self, provider):
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=SimpleNamespace(text=None, usage_metadata=None))

        resp = await provider.generate("Frage")

        assert resp.content == ""
        assert "tokens_in" not in resp.usage

    async def test_retries_quota_errors_with_backoff(self, provider):
        ok = SimpleNamespace(text="ok", usage_metadata=None)
        provider._client.aio.models.generate_content = AsyncMock(side_effect=[_quota_error(), _quota_error(), ok])
        provider._rate_limit = AsyncMock()   # only the backoff sleeps remain

        assert (await provider.generate("Frage")).content == "ok"
        assert provider._client.aio.models.generate_content.await_count == 3
        assert provider.sleeps == [4, 8]   # 2**attempt * 4

    async def test_gives_up_after_max_retries(self, provider):
        provider._client.aio.models.generate_content = AsyncMock(side_effect=_quota_error())

        with pytest.raises(ClientError):
            await provider.generate("Frage")
        assert provider._client.aio.models.generate_content.await_count == 4   # 1 + 3 retries

    async def test_other_client_errors_are_not_retried(self, provider):
        provider._client.aio.models.generate_content = AsyncMock(side_effect=_bad_request())

        with pytest.raises(ClientError):
            await provider.generate("Frage")
        assert provider._client.aio.models.generate_content.await_count == 1


class TestRateLimit:
    async def test_second_request_waits_for_the_free_tier_interval(self, provider, monkeypatch):
        now = iter([100.0, 100.0, 101.0, 105.0])   # first call, then 1 s later
        # Only the provider's clock — the event loop keeps the real one
        monkeypatch.setattr(gemini, "time", SimpleNamespace(monotonic=lambda: next(now)))
        provider._last_request_time = 0

        await provider._rate_limit()
        await provider._rate_limit()

        assert provider.sleeps == [pytest.approx(gemini.MIN_REQUEST_INTERVAL - 1)]


class TestGenerateStream:
    async def test_yields_text_and_records_usage(self, provider):
        chunks = [
            SimpleNamespace(text="Ein", usage_metadata=None),
            SimpleNamespace(text="", usage_metadata=None),   # empty chunks are skipped
            SimpleNamespace(text="e Antwort", usage_metadata=SimpleNamespace(
                prompt_token_count=20, candidates_token_count=4, thoughts_token_count=7)),
        ]

        async def stream():
            for c in chunks:
                yield c

        provider._client.aio.models.generate_content_stream = AsyncMock(return_value=stream())
        result = StreamResult()

        tokens = [t async for t in provider.generate_stream("Frage", stream_result=result)]

        assert tokens == ["Ein", "e Antwort"]
        assert result.usage == {"tokens_in": 20, "tokens_out": 4, "tokens_thinking": 7}

    async def test_missing_usage_fields_count_as_zero(self, provider):
        async def stream():
            yield SimpleNamespace(text="x", usage_metadata=SimpleNamespace(
                prompt_token_count=None, candidates_token_count=2))

        provider._client.aio.models.generate_content_stream = AsyncMock(return_value=stream())
        result = StreamResult()

        _ = [t async for t in provider.generate_stream("Frage", stream_result=result)]

        assert result.usage == {"tokens_in": 0, "tokens_out": 2, "tokens_thinking": 0}


class TestHealthCheck:
    async def test_healthy_when_model_info_loads(self, provider):
        provider._client.aio.models.get = AsyncMock(return_value=object())
        assert await provider.health_check() is True

    async def test_unhealthy_on_any_error(self, provider):
        provider._client.aio.models.get = AsyncMock(side_effect=RuntimeError("invalid key"))
        assert await provider.health_check() is False
