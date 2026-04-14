import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator

from dotenv import dotenv_values
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from src.config.manager import GeminiConfig
from .base import BaseLLMProvider, LLMResponse, StreamResult

logger = logging.getLogger(__name__)

# Gemini Free Tier: 15 RPM
MAX_RPM = 15
MIN_REQUEST_INTERVAL = 60.0 / MAX_RPM  # 4 seconds between requests


# Runtime override: lets the user supply a Gemini API key from the UI without
# touching .env. Held in process memory only — never persisted.
_runtime_api_key: str | None = None


def set_runtime_api_key(key: str | None) -> None:
    """Store/clear the runtime Gemini API key (process-local, not persisted)."""
    global _runtime_api_key
    _runtime_api_key = key or None


def get_api_key_status(config: GeminiConfig) -> dict:
    """Return whether a Gemini API key is available and where it came from.
    Never returns the key itself."""
    if _runtime_api_key:
        return {"has_key": True, "source": "runtime"}
    if os.environ.get(config.api_key_env) or dotenv_values(".env").get(config.api_key_env):
        return {"has_key": True, "source": "env"}
    return {"has_key": False, "source": "none"}


class GeminiProvider(BaseLLMProvider):
    """LLM provider for Google Gemini, with built-in rate limiting and 429 retry logic."""

    provider = "gemini"

    def __init__(self, config: GeminiConfig):
        """Resolve the API key (runtime override > env > .env) and build the genai client."""
        self.config = config
        self.model = config.model

        api_key = (
            _runtime_api_key
            or os.environ.get(config.api_key_env, "")
            or dotenv_values(".env").get(config.api_key_env, "")
        )
        if not api_key:
            raise ValueError(f"No Gemini API key — set {config.api_key_env} or supply one via the UI")

        self._client = genai.Client(api_key=api_key)
        self._last_request_time: float = 0
        self._lock = asyncio.Lock()

    async def _rate_limit(self):
        """Sleep if needed so we never issue requests faster than Gemini's 15 RPM free tier."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < MIN_REQUEST_INTERVAL:
                wait = MIN_REQUEST_INTERVAL - elapsed
                logger.debug(f"Rate limit: waiting {wait:.1f}s")
                await asyncio.sleep(wait)
            self._last_request_time = time.monotonic()

    async def _retry_on_429(self, coro_fn, max_retries: int = 3):
        """Invoke `coro_fn`, retrying with exponential backoff on 429 quota errors."""
        for attempt in range(max_retries + 1):
            await self._rate_limit()
            try:
                return await coro_fn()
            except ClientError as e:
                if "429" in str(e) and attempt < max_retries:
                    wait = min(2 ** attempt * 4, 60)
                    logger.warning(f"Gemini 429, retry {attempt + 1}/{max_retries} in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise

    def _build_config(self, **kwargs) -> types.GenerateContentConfig:
        """Assemble the per-request GenerateContentConfig from defaults + per-call overrides."""
        return types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            system_instruction=kwargs.get("system_prompt", None),
        )

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        """Produce a complete, non-streamed Gemini response with rate limiting and 429 retries."""
        gen_config = self._build_config(system_prompt=system_prompt or None, **kwargs)

        start = time.time()

        async def _call():
            """Single Gemini API invocation — wrapped so _retry_on_429 can retry it."""
            return await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_config,
            )

        response = await self._retry_on_429(_call)
        latency_ms = (time.time() - start) * 1000

        content = response.text or ""
        usage = {"latency_ms": latency_ms}
        if response.usage_metadata:
            usage["tokens_in"] = response.usage_metadata.prompt_token_count
            usage["tokens_out"] = response.usage_metadata.candidates_token_count

        return LLMResponse(
            content=content,
            model=self.model,
            provider="gemini",
            usage=usage,
            raw_response={},
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        stream_result: StreamResult | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream Gemini tokens, updating ``stream_result`` with usage metadata as it arrives."""
        gen_config = self._build_config(system_prompt=system_prompt or None, **kwargs)

        await self._rate_limit()

        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=gen_config,
        ):
            um = getattr(chunk, "usage_metadata", None)
            if um and stream_result is not None:
                stream_result.usage = {
                    "tokens_in": getattr(um, "prompt_token_count", 0) or 0,
                    "tokens_out": getattr(um, "candidates_token_count", 0) or 0,
                    "tokens_thinking": getattr(um, "thoughts_token_count", 0) or 0,
                }
            if chunk.text:
                yield chunk.text

    async def health_check(self) -> bool:
        """Return True if Gemini accepts a trivial model-info call with the configured key."""
        try:
            await self._client.aio.models.get(model=self.model)
            return True
        except Exception:
            return False
