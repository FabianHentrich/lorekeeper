"""Google Gemini provider (google-genai SDK) — the paid cloud option and fallback.

API key order: runtime override from the UI (process memory only, never written to disk) > environment >
``.env``. Requests are spaced to the free tier's 15 RPM and 429 answers are retried. Building the provider
without a key raises ``ValueError``.
"""
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


def _resolve_api_key(config: GeminiConfig) -> tuple[str, str]:
    """The Gemini API key and where it came from: runtime override (UI) > environment > .env.
    Returns ``("", "none")`` when there is none."""
    if _runtime_api_key:
        return _runtime_api_key, "runtime"
    key = os.environ.get(config.api_key_env) or dotenv_values(".env").get(config.api_key_env) or ""
    return key, ("env" if key else "none")


def get_api_key_status(config: GeminiConfig) -> dict:
    """Return whether a Gemini API key is available and where it came from.
    Never returns the key itself."""
    _, source = _resolve_api_key(config)
    return {"has_key": source != "none", "source": source}


class GeminiProvider(BaseLLMProvider):
    """LLM provider for Google Gemini, with built-in rate limiting and 429 retry logic."""

    provider = "gemini"

    def __init__(self, config: GeminiConfig):
        """Resolve the API key (runtime override > env > .env) and build the genai client."""
        self.config = config
        self.model = config.model

        api_key, _ = _resolve_api_key(config)
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

    def _build_config(self, system_prompt: str | None = None) -> types.GenerateContentConfig:
        """Per-request GenerateContentConfig; values are read per call so Settings-page edits apply live."""
        return types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_tokens,
            system_instruction=system_prompt,
            http_options=types.HttpOptions(timeout=self.config.timeout * 1000),
        )

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        """Produce a complete, non-streamed Gemini response with rate limiting and 429 retries."""
        gen_config = self._build_config(system_prompt or None)

        async def _call():
            """Single Gemini API invocation — wrapped so _retry_on_429 can retry it."""
            return await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_config,
            )

        response = await self._retry_on_429(_call)

        content = response.text or ""
        usage = {}
        if response.usage_metadata:
            usage["tokens_in"] = response.usage_metadata.prompt_token_count
            usage["tokens_out"] = response.usage_metadata.candidates_token_count

        return LLMResponse(
            content=content,
            model=self.model,
            provider="gemini",
            usage=usage,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        stream_result: StreamResult | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream Gemini tokens, updating ``stream_result`` with usage metadata as it arrives."""
        gen_config = self._build_config(system_prompt or None)

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
        except Exception as e:
            logger.debug("Gemini health check failed: %s: %s", type(e).__name__, e)
            return False
