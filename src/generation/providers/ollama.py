import logging
import re
import time
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from src.config.manager import OllamaConfig
from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Qwen3 returns <think>...</think> blocks before the actual answer
THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return THINK_PATTERN.sub("", text).strip()


class OllamaProvider(BaseLLMProvider):
    provider = "ollama"

    def __init__(self, config: OllamaConfig | None = None, base_url: str | None = None, model: str | None = None):
        if config:
            self.base_url = base_url or config.base_url
            self.model = model or config.model
            self.temperature = config.temperature
            self.top_p = config.top_p
            self.max_tokens = config.max_tokens
            self.timeout = config.timeout
        else:
            self.base_url = base_url or "http://localhost:11434"
            self.model = model or "qwen3:8b"
            self.temperature = 0.3
            self.top_p = 0.9
            self.max_tokens = 1024
            self.timeout = 120

        self._client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",
            timeout=300,  # Generous timeout for slow local models
        )
        self._last_stream_usage: dict = {}

    def _is_qwen3(self) -> bool:
        return "qwen3" in self.model.lower()

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Disable Qwen3 thinking mode for faster responses
        user_content = prompt + " /no_think" if self._is_qwen3() else prompt
        messages.append({"role": "user", "content": user_content})

        start = time.time()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=False,
        )
        latency_ms = (time.time() - start) * 1000

        content = _strip_thinking(response.choices[0].message.content or "")
        usage = {}
        if response.usage:
            usage = {
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
                "latency_ms": latency_ms,
            }

        return LLMResponse(
            content=content,
            model=self.model,
            provider="ollama",
            usage=usage,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def generate_stream(self, prompt: str, system_prompt: str = "", **kwargs) -> AsyncGenerator[str, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = prompt + " /no_think" if self._is_qwen3() else prompt
        messages.append({"role": "user", "content": user_content})

        self._last_stream_usage = {}
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
            stream_options={"include_usage": True},
        )

        in_think_block = False
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                self._last_stream_usage = {
                    "tokens_in": chunk.usage.prompt_tokens or 0,
                    "tokens_out": chunk.usage.completion_tokens or 0,
                    "tokens_thinking": 0,
                }
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content

                # Filter out <think>...</think> blocks from stream
                if "<think>" in token:
                    in_think_block = True
                    continue
                if "</think>" in token:
                    in_think_block = False
                    continue
                if in_think_block:
                    continue

                yield token

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
