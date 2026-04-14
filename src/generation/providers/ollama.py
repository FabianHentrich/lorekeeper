import logging
import re
import time
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from src.config.manager import OllamaConfig
from .base import BaseLLMProvider, LLMResponse, StreamResult

logger = logging.getLogger(__name__)

# Qwen3 returns <think>...</think> blocks before the actual answer
THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return THINK_PATTERN.sub("", text).strip()


class OllamaProvider(BaseLLMProvider):
    provider = "ollama"

    def __init__(self, config: OllamaConfig):
        self.base_url = config.base_url
        self.model = config.model
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout

        self._client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",
            timeout=300,  # Generous timeout for slow local models
        )

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

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        stream_result: StreamResult | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = prompt + " /no_think" if self._is_qwen3() else prompt
        messages.append({"role": "user", "content": user_content})

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
            stream_options={"include_usage": True},
        )

        # Buffer-based <think>…</think> filter: robust against tags split
        # across tokens ("<thi" + "nk>") and against real content appearing
        # before/after the tag in the same token.
        buffer = ""
        in_think = False
        TAG_OPEN = "<think>"
        TAG_CLOSE = "</think>"
        MAX_PARTIAL = max(len(TAG_OPEN), len(TAG_CLOSE)) - 1

        async for chunk in stream:
            if getattr(chunk, "usage", None) and stream_result is not None:
                stream_result.usage = {
                    "tokens_in": chunk.usage.prompt_tokens or 0,
                    "tokens_out": chunk.usage.completion_tokens or 0,
                    "tokens_thinking": 0,
                }
            if not (chunk.choices and chunk.choices[0].delta.content):
                continue

            buffer += chunk.choices[0].delta.content
            out = ""

            while buffer:
                if in_think:
                    idx = buffer.find(TAG_CLOSE)
                    if idx == -1:
                        # keep only trailing bytes that could start the close tag
                        buffer = buffer[-MAX_PARTIAL:] if len(buffer) > MAX_PARTIAL else buffer
                        break
                    buffer = buffer[idx + len(TAG_CLOSE):]
                    in_think = False
                else:
                    idx = buffer.find(TAG_OPEN)
                    if idx == -1:
                        # Flush everything except a possible partial tag at the tail
                        if len(buffer) > MAX_PARTIAL:
                            out += buffer[:-MAX_PARTIAL]
                            buffer = buffer[-MAX_PARTIAL:]
                        break
                    out += buffer[:idx]
                    buffer = buffer[idx + len(TAG_OPEN):]
                    in_think = True

            if out:
                yield out

        # Flush any tail bytes that survived (no tag match possible now)
        if buffer and not in_think:
            yield buffer

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
