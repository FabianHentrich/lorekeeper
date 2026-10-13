import logging
import re
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.config.manager import OllamaConfig

from .base import BaseLLMProvider, LLMResponse, StreamResult

logger = logging.getLogger(__name__)

# Qwen3 returns <think>...</think> blocks before the actual answer
THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from a complete response."""
    return THINK_PATTERN.sub("", text).strip()


class OllamaProvider(BaseLLMProvider):
    """LLM provider talking to a local Ollama daemon via its OpenAI-compatible API."""

    provider = "ollama"

    def __init__(self, config: OllamaConfig):
        """Store the config reference and build the AsyncOpenAI client pointed at Ollama.

        Generation parameters (temperature/top_p/max_tokens/timeout) are read
        from ``self.config`` on each call, so runtime mutation of the shared
        OllamaConfig instance propagates without rebuilding the provider.
        base_url and model are captured at init — they require a provider
        rebuild to change.
        """
        self.config = config
        self.base_url = config.base_url
        self.model = config.model

        self._client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="ollama",
        )

    def _is_qwen3(self) -> bool:
        """True when the configured model is a Qwen3 variant (thinking is switched off)."""
        return "qwen3" in self.model.lower()

    def _thinking_off(self) -> dict:
        """Request body extension that disables Qwen3 reasoning.

        Ollama ignores the old "/no_think" prompt suffix and streams reasoning in a
        separate ``reasoning`` field (~7 s before the first answer token);
        ``reasoning_effort: "none"`` switches it off, ``think: false`` does not.
        """
        return {"reasoning_effort": "none"} if self._is_qwen3() else {}

    async def generate(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        """Produce a complete, non-streamed response and return it wrapped in an LLMResponse."""
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,   # read per call: Settings-page edits apply live
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            stream=False,
            extra_body=self._thinking_off(),
        )

        content = _strip_thinking(response.choices[0].message.content or "")
        usage = {}
        if response.usage:
            usage = {
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=content,
            model=self.model,
            provider="ollama",
            usage=usage,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        stream_result: StreamResult | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens incrementally, filtering Qwen3 <think> blocks that may span chunks.

        Uses a rolling buffer so <think>...</think> markers split across chunks
        (e.g. "<th" + "ink>") are still detected. Token-usage data from the final
        chunk is written back into ``stream_result`` when provided.
        """
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=self._thinking_off(),
        )

        # Buffer-based <think>…</think> filter: robust against tags split
        # across tokens ("<thi" + "nk>") and against real content appearing
        # before/after the tag in the same token.
        buffer = ""
        in_think = False
        TAG_OPEN = "<think>"
        TAG_CLOSE = "</think>"
        MAX_PARTIAL = max(len(TAG_OPEN), len(TAG_CLOSE)) - 1
        # Reasoning Ollama still sends in the separate field (thinking models other
        # than Qwen3, or a server that ignores reasoning_effort) — counted, never shown
        reasoning_chunks = 0

        async for chunk in stream:
            if chunk.choices:
                extra = getattr(chunk.choices[0].delta, "model_extra", None)
                if isinstance(extra, dict) and extra.get("reasoning"):
                    reasoning_chunks += 1
            usage = getattr(chunk, "usage", None)
            if usage and stream_result is not None:
                stream_result.usage = {
                    "tokens_in": usage.prompt_tokens or 0,
                    "tokens_out": usage.completion_tokens or 0,
                    "tokens_thinking": reasoning_chunks,
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
        """Return True if the Ollama daemon responds and the configured model is pulled.

        A running daemon without the model would report healthy while every query fails.
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                names = {m.get("name", "") for m in resp.json().get("models", [])}
        except Exception:
            return False
        # Ollama stores an untagged name ("qwen3") as "qwen3:latest"
        available = self.model in names or f"{self.model}:latest" in names
        if not available:
            logger.warning(f"Ollama model '{self.model}' is not pulled — run: ollama pull {self.model}")
        return available
