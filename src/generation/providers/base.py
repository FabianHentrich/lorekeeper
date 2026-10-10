from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)


@dataclass
class StreamResult:
    """Per-call context for generate_stream. Populated by the provider while
    the stream runs so concurrent streams do not share state."""
    usage: dict = field(default_factory=dict)
    model: str | None = None  # model that actually produced the stream (set by Generator, fallback-aware)


class BaseLLMProvider(ABC):
    """Abstract interface every LLM backend (Ollama, Gemini, ...) must implement."""

    provider: str = "unknown"
    model: str  # set by each implementation in __init__ — no default: a missing one raises on first access

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Return a full, non-streamed completion for ``prompt``."""

    # Plain `def`: implementations are async generators, so calling one returns the
    # AsyncGenerator directly (an `async def` without yield would type it as a coroutine).
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        stream_result: StreamResult | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Yield tokens one by one; populate ``stream_result.usage`` when known."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and ready to serve requests."""
