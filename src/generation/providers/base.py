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


class BaseLLMProvider(ABC):
    provider: str = "unknown"

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        stream_result: StreamResult | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def health_check(self) -> bool: ...
