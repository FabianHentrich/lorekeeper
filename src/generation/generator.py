import logging
import time
from collections.abc import AsyncGenerator

from src.retrieval.retriever import RetrievedChunk
from .providers.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback_provider: BaseLLMProvider | None = None,
    ):
        self.provider = provider
        self.fallback_provider = fallback_provider
        self.last_usage: dict = {}

    async def generate(
        self,
        system_prompt: str,
        qa_prompt: str,
    ) -> LLMResponse:
        try:
            return await self.provider.generate(qa_prompt, system_prompt=system_prompt)
        except Exception as e:
            if self.fallback_provider:
                logger.warning(f"Primary provider failed ({e}), trying fallback")
                return await self.fallback_provider.generate(qa_prompt, system_prompt=system_prompt)
            raise

    async def generate_stream(
        self,
        system_prompt: str,
        qa_prompt: str,
    ) -> AsyncGenerator[str, None]:
        self.last_usage = {}
        try:
            async for token in self.provider.generate_stream(qa_prompt, system_prompt=system_prompt):
                yield token
            self.last_usage = dict(getattr(self.provider, "_last_stream_usage", {}) or {})
        except Exception as e:
            if self.fallback_provider:
                logger.warning(f"Primary provider failed ({e}), trying fallback")
                async for token in self.fallback_provider.generate_stream(qa_prompt, system_prompt=system_prompt):
                    yield token
                self.last_usage = dict(getattr(self.fallback_provider, "_last_stream_usage", {}) or {})
            else:
                raise

    async def condense_question(
        self,
        condense_prompt: str,
        condense_provider: BaseLLMProvider | None = None,
    ) -> str:
        provider = condense_provider or self.provider
        response = await provider.generate(condense_prompt)
        return response.content.strip()
