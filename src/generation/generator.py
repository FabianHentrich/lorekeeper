import logging
from collections.abc import AsyncGenerator

from .providers.base import BaseLLMProvider, LLMResponse, StreamResult

logger = logging.getLogger(__name__)


class Generator:
    """Manages interaction with Language Learning Model (LLM) providers to generate answers.

    This class wraps raw LLM providers (like Ollama or Gemini), handles fallback logic
    if the primary provider fails, and provides streaming and synchronous interfaces
    for question answering and question condensation (for follow-up context).
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        fallback_provider: BaseLLMProvider | None = None,
    ):
        self.provider = provider
        self.fallback_provider = fallback_provider

    async def generate(
        self,
        system_prompt: str,
        qa_prompt: str,
    ) -> LLMResponse:
        """Generate a complete, synchronous response from the configured LLM provider.

        Attempts to use the primary provider. If an exception occurs and a fallback
        provider is configured, it will automatically attempt to retry using the fallback.
        """
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
        stream_result: StreamResult | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated response tokens back as they are produced by the LLM.

        Yields tokens incrementally to be sent via Server-Sent Events (SSE). If the primary
        provider crashes mid-stream or during initialization, it intercepts the error and
        attempts to fail over to the fallback provider if one is set.
        """
        try:
            async for token in self.provider.generate_stream(
                qa_prompt,
                system_prompt=system_prompt,
                stream_result=stream_result,
            ):
                yield token
        except Exception as e:
            if self.fallback_provider:
                logger.warning(f"Primary provider failed ({e}), trying fallback")
                async for token in self.fallback_provider.generate_stream(
                    qa_prompt,
                    system_prompt=system_prompt,
                    stream_result=stream_result,
                ):
                    yield token
            else:
                raise

    async def condense_question(
        self,
        condense_prompt: str,
        condense_provider: BaseLLMProvider | None = None,
    ) -> str:
        """Rewrite a follow-up question into a standalone query using conversation history.

        This abstracts the context so that the standalone question can be embedded
        and searched against the vector store effectively without losing pronoun references.
        """
        provider = condense_provider or self.provider
        response = await provider.generate(condense_prompt)
        return response.content.strip()
