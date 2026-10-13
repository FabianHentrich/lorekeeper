"""Builds LLM providers from the config: the active one, the optional fallback and the optional condense
provider (always Ollama — same ``base_url``, different model).
"""
import logging

from src.config.manager import ConversationConfig, LLMConfig, Settings

from .generator import Generator
from .providers.base import BaseLLMProvider
from .providers.gemini import GeminiProvider
from .providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)


def _build(name: str, config: LLMConfig) -> BaseLLMProvider:
    """Instantiate a provider class based on its string identifier ("ollama" or "gemini")."""
    match name:
        case "ollama":
            return OllamaProvider(config=config.ollama)
        case "gemini":
            return GeminiProvider(config=config.gemini)
        case _:
            raise ValueError(f"Unknown LLM provider: {name}")


class ProviderFactory:
    """Factory handles the initialization of language model providers dynamically
    based on the current application configuration.
    """

    @staticmethod
    def build_all(settings: Settings) -> tuple[BaseLLMProvider, BaseLLMProvider | None, Generator]:
        """Active provider, optional condense provider and the Generator (with its fallback) — always built
        together: the Generator fixes its fallback at construction, so rebuilding only one of them left
        the others on old settings (roadmap 11.0 #2). Raises whatever building the active provider raises."""
        provider = ProviderFactory.create(settings.llm)
        condense = ProviderFactory.create_condense_provider(settings.llm, settings.conversation)
        generator = Generator(provider=provider, fallback_provider=ProviderFactory.create_fallback(settings.llm))
        return provider, condense, generator

    @staticmethod
    def create(config: LLMConfig) -> BaseLLMProvider:
        """Create the primary LLM provider instances as specified in the configuration."""
        return _build(config.provider, config)

    @staticmethod
    def create_condense_provider(
        llm_config: LLMConfig,
        conversation_config: ConversationConfig,
    ) -> BaseLLMProvider | None:
        """Creates a separate provider for condense (question rewriting).
        Returns None if condense_model is not set (main provider will be used).
        condense_model is always an Ollama model name in v1."""
        if not conversation_config.condense_model:
            return None

        condense_cfg = llm_config.ollama.model_copy(
            update={"model": conversation_config.condense_model},
        )
        return OllamaProvider(config=condense_cfg)

    @staticmethod
    def create_fallback(config: LLMConfig) -> BaseLLMProvider | None:
        """Create a backup LLM provider to be used if the primary provider crashes.
        Returns None if no fallback provider is enabled or configured, or if it can't be built (e.g. Gemini
        without a key) — logged, since the fallback toggle would otherwise silently do nothing.
        """
        if not config.fallback_enabled or not config.fallback_provider:
            return None
        try:
            return _build(config.fallback_provider, config)
        except ValueError as e:
            logger.warning("Fallback provider %s unavailable: %s", config.fallback_provider, e)
            return None
