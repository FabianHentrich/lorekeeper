from src.config.manager import ConversationConfig, LLMConfig
from .providers.base import BaseLLMProvider
from .providers.gemini import GeminiProvider
from .providers.ollama import OllamaProvider


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
        Returns None if no fallback provider is enabled or configured.
        """
        if not config.fallback_enabled or not config.fallback_provider:
            return None
        try:
            return _build(config.fallback_provider, config)
        except ValueError:
            return None
