from src.config.manager import LLMConfig, ConversationConfig, OllamaConfig
from .providers.base import BaseLLMProvider
from .providers.gemini import GeminiProvider
from .providers.ollama import OllamaProvider


class ProviderFactory:
    @staticmethod
    def create(config: LLMConfig) -> BaseLLMProvider:
        match config.provider:
            case "ollama":
                return OllamaProvider(config=config.ollama)
            case "gemini":
                return GeminiProvider(config=config.gemini)
            case _:
                raise ValueError(f"Unknown LLM provider: {config.provider}")

    @staticmethod
    def create_condense_provider(
        llm_config: LLMConfig,
        conversation_config: ConversationConfig,
    ) -> BaseLLMProvider | None:
        """Creates a separate provider for condense (question rewriting).
        Returns None if condense_model is not set (main provider will be used)."""
        if not conversation_config.condense_model:
            return None

        # condense_model is always an Ollama model name
        return OllamaProvider(
            base_url=llm_config.ollama.base_url,
            model=conversation_config.condense_model,
        )

    @staticmethod
    def create_fallback(config: LLMConfig) -> BaseLLMProvider | None:
        if not config.fallback_enabled or not config.fallback_provider:
            return None

        match config.fallback_provider:
            case "ollama":
                return OllamaProvider(config=config.ollama)
            case "gemini":
                return GeminiProvider(config=config.gemini)
            case _:
                return None
