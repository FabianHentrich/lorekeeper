import pytest

import src.generation.provider_factory as factory_module
from src.config.manager import ConversationConfig, LLMConfig
from src.generation.provider_factory import ProviderFactory


class _FakeProvider:
    def __init__(self, config):
        self.config = config


@pytest.fixture(autouse=True)
def fake_providers(monkeypatch):
    """Replace real provider classes — no network clients or API keys needed."""
    ollama = type("FakeOllama", (_FakeProvider,), {})
    gemini = type("FakeGemini", (_FakeProvider,), {})
    monkeypatch.setattr(factory_module, "OllamaProvider", ollama)
    monkeypatch.setattr(factory_module, "GeminiProvider", gemini)
    return ollama, gemini


class TestCreate:
    """Test suite for primary provider creation."""

    def test_ollama(self, fake_providers):
        ollama, _ = fake_providers
        cfg = LLMConfig(provider="ollama")
        p = ProviderFactory.create(cfg)
        assert isinstance(p, ollama)
        assert p.config is cfg.ollama

    def test_gemini(self, fake_providers):
        _, gemini = fake_providers
        cfg = LLMConfig(provider="gemini")
        p = ProviderFactory.create(cfg)
        assert isinstance(p, gemini)
        assert p.config is cfg.gemini

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            ProviderFactory.create(LLMConfig(provider="openai"))


class TestCondenseProvider:
    """Test suite for the optional second (condense) provider — always Ollama."""

    def test_none_without_condense_model(self):
        assert ProviderFactory.create_condense_provider(LLMConfig(), ConversationConfig()) is None

    def test_ollama_with_overridden_model(self, fake_providers):
        """Verify the condense provider reuses the Ollama base_url but swaps the model."""
        ollama, _ = fake_providers
        llm = LLMConfig(provider="gemini")
        p = ProviderFactory.create_condense_provider(llm, ConversationConfig(condense_model="qwen3:4b"))
        assert isinstance(p, ollama)
        assert p.config.model == "qwen3:4b"
        assert p.config.base_url == llm.ollama.base_url
        assert llm.ollama.model != "qwen3:4b"  # original config untouched


class TestFallback:
    """Test suite for fallback provider creation."""

    def test_disabled(self):
        assert ProviderFactory.create_fallback(LLMConfig(fallback_enabled=False)) is None

    def test_enabled_without_provider(self):
        assert ProviderFactory.create_fallback(
            LLMConfig(fallback_enabled=True, fallback_provider=None)
        ) is None

    def test_enabled_builds_provider(self, fake_providers):
        _, gemini = fake_providers
        p = ProviderFactory.create_fallback(LLMConfig(fallback_enabled=True, fallback_provider="gemini"))
        assert isinstance(p, gemini)

    def test_unknown_fallback_returns_none(self):
        assert ProviderFactory.create_fallback(
            LLMConfig(fallback_enabled=True, fallback_provider="openai")
        ) is None



class TestBuildAll:
    """Provider, condense provider and generator are built together (roadmap 11.0 #2)."""

    def test_generator_gets_the_configured_fallback(self, fake_providers):
        from src.config.manager import Settings
        ollama, gemini = fake_providers
        settings = Settings(llm=LLMConfig(provider="ollama", fallback_enabled=True, fallback_provider="gemini"))

        provider, condense, generator = ProviderFactory.build_all(settings)

        assert isinstance(provider, ollama) and condense is None
        assert generator.provider is provider
        assert isinstance(generator.fallback_provider, gemini)

    def test_unbuildable_fallback_is_logged(self, monkeypatch, caplog):
        """Enabling the fallback without a usable provider must not fail silently."""
        def no_key(config):
            raise ValueError("No Gemini API key")

        monkeypatch.setattr(factory_module, "GeminiProvider", no_key)
        fallback = ProviderFactory.create_fallback(LLMConfig(fallback_enabled=True, fallback_provider="gemini"))

        assert fallback is None
        assert "Fallback provider gemini unavailable: No Gemini API key" in caplog.text
