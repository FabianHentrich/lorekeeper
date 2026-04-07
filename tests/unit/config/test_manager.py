from pathlib import Path

import pytest

from src.config.manager import ConfigManager, Settings


class TestConfigManager:
    def test_loads_yaml(self, tmp_path):
        settings = tmp_path / "settings.yaml"
        settings.write_text("llm:\n  provider: gemini\n", encoding="utf-8")
        prompts = tmp_path / "prompts.yaml"
        prompts.write_text("system: 'Test prompt'\n", encoding="utf-8")

        cm = ConfigManager(settings_path=settings, prompts_path=prompts)
        assert cm.settings.llm.provider == "gemini"
        assert cm.prompts["system"] == "Test prompt"

    def test_defaults_without_yaml(self, tmp_path):
        cm = ConfigManager(
            settings_path=tmp_path / "nonexistent.yaml",
            prompts_path=tmp_path / "nonexistent.yaml",
        )
        assert cm.settings.llm.provider == "ollama"
        assert cm.prompts == {}

    def test_nested_config(self, tmp_path):
        settings = tmp_path / "settings.yaml"
        settings.write_text(
            "chunking:\n  strategy: recursive\n  max_chunk_size: 256\n",
            encoding="utf-8",
        )

        cm = ConfigManager(settings_path=settings, prompts_path=tmp_path / "p.yaml")
        assert cm.settings.chunking.strategy == "recursive"
        assert cm.settings.chunking.max_chunk_size == 256
        # Defaults still work
        assert cm.settings.chunking.chunk_overlap == 30


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.llm.provider == "ollama"
        assert s.vectorstore.mode == "embedded"
        assert s.retrieval.top_k == 15

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM__PROVIDER", "gemini")
        s = Settings()
        assert s.llm.provider == "gemini"

    def test_chroma_mode_override(self, monkeypatch):
        monkeypatch.setenv("VECTORSTORE__MODE", "client")
        s = Settings()
        assert s.vectorstore.mode == "client"
