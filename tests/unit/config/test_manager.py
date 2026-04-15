from pathlib import Path

import pytest
import yaml

from src.config.manager import ConfigManager, Settings


class TestConfigManager:
    """Test suite for the ConfigManager class, verifying configuration loading and saving."""

    def test_loads_yaml(self, tmp_path):
        """
        Verify that ConfigManager correctly loads values from YAML settings and prompts files.

        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
        settings = tmp_path / "settings.yaml"
        settings.write_text("llm:\n  provider: gemini\n", encoding="utf-8")
        prompts = tmp_path / "prompts.yaml"
        prompts.write_text("system: 'Test prompt'\n", encoding="utf-8")

        cm = ConfigManager(settings_path=settings, prompts_path=prompts)
        assert cm.settings.llm.provider == "gemini"
        assert cm.prompts["system"] == "Test prompt"

    def test_defaults_without_yaml(self, tmp_path):
        """
        Verify that ConfigManager falls back to default values when YAML files are missing.

        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
        cm = ConfigManager(
            settings_path=tmp_path / "nonexistent.yaml",
            prompts_path=tmp_path / "nonexistent.yaml",
        )
        assert cm.settings.llm.provider == "ollama"
        assert cm.prompts == {}

    def test_nested_config(self, tmp_path):
        """
        Verify that nested configuration values are correctly parsed and defaults are preserved.

        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
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

    def test_save_prompts(self, tmp_path):
        """
        Verify that saving prompts updates the manager's state and writes to the prompts file.

        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
        settings = tmp_path / "settings.yaml"
        settings.write_text("", encoding="utf-8")
        prompts = tmp_path / "prompts.yaml"
        prompts.write_text("system: 'Original'\n", encoding="utf-8")

        cm = ConfigManager(settings_path=settings, prompts_path=prompts)
        assert cm.prompts["system"] == "Original"

        cm.save_prompts({"system": "Updated", "qa": "Q", "condense": "C", "no_context": "N"})

        assert cm.prompts["system"] == "Updated"
        assert cm.prompts["qa"] == "Q"
        # File was written
        reloaded = yaml.safe_load(prompts.read_text(encoding="utf-8"))
        assert reloaded["system"] == "Updated"

    def test_save_prompts_strips_meta(self, tmp_path):
        """
        Verify that saving prompts removes the '_meta' key before writing to the file.

        Args:
            tmp_path (Path): Pytest fixture providing a temporary directory.
        """
        settings = tmp_path / "settings.yaml"
        settings.write_text("", encoding="utf-8")
        prompts = tmp_path / "prompts.yaml"
        prompts.write_text("system: 'X'\n", encoding="utf-8")

        cm = ConfigManager(settings_path=settings, prompts_path=prompts)
        cm.save_prompts({"_meta": {"name": "test"}, "system": "S", "qa": "Q"})

        reloaded = yaml.safe_load(prompts.read_text(encoding="utf-8"))
        assert "_meta" not in reloaded
        assert reloaded["system"] == "S"


class TestSaveSettings:
    """Test suite for ConfigManager.save_settings — allow-list, validation, persistence, live mutation."""

    def _make_cm(self, tmp_path):
        """Build a ConfigManager backed by empty YAML files in tmp_path."""
        settings = tmp_path / "settings.yaml"
        settings.write_text("", encoding="utf-8")
        return ConfigManager(
            settings_path=settings,
            prompts_path=tmp_path / "p.yaml",
            sources_path=tmp_path / "s.yaml",
        ), settings

    def test_happy_path_writes_yaml_and_mutates_live(self, tmp_path):
        """A valid update should be persisted and visible on the running settings object."""
        cm, settings_path = self._make_cm(tmp_path)
        retrieval_ref = cm.settings.retrieval  # hold a reference to verify in-place mutation

        cm.save_settings({"retrieval": {"top_k": 22, "score_threshold": 0.66}})

        assert cm.settings.retrieval.top_k == 22
        assert cm.settings.retrieval.score_threshold == 0.66
        # Same object — services holding references stay consistent
        assert cm.settings.retrieval is retrieval_ref

        reloaded = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        assert reloaded["retrieval"]["top_k"] == 22
        assert reloaded["retrieval"]["score_threshold"] == 0.66

    def test_nested_update_mutates_nested_model_in_place(self, tmp_path):
        """Nested updates (reranking, hybrid) must mutate the nested model in place."""
        cm, _ = self._make_cm(tmp_path)
        rerank_ref = cm.settings.retrieval.reranking

        cm.save_settings({"retrieval": {"reranking": {"top_k_rerank": 3, "enabled": False}}})

        assert cm.settings.retrieval.reranking.top_k_rerank == 3
        assert cm.settings.retrieval.reranking.enabled is False
        assert cm.settings.retrieval.reranking is rerank_ref

    def test_disallowed_keys_are_dropped(self, tmp_path):
        """Keys outside the allow-list must not reach the live settings or disk."""
        cm, settings_path = self._make_cm(tmp_path)
        original_model = cm.settings.embeddings.model

        cm.save_settings({
            "embeddings": {"model": "evil/model"},           # full section disallowed
            "retrieval": {"top_k": 7, "secret_key": "nope"}, # unknown nested key
        })

        assert cm.settings.embeddings.model == original_model
        assert cm.settings.retrieval.top_k == 7
        reloaded = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        assert "embeddings" not in reloaded
        assert "secret_key" not in reloaded["retrieval"]

    def test_invalid_value_raises_and_does_not_persist(self, tmp_path):
        """Pydantic validation errors must bubble up and leave state untouched."""
        from pydantic import ValidationError

        cm, settings_path = self._make_cm(tmp_path)
        original_top_k = cm.settings.retrieval.top_k

        with pytest.raises(ValidationError):
            cm.save_settings({"retrieval": {"top_k": "not-an-int"}})

        assert cm.settings.retrieval.top_k == original_top_k
        # File should remain empty since validation failed before write
        assert settings_path.read_text(encoding="utf-8").strip() == ""

    def test_empty_update_is_noop(self, tmp_path):
        """Empty or fully-disallowed updates should not touch disk."""
        cm, settings_path = self._make_cm(tmp_path)
        mtime_before = settings_path.stat().st_mtime_ns

        snapshot = cm.save_settings({})
        snapshot2 = cm.save_settings({"api": {"port": 9999}})  # disallowed section

        assert snapshot["retrieval"]["top_k"] == cm.settings.retrieval.top_k
        assert snapshot == snapshot2
        assert settings_path.stat().st_mtime_ns == mtime_before

    def test_snapshot_excludes_secrets(self, tmp_path):
        """The editable snapshot must not contain API keys, file paths, or logging."""
        cm, _ = self._make_cm(tmp_path)
        snap = cm.editable_snapshot()

        assert "api_key_env" not in snap["llm"]["gemini"]
        assert "logging" not in snap
        assert "persist_directory" not in snap["vectorstore"]

    def test_preserves_unrelated_yaml_sections(self, tmp_path):
        """Existing YAML keys outside the update must survive the round-trip."""
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(
            "logging:\n  level: DEBUG\nretrieval:\n  top_k: 10\n",
            encoding="utf-8",
        )
        cm = ConfigManager(
            settings_path=settings_path,
            prompts_path=tmp_path / "p.yaml",
            sources_path=tmp_path / "s.yaml",
        )

        cm.save_settings({"retrieval": {"top_k": 42}})

        reloaded = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        assert reloaded["logging"]["level"] == "DEBUG"
        assert reloaded["retrieval"]["top_k"] == 42


class TestSettings:
    """Test suite for the Settings Pydantic model overrides and defaults."""

    def test_default_values(self):
        """Verify that Settings initializes with the correct default values."""
        s = Settings()
        assert s.llm.provider == "ollama"
        assert s.vectorstore.mode == "embedded"
        assert s.retrieval.top_k == 15

    def test_env_override(self, monkeypatch):
        """
        Verify that environment variables correctly override default Settings values.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock environment variables.
        """
        monkeypatch.setenv("LLM__PROVIDER", "gemini")
        s = Settings()
        assert s.llm.provider == "gemini"

    def test_chroma_mode_override(self, monkeypatch):
        """
        Verify that the Chroma mode can be overridden via environment variable.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest fixture to mock environment variables.
        """
        monkeypatch.setenv("VECTORSTORE__MODE", "client")
        s = Settings()
        assert s.vectorstore.mode == "client"
