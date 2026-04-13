import pytest
import yaml

from src.api.prompt_schemas import PromptSet
from src.api.prompt_routes import (
    _validate_templates,
    _prompts_dict_from_set,
    _read_variant_file,
)
from fastapi import HTTPException


_SAMPLE = PromptSet(
    system="System prompt",
    qa="{{ question }}",
    condense="{{ question }}",
    no_context="{{ question }}",
)


class TestValidateTemplates:
    def test_valid_templates(self):
        _validate_templates(_SAMPLE)

    def test_invalid_jinja2_raises_422(self):
        bad = PromptSet(
            system="{% if %}",
            qa="ok",
            condense="ok",
            no_context="ok",
        )
        with pytest.raises(HTTPException) as exc_info:
            _validate_templates(bad)
        assert exc_info.value.status_code == 422
        assert "system" in exc_info.value.detail

    def test_complex_template_valid(self):
        ps = PromptSet(
            system="Du bist ein Assistent.",
            qa="{% for c in chunks %}[{{ c.source_file }}]{% endfor %}\n{{ question }}",
            condense="{% for m in history %}{{ m.role }}{% endfor %}\n{{ question }}",
            no_context="Nichts gefunden: {{ question }}",
        )
        _validate_templates(ps)


class TestPromptsDictFromSet:
    def test_returns_all_keys(self):
        result = _prompts_dict_from_set(_SAMPLE)
        assert set(result.keys()) == {"system", "qa", "condense", "no_context"}
        assert result["system"] == "System prompt"


class TestReadVariantFile:
    def test_reads_yaml(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text(
            yaml.safe_dump({"system": "hello", "qa": "world"}),
            encoding="utf-8",
        )
        result = _read_variant_file(f)
        assert result["system"] == "hello"
        assert result["qa"] == "world"

    def test_empty_file_returns_empty_dict(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        assert _read_variant_file(f) == {}
