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
    """Test suite ensuring runtime template updates are syntactically valid before application."""

    def test_valid_templates(self):
        """Verify properly formatted Jinja templates traverse validation without alerting."""
        _validate_templates(_SAMPLE)

    def test_invalid_jinja2_raises_422(self):
        """Verify broken semantic boundaries or loops raise distinct HTTP validation blockades."""
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
        """Verify advanced template operators traverse parsing without fault."""
        ps = PromptSet(
            system="Du bist ein Assistent.",
            qa="{% for c in chunks %}[{{ c.source_file }}]{% endfor %}\n{{ question }}",
            condense="{% for m in history %}{{ m.role }}{% endfor %}\n{{ question }}",
            no_context="Nichts gefunden: {{ question }}",
        )
        _validate_templates(ps)


class TestPromptsDictFromSet:
    """Test suite evaluating the schema conversions from robust objects back into base dictionaries."""

    def test_returns_all_keys(self):
        """Verify mapping covers all discrete template elements sequentially."""
        result = _prompts_dict_from_set(_SAMPLE)
        assert set(result.keys()) == {"system", "qa", "condense", "no_context"}
        assert result["system"] == "System prompt"


class TestReadVariantFile:
    """Test suite handling file-based variant loading."""

    def test_reads_yaml(self, tmp_path):
        """Verify structured extraction directly imports content configurations intact."""
        f = tmp_path / "test.yaml"
        f.write_text(
            yaml.safe_dump({"system": "hello", "qa": "world"}),
            encoding="utf-8",
        )
        result = _read_variant_file(f)
        assert result["system"] == "hello"
        assert result["qa"] == "world"

    def test_empty_file_returns_empty_dict(self, tmp_path):
        """Verify blank YAML inputs degrade safely avoiding syntax exceptions."""
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        assert _read_variant_file(f) == {}
