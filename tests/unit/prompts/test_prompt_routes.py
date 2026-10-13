import pytest
import yaml
from fastapi import HTTPException

from src.api.prompt_routes import (
    _read_variant_file,
    _templates,
    _validate_templates,
    _variant_from_raw,
)
from src.api.prompt_schemas import PromptSet

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


class TestTemplatesAndVariants:
    """Loaded prompts files → the four templates / a PromptVariant."""

    def test_templates_fill_missing_keys_and_drop_meta(self):
        result = _templates({"_meta": {"name": "x"}, "system": "S", "qa": "Q"})
        assert result == {"system": "S", "qa": "Q", "condense": "", "no_context": ""}

    def test_variant_takes_name_from_meta_or_default(self):
        raw = {**_SAMPLE.model_dump(), "_meta": {"name": "Knapp", "description": "kurz"}}
        v = _variant_from_raw(raw, "datei")
        assert (v.name, v.description, v.prompts) == ("Knapp", "kurz", _SAMPLE)
        assert _variant_from_raw(_SAMPLE.model_dump(), "datei").name == "datei"


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
