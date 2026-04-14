import pytest

from src.prompts.manager import PromptManager


@pytest.fixture
def pm():
    """
    Provide a functionally mocked PromptManager instance populated with static inline templates.
    """
    return PromptManager(prompts_dict={
        "system": "Du bist ein Assistent.",
        "qa": "Quellen:\n{% for chunk in chunks %}[{{ chunk.source_file }}] {{ chunk.content }}\n{% endfor %}\nFrage: {{ question }}",
        "condense": "{% for msg in history %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}Frage: {{ question }}",
        "no_context": "Keine Ergebnisse für: {{ question }}",
    })


class TestPromptManager:
    """Test suite validating template rendering logic using the centralized Jinja2 environment."""

    def test_get_system_prompt(self, pm):
        """Verify the manager successfully surfaces the static system instruction."""
        assert pm.get_system_prompt() == "Du bist ein Assistent."

    def test_render_qa(self, pm):
        """Verify that standard QA contexts bind semantic chunks and questions successfully."""
        result = pm.render_qa(
            chunks=[{"source_file": "test.md", "content": "Inhalt"}],
            question="Was ist das?",
        )
        assert "[test.md] Inhalt" in result
        assert "Was ist das?" in result

    def test_render_condense(self, pm):
        """Verify history condensation formats dictionaries into conversational block styles."""
        result = pm.render_condense(
            history=[{"role": "user", "content": "Hallo"}, {"role": "assistant", "content": "Hi"}],
            question="Und weiter?",
        )
        assert "user: Hallo" in result
        assert "Und weiter?" in result

    def test_render_no_context(self, pm):
        """Verify the distinct fallback formatting used when search yields no matches."""
        result = pm.render_no_context(question="Testefrage")
        assert "Testefrage" in result

    def test_unknown_template_raises(self, pm):
        """Verify rendering unregistered template identifiers halts execution explicitly."""
        with pytest.raises(KeyError, match="nonexistent"):
            pm.render("nonexistent")

    def test_multiple_chunks(self, pm):
        """Verify loop-based rendering naturally cascades across unbounded lists of chunks."""
        chunks = [
            {"source_file": "a.md", "content": "Eins"},
            {"source_file": "b.md", "content": "Zwei"},
        ]
        result = pm.render_qa(chunks=chunks, question="Test")
        assert "[a.md] Eins" in result
        assert "[b.md] Zwei" in result
