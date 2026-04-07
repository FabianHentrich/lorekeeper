import pytest

from src.prompts.manager import PromptManager


@pytest.fixture
def pm():
    return PromptManager(prompts_dict={
        "system": "Du bist ein Assistent.",
        "qa": "Quellen:\n{% for chunk in chunks %}[{{ chunk.source_file }}] {{ chunk.content }}\n{% endfor %}\nFrage: {{ question }}",
        "condense": "{% for msg in history %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}Frage: {{ question }}",
        "no_context": "Keine Ergebnisse für: {{ question }}",
    })


class TestPromptManager:
    def test_get_system_prompt(self, pm):
        assert pm.get_system_prompt() == "Du bist ein Assistent."

    def test_render_qa(self, pm):
        result = pm.render_qa(
            chunks=[{"source_file": "test.md", "content": "Inhalt"}],
            question="Was ist das?",
        )
        assert "[test.md] Inhalt" in result
        assert "Was ist das?" in result

    def test_render_condense(self, pm):
        result = pm.render_condense(
            history=[{"role": "user", "content": "Hallo"}, {"role": "assistant", "content": "Hi"}],
            question="Und weiter?",
        )
        assert "user: Hallo" in result
        assert "Und weiter?" in result

    def test_render_no_context(self, pm):
        result = pm.render_no_context(question="Testefrage")
        assert "Testefrage" in result

    def test_unknown_template_raises(self, pm):
        with pytest.raises(KeyError, match="nonexistent"):
            pm.render("nonexistent")

    def test_multiple_chunks(self, pm):
        chunks = [
            {"source_file": "a.md", "content": "Eins"},
            {"source_file": "b.md", "content": "Zwei"},
        ]
        result = pm.render_qa(chunks=chunks, question="Test")
        assert "[a.md] Eins" in result
        assert "[b.md] Zwei" in result
