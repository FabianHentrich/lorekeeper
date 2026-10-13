"""Prompts page: saving, variants, activation, comparison and preview (roadmap 9.0 — page had 53 % coverage).

Everything the page writes goes to the backend; these tests check that the right request with the
right body goes out and that the user sees the outcome.
"""
import pytest

from tests.ui import conftest
from tests.ui.conftest import open_page

VARIANT = {"name": "Variante", "description": "kurz", "prompts": {**conftest._PROMPTS, "system": "Sei knapp."}}


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.fixture
def page(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/prompts/variants/Variante", VARIANT)
    monkeypatch.setitem(conftest.PAYLOADS, "/prompts/preview", {"rendered": "Frage: Was ist Arkenfeld?"})
    return _ok(open_page("prompts").run())


def test_saving_the_active_prompts_sends_the_edited_text(page, fake_backend):
    page.text_area(key="active_qa").input("Neu: {{ question }}")
    _ok(page.button(key="save_active").click().run())

    [body] = fake_backend.bodies["PUT /prompts/active"]
    assert body["prompts"] == {**conftest._PROMPTS, "qa": "Neu: {{ question }}"}
    assert any("gespeichert" in s.value for s in page.success)


def test_preview_renders_with_sample_data(page, fake_backend):
    _ok(page.button(key="preview_qa").click().run())

    [body] = fake_backend.bodies["POST /prompts/preview"]
    assert body["template_text"] == conftest._PROMPTS["qa"]
    assert body["sample_data"]["question"] == "Was ist Arkenfeld?" and body["sample_data"]["chunks"]
    assert "Frage: Was ist Arkenfeld?" in [c.value for c in page.code]


def test_active_prompts_can_be_saved_as_a_new_variant(page, fake_backend):
    page.text_input(key="var_save_name").input("Neue Variante")
    page.text_input(key="var_save_desc").input("Test")
    _ok(page.button(key="var_save_btn").click().run())

    [body] = fake_backend.bodies["PUT /prompts/variants/Neue%20Variante"]
    assert body == {"name": "Neue Variante", "description": "Test", "prompts": conftest._PROMPTS}


def test_saving_a_copy_needs_a_name(page, fake_backend):
    _ok(page.button(key="var_save_btn").click().run())
    assert "Name ist Pflichtfeld." in [e.value for e in page.error]
    assert not [c for c in fake_backend.calls if c.startswith("PUT /prompts/variants/")]


def test_editing_and_activating_a_variant(page, fake_backend):
    page.text_area(key="var_Variante_system").input("Sei sehr knapp.")
    _ok(page.button(key="save_Variante").click().run())
    [body] = fake_backend.bodies["PUT /prompts/variants/Variante"]
    assert body["prompts"]["system"] == "Sei sehr knapp." and body["description"] == "kurz"

    _ok(page.button(key="activate_Variante").click().run())
    assert "POST /prompts/activate/Variante" in fake_backend.calls


def test_deleting_a_variant(page, fake_backend):
    _ok(page.button(key="del_Variante").click().run())
    assert "DELETE /prompts/variants/Variante" in fake_backend.calls
    assert not page.error


def test_comparison_marks_the_templates_that_differ(page):
    page.selectbox(key="cmp_a").select("(Aktiv)")
    page.selectbox(key="cmp_b").select("Variante")
    _ok(page.button(key="cmp_run").click().run())

    headings = [m.value for m in page.markdown if m.value.startswith("### ")]
    assert "### System-Prompt ⚠ unterschiedlich" in headings
    assert "### QA-Template ✅ identisch" in headings
