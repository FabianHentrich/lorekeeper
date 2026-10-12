"""Evaluation → "Frage testen": the tested question can be added to the golden set (roadmap 7.4 A2).

The add button used to sit inside the "🔍 Testen" block, so its click reran the page with the
outer button False and the add never happened.
"""
import pytest

from tests.ui import conftest
from tests.ui.conftest import open_page

PREVIEW = {"chunks": [
    {"source_id": "vault", "source_file": "Orte/Arkenfeld.md", "document_type": "markdown",
     "heading": "Arkenfeld", "score": 0.9, "content_preview": "Arkenfeld ist eine Handelsstadt."},
], "latency_ms": 12.0}


@pytest.fixture
def tested(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/preview", PREVIEW)
    at = open_page("evaluation")
    at.run()
    at.text_input(key="test_question").set_value("Was ist Arkenfeld?").run()
    at.button(key="test_run").click().run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_tested_question_is_added_after_the_rerun(tested, fake_backend):
    tested.text_input(key="test_add_id_1").set_value("m099").run()
    tested.button(key="test_add_btn_1").click().run()

    assert not tested.exception, [e.message for e in tested.exception]
    [body] = fake_backend.bodies["PUT /eval/qa-pairs"]
    added = body["pairs"][-1]
    assert (added["id"], added["question"], added["expected_sources"]) == \
        ("m099", "Was ist Arkenfeld?", ["Orte/Arkenfeld.md"])
    assert any("m099" in s.value for s in tested.success)


def test_existing_id_is_refused(tested, fake_backend):
    tested.text_input(key="test_add_id_1").set_value("p001").run()   # p001 is in the fake golden set
    tested.button(key="test_add_btn_1").click().run()

    assert "PUT /eval/qa-pairs" not in fake_backend.calls
    assert any("gibt es im Golden Set schon" in e.value for e in tested.error)
