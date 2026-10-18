"""Chat state that must survive a page change, and answers that must not count as complete (roadmap 13.6).

Streamlit drops widget state when the user opens another page; the chat keeps copies in plain session
state. ``AppTest.switch_page`` reproduces the round trip Chat → Sources → Chat.
"""
import pytest

from tests.ui import conftest
from tests.ui.conftest import open_page


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


def _round_trip(at):
    """Chat → Sources → Chat, as a user clicking through the pages."""
    _ok(at.switch_page("pages/1_Sources.py").run())
    return _ok(at.switch_page("LoreKeeper.py").run())


@pytest.fixture
def chat(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/feedback", {"saved": True})
    return fake_backend


def test_rating_and_comment_survive_a_page_change(chat):
    """Re-rating after a page change sent an empty comment, which replaced the stored one."""
    at = _ok(open_page("chat").run())
    at.chat_input[0].set_value("Was ist eine Marke?").run()
    mid = at.session_state["messages"][-1]["id"]
    at.feedback(key=f"fb_{mid}").set_value(0).run()
    at.text_input(key=f"fbc_{mid}").input("Quelle fehlte")
    _ok(at.button(key=f"fbs_{mid}").click().run())

    at = _round_trip(at)
    assert at.feedback(key=f"fb_{mid}").value == 0
    assert at.text_input(key=f"fbc_{mid}").value == "Quelle fehlte"

    at.feedback(key=f"fb_{mid}").set_value(1).run()   # changing the thumb keeps the comment
    assert (chat.bodies["POST /feedback"][-1]["rating"], chat.bodies["POST /feedback"][-1]["comment"]) == \
        ("up", "Quelle fehlte")


def test_source_filter_and_hybrid_survive_a_page_change(chat):
    """A narrowed filter silently went back to "everything" after visiting another page."""
    at = _ok(open_page("chat").run())
    at.checkbox(key="src_🗺️ Lore").uncheck().run()
    at.checkbox(key="_hybrid_search_cb").check().run()

    at = _round_trip(at)
    assert at.checkbox(key="src_🗺️ Lore").value is False
    assert at.checkbox(key="_hybrid_search_cb").value is True

    at.chat_input[0].set_value("Wer ist Malek?").run()
    body = chat.bodies["POST /query/stream"][-1]
    assert body["hybrid_search"] is True
    assert body["metadata_filters"]["group"] == {"$in": ["adventure", "rules", "simulation"]}


def test_hybrid_box_starts_at_the_configured_default_and_can_switch_it_off(chat, monkeypatch):
    """With hybrid on in Settings the box started off and unticking it changed nothing (only "on" was sent)."""
    monkeypatch.setitem(conftest.PAYLOADS, "/sidebar-state",
                        {**conftest.PAYLOADS["/sidebar-state"], "hybrid_default": True})
    at = _ok(open_page("chat").run())
    assert at.checkbox(key="_hybrid_search_cb").value is True

    at.checkbox(key="_hybrid_search_cb").uncheck().run()
    at.chat_input[0].set_value("Wer ist Malek?").run()

    assert chat.bodies["POST /query/stream"][-1]["hybrid_search"] is False


def test_stream_without_done_is_an_error_with_retry(chat, monkeypatch):
    """Backend restarted mid-answer: the cut-off text was stored as an answer, rateable, without retry."""
    monkeypatch.setattr(conftest, "_STREAM_EVENTS", [{"type": "token", "content": "Eine Mar"}])
    at = _ok(open_page("chat").run())
    at.chat_input[0].set_value("Was ist eine Marke?").run()

    last = at.session_state["messages"][-1]
    assert last["content"].startswith("⚠️ Antwort abgebrochen") and last["retry"] == "Was ist eine Marke?"
    assert not last.get("id")   # no thumbs under it
    assert at.button(key="retry_last")


def test_interrupted_question_gets_a_retry(chat):
    """A page change while streaming left the question without answer and without retry."""
    at = open_page("chat")
    at.session_state["messages"] = [{"role": "user", "content": "Was ist eine Marke?"}]
    _ok(at.run())

    assert "⚠️ Keine Antwort — die Anfrage wurde unterbrochen." in [c.value for c in at.caption]
    _ok(at.button(key="retry_last").click().run())

    assert chat.bodies["POST /query/stream"][-1]["question"] == "Was ist eine Marke?"
    assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant"]
