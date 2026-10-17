"""Chat: "Erneut senden" under an error (roadmap 14.1) — the failed question goes out again without retyping."""
from tests.ui import conftest
from tests.ui.conftest import open_page

QUESTION = "Was ist eine Marke?"


def _ask_and_fail(fake_backend, monkeypatch, *, backend_down: bool):
    if backend_down:
        fake_backend.fail_paths.add("/query/stream")
    else:
        monkeypatch.setattr(conftest, "_STREAM_EVENTS", [{"type": "error", "content": "Ollama lädt noch"}])
    at = open_page("chat").run()
    at.chat_input[0].set_value(QUESTION).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_backend_down_keeps_the_error_with_a_retry_button(fake_backend, monkeypatch):
    """Before 14.1 a connection error vanished with the next rerun; now it stays, with the button."""
    at = _ask_and_fail(fake_backend, monkeypatch, backend_down=True)
    at.run()   # any rerun — the error and the button must survive it

    assert at.session_state["messages"][-1]["content"].startswith("⚠️ Verbindung zum Backend fehlgeschlagen")
    assert at.button(key="retry_last")


def test_retry_sends_the_same_question_once_and_replaces_the_failed_exchange(fake_backend, monkeypatch):
    at = _ask_and_fail(fake_backend, monkeypatch, backend_down=True)
    fake_backend.fail_paths.clear()   # backend is back

    at.button(key="retry_last").click().run()

    assert [b["question"] for b in fake_backend.bodies["POST /query/stream"]] == [QUESTION, QUESTION]
    msgs = at.session_state["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", QUESTION), ("assistant", "Eine Marke ist eine Ressource.")]
    assert not [b for b in at.button if b.key == "retry_last"]     # an answer has no retry button


def test_provider_error_offers_retry(fake_backend, monkeypatch):
    at = _ask_and_fail(fake_backend, monkeypatch, backend_down=False)

    assert at.session_state["messages"][-1]["retry"] == QUESTION
    assert at.button(key="retry_last")
    assert len(at.feedback) == 0                                   # still no thumbs on an error


def test_only_the_last_error_has_the_button(fake_backend, monkeypatch):
    """Retrying an older failed question would reorder the chat — only the newest message gets one."""
    at = _ask_and_fail(fake_backend, monkeypatch, backend_down=True)
    fake_backend.fail_paths.clear()
    at.chat_input[0].set_value("Und was ist Überschaden?").run()

    assert at.session_state["messages"][1].get("retry") == QUESTION     # old error kept in the history
    assert not [b for b in at.button if b.key == "retry_last"]
