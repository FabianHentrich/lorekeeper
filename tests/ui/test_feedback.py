"""Thumbs up/down under chat answers (roadmap 3.4) — players and game masters can rate."""
import pytest
import roles

from tests.ui import conftest
from tests.ui.conftest import open_page


@pytest.fixture
def chat(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/feedback", {"saved": True})
    return fake_backend


def _ask(role):
    at = open_page("chat", role=role).run()
    at.chat_input[0].set_value("Was ist eine Marke?").run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize("role", [roles.PLAYER, roles.GAME_MASTER])
def test_rating_an_answer_posts_it_with_question_and_role(chat, role):
    at = _ask(role)
    answer = at.session_state["messages"][-1]

    at.feedback(key=f"fb_{answer['id']}").set_value(0).run()

    body = chat.bodies["POST /feedback"][-1]
    assert (body["message_id"], body["rating"], body["role"]) == (answer["id"], "down", role)
    assert body["question"] == "Was ist eine Marke?"
    assert body["answer"] == "Eine Marke ist eine Ressource."
    assert body["model"] == "qwen3:14b"


def test_withdrawing_a_rating_is_sent_as_none(chat):
    at = _ask(roles.GAME_MASTER)
    fb = at.feedback(key=f"fb_{at.session_state['messages'][-1]['id']}")
    fb.set_value(1).run()
    at.feedback(key=fb.key).set_value(None).run()

    assert [b["rating"] for b in chat.bodies["POST /feedback"]] == ["up", None]


def test_no_request_without_a_click(chat):
    """Rendering the thumbs must not cost a backend call (request budgets in test_backend_calls)."""
    at = _ask(roles.PLAYER)
    at.run()
    assert "POST /feedback" not in chat.calls


def test_errors_get_no_thumbs(chat, monkeypatch):
    monkeypatch.setattr(conftest, "_STREAM_EVENTS", [{"type": "error", "content": "Ollama down"}])
    at = _ask(roles.GAME_MASTER)
    assert len(at.feedback) == 0
