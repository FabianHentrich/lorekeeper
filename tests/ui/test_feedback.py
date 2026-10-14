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


def test_comment_field_appears_after_rating_and_is_sent_with_it(chat):
    """Roadmap 3.4: the comment is saved together with the current rating (the last entry counts),
    and a later rating change keeps it."""
    at = _ask(roles.PLAYER)
    msg_id = at.session_state["messages"][-1]["id"]
    assert f"fbc_{msg_id}" not in [t.key for t in at.text_input]   # no field before a rating

    at.feedback(key=f"fb_{msg_id}").set_value(0).run()
    at.text_input(key=f"fbc_{msg_id}").input("  Quelle fehlte  ").run()
    at.feedback(key=f"fb_{msg_id}").set_value(1).run()

    bodies = chat.bodies["POST /feedback"]
    assert [(b["rating"], b["comment"]) for b in bodies] == \
        [("down", ""), ("down", "Quelle fehlte"), ("up", "Quelle fehlte")]


def test_comment_field_is_bounded_like_the_api(chat):
    from src.api.schemas import MAX_FEEDBACK_COMMENT_CHARS
    at = _ask(roles.PLAYER)
    msg_id = at.session_state["messages"][-1]["id"]
    at.feedback(key=f"fb_{msg_id}").set_value(1).run()
    assert at.text_input(key=f"fbc_{msg_id}").proto.max_chars == MAX_FEEDBACK_COMMENT_CHARS


def test_withdrawn_rating_drops_the_comment(chat):
    at = _ask(roles.GAME_MASTER)
    msg_id = at.session_state["messages"][-1]["id"]
    at.feedback(key=f"fb_{msg_id}").set_value(0).run()
    at.text_input(key=f"fbc_{msg_id}").input("falsch").run()
    at.feedback(key=f"fb_{msg_id}").set_value(None).run()

    last = chat.bodies["POST /feedback"][-1]
    assert (last["rating"], last["comment"]) == (None, "")
    assert f"fbc_{msg_id}" not in [t.key for t in at.text_input]


def test_no_request_without_a_click(chat):
    """Rendering the thumbs must not cost a backend call (request budgets in test_backend_calls)."""
    at = _ask(roles.PLAYER)
    at.run()
    assert "POST /feedback" not in chat.calls


def test_errors_get_no_thumbs(chat, monkeypatch):
    monkeypatch.setattr(conftest, "_STREAM_EVENTS", [{"type": "error", "content": "Ollama down"}])
    at = _ask(roles.GAME_MASTER)
    assert len(at.feedback) == 0


# ─── Feedback tab on the Evaluation page ────────────────────────────────────────

import feedback  # noqa: E402  (ui/ is on sys.path via conftest)

UP = {"ts": "2026-10-12T10:00:00", "message_id": "a", "rating": "up", "role": "player",
      "question": "Was ist Überschaden?", "answer": "…", "sources": ["pnp_regelbuch.pdf", "pnp_regelbuch.pdf"]}
DOWN = {"ts": "2026-10-12T11:00:00", "message_id": "b", "rating": "down", "role": "game_master",
        "question": "Wer ist Kalen?", "answer": "…", "sources": ["NPCs/Falscher.md"]}


class TestPairsFromFeedback:
    def test_thumbs_up_keeps_its_sources(self):
        [pair] = feedback.pairs_from_feedback([UP], [])
        assert pair["id"] == "f001"
        assert (pair["expected_sources"], pair["source_type"], pair["category"]) == \
            (["pnp_regelbuch.pdf"], "pdf", "feedback")

    def test_thumbs_down_leaves_the_source_to_fill_in(self):
        [pair] = feedback.pairs_from_feedback([DOWN], [])
        assert pair["expected_sources"] == []
        assert "erwartete Quelle eintragen" in pair["notes"] and "NPCs/Falscher.md" in pair["notes"]

    def test_condensed_follow_up_is_taken_standalone(self):
        """'Und wer ist sein Vater?' is useless without the chat — the condensed question goes in."""
        follow_up = {**UP, "question": "Und wer ist sein Vater?", "search_query": "Wer ist der Vater von Kalen?"}
        [pair] = feedback.pairs_from_feedback([follow_up], [])
        assert pair["question"] == "Wer ist der Vater von Kalen?"
        assert "Und wer ist sein Vater?" in pair["notes"]

    def test_comment_goes_into_the_notes(self):
        [pair] = feedback.pairs_from_feedback([{**DOWN, "comment": "Kalen ist der Schmied"}], [])
        assert "Kommentar: „Kalen ist der Schmied“" in pair["notes"]

    def test_ids_continue_and_known_questions_are_skipped(self):
        existing = [{"id": "f001", "question": "Alt"}, {"id": "m001", "question": "Was ist Überschaden?"}]
        pairs = feedback.pairs_from_feedback([UP, DOWN, DOWN], existing)
        assert [(p["id"], p["question"]) for p in pairs] == [("f002", "Wer ist Kalen?")]


def test_tab_shows_thumbs_down_and_thumbs_up_on_request(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/feedback", [DOWN, UP])
    at = open_page("evaluation").run()
    assert not at.exception, [e.message for e in at.exception]

    table = at.dataframe[-1].value
    assert list(table["question"]) == ["Wer ist Kalen?"]

    at.toggle(key="fb_show_up").set_value(True).run()
    assert list(at.dataframe[-1].value["question"]) == ["Wer ist Kalen?", "Was ist Überschaden?"]


def test_tab_shows_the_comment(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/feedback", [{**DOWN, "comment": "Falsche Person"}])
    at = open_page("evaluation").run()
    assert list(at.dataframe[-1].value["comment"]) == ["Falsche Person"]
