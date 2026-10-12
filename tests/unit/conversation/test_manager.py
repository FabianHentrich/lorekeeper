import pytest

from src.config.manager import ConversationConfig
from src.conversation.manager import ConversationManager, needs_condense


@pytest.fixture
def cm():
    """
    Provide a configured ConversationManager instance for testing.

    Returns:
        ConversationManager: An instance configured with a window size of 3
                             and a session timeout of 1 minute.
    """
    return ConversationManager(ConversationConfig(window_size=3, session_timeout_minutes=1))


class TestConversationManager:
    """Test suite for ConversationManager, validating session lifecycle and message history."""

    def test_create_session(self, cm):
        """
        Verify that getting or creating a session without an ID creates a new, empty session.
        """
        session = cm.get_or_create_session()
        assert session.session_id
        assert len(session.messages) == 0

    def test_reuse_session(self, cm):
        """
        Verify that an existing session can be retrieved using its session ID.
        """
        s1 = cm.get_or_create_session()
        s2 = cm.get_or_create_session(s1.session_id)
        assert s1.session_id == s2.session_id

    def test_add_messages(self, cm):
        """
        Verify that messages can be correctly appended to a session's message list.
        """
        session = cm.get_or_create_session()
        session.add_message("user", "Hallo")
        session.add_message("assistant", "Hi!")
        assert len(session.messages) == 2

    def test_sliding_window(self, cm):
        """
        Verify that the conversation history respects the configured window size.
        When more messages are added than the window permits, older ones should be dropped.
        """
        session = cm.get_or_create_session()
        # Add 5 pairs (10 messages), window_size=3 → keep last 6
        for i in range(5):
            session.add_message("user", f"Frage {i}")
            session.add_message("assistant", f"Antwort {i}")

        history = cm.get_history(session.session_id)
        assert len(history) == 6  # 3 pairs
        assert history[0].content == "Frage 2"

    def test_history_respects_max_context_tokens(self):
        """
        Verify oldest pairs are dropped when the window still exceeds max_context_tokens.
        """
        cm = ConversationManager(ConversationConfig(window_size=8, max_context_tokens=100))
        session = cm.get_or_create_session()
        # Each pair ≈ 2 * 140 chars / 3.5 = 80 tokens → only the newest pair fits into 100
        for i in range(3):
            session.add_message("user", f"F{i}".ljust(140, "x"))
            session.add_message("assistant", f"A{i}".ljust(140, "x"))

        history = cm.get_history(session.session_id)
        assert [m.content[:2] for m in history] == ["F2", "A2"]

    def test_get_history_for_condense(self, cm):
        """
        Verify that the history formatted for condensation represents messages as dictionaries.
        """
        session = cm.get_or_create_session()
        session.add_message("user", "Test")
        session.add_message("assistant", "Reply")

        history = cm.get_history_for_condense(session.session_id)
        assert history == [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Reply"},
        ]

    def test_answer_history_is_the_last_pair(self, cm):
        """The answer prompt gets only the newest QA pair, independent of window_size."""
        session = cm.get_or_create_session()
        for i in range(3):
            session.add_message("user", f"Frage {i}")
            session.add_message("assistant", f"Antwort {i}")

        assert cm.get_answer_history(session.session_id) == [
            {"role": "user", "content": "Frage 2"},
            {"role": "assistant", "content": "Antwort 2"},
        ]

    def test_answer_history_cuts_long_messages(self):
        """Long answers and long (pasted) questions are both cut with a marker."""
        cm = ConversationManager(ConversationConfig(answer_history_pairs=2, answer_history_max_chars=10))
        session = cm.get_or_create_session()
        for i in range(2):
            session.add_message("user", f"Frage {i}".ljust(30, "?"))
            session.add_message("assistant", "A" * 30)

        history = cm.get_answer_history(session.session_id)
        assert len(history) == 4
        assert history[0]["content"] == "Frage 0???" + " […]"
        assert history[1]["content"] == "A" * 10 + " […]"

    def test_answer_history_off_or_unknown(self):
        """0 pairs switches it off; unknown sessions have no history."""
        cm = ConversationManager(ConversationConfig(answer_history_pairs=0))
        session = cm.get_or_create_session()
        session.add_message("user", "Frage")
        session.add_message("assistant", "Antwort")

        assert cm.get_answer_history(session.session_id) == []
        assert cm.get_answer_history("nonexistent") == []

    def test_delete_session(self, cm):
        """
        Verify that sessions can be successfully deleted, and deleting a non-existent
        session returns an appropriate False status.
        """
        session = cm.get_or_create_session()
        sid = session.session_id
        assert cm.delete_session(sid) is True
        assert cm.get_session(sid) is None
        assert cm.delete_session(sid) is False

    def test_nonexistent_session(self, cm):
        """
        Verify that requesting data for a nonexistent session ID safely returns None or
        an empty list.
        """
        assert cm.get_session("nonexistent") is None
        assert cm.get_history("nonexistent") == []


class TestNeedsCondense:
    """Heuristic deciding whether a follow-up is rewritten before retrieval."""

    @pytest.mark.parametrize("question", [
        "was ist überschaden",
        "Was ist das Schicksalsdeck?",
        "Gibt es eine Karte von Arkenfeld?",
        "Welche Fähigkeiten hat der Waldläufer?",
    ])
    def test_standalone_questions_skip_condense(self, question):
        """Self-contained questions — including 'das'/'es' as article or idiom — go to retrieval verbatim."""
        assert needs_condense(question) is False

    @pytest.mark.parametrize("question", [
        "Wer ist sein Vater?",
        "Welche Ausrüstung trägt er?",
        "Wie funktioniert das?",
        "Und der Mönch?",
        "Was steht dazu im Regelwerk?",
        "Warum?",
        "Mehr Details",
    ])
    def test_follow_ups_need_condense(self, question):
        """Pronouns, back-references, leading connectors and very short questions trigger the rewrite."""
        assert needs_condense(question) is True


class TestSessionRoles:
    """Sessions never cross roles — no game-master history behind a player's follow-up."""

    def test_same_role_reuses_the_session(self, cm):
        s = cm.get_or_create_session(access_role="player")
        s.add_message("user", "Was ist Überschaden?")
        again = cm.get_or_create_session(s.session_id, access_role="player")
        assert again is s

    def test_other_role_gets_a_fresh_session_with_a_new_id(self, cm):
        gm = cm.get_or_create_session(access_role="game_master")
        gm.add_message("user", "Wer ist Sol Invictus wirklich?")

        player = cm.get_or_create_session(gm.session_id, access_role="player")

        assert player.session_id != gm.session_id
        assert player.messages == [] and player.access_role == "player"
        assert cm.get_session(gm.session_id) is gm   # the game master's session is left intact

    def test_unknown_id_keeps_the_requested_id(self, cm):
        assert cm.get_or_create_session("abc", access_role="player").session_id == "abc"
