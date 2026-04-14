import pytest

from src.config.manager import ConversationConfig
from src.conversation.manager import ConversationManager


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
