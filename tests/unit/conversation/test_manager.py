import pytest

from src.config.manager import ConversationConfig
from src.conversation.manager import ConversationManager


@pytest.fixture
def cm():
    return ConversationManager(ConversationConfig(window_size=3, session_timeout_minutes=1))


class TestConversationManager:
    def test_create_session(self, cm):
        session = cm.get_or_create_session()
        assert session.session_id
        assert len(session.messages) == 0

    def test_reuse_session(self, cm):
        s1 = cm.get_or_create_session()
        s2 = cm.get_or_create_session(s1.session_id)
        assert s1.session_id == s2.session_id

    def test_add_messages(self, cm):
        session = cm.get_or_create_session()
        session.add_message("user", "Hallo")
        session.add_message("assistant", "Hi!")
        assert len(session.messages) == 2

    def test_sliding_window(self, cm):
        session = cm.get_or_create_session()
        # Add 5 pairs (10 messages), window_size=3 → keep last 6
        for i in range(5):
            session.add_message("user", f"Frage {i}")
            session.add_message("assistant", f"Antwort {i}")

        history = cm.get_history(session.session_id)
        assert len(history) == 6  # 3 pairs
        assert history[0].content == "Frage 2"

    def test_get_history_for_condense(self, cm):
        session = cm.get_or_create_session()
        session.add_message("user", "Test")
        session.add_message("assistant", "Reply")

        history = cm.get_history_for_condense(session.session_id)
        assert history == [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Reply"},
        ]

    def test_delete_session(self, cm):
        session = cm.get_or_create_session()
        sid = session.session_id
        assert cm.delete_session(sid) is True
        assert cm.get_session(sid) is None
        assert cm.delete_session(sid) is False

    def test_nonexistent_session(self, cm):
        assert cm.get_session("nonexistent") is None
        assert cm.get_history("nonexistent") == []
