import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.config.manager import ConversationConfig

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Session:
    """Represents an active or cached chat session with history and token usage tracking."""
    session_id: str
    messages: list[Message] = field(default_factory=list)
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    usage_totals: dict = field(default_factory=lambda: {
        "tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0,
    })

    def add_message(self, role: str, content: str):
        """Append a new user or assistant message to the session's chat history
        and update the last active timestamp.
        """
        self.messages.append(Message(role=role, content=content))
        self.last_active = datetime.now(timezone.utc)

    def add_usage(self, usage: dict):
        """Increment the session-wide token usage counters by the values provided."""
        for k in ("tokens_in", "tokens_out", "tokens_thinking"):
            self.usage_totals[k] += int(usage.get(k, 0) or 0)


class ConversationManager:
    """Manages chat sessions across multiple users, including creation, retrieval,
    and automatic garbage collection (GC) of expired sessions from memory.
    """
    def __init__(self, config: ConversationConfig):
        self.config = config
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str | None = None) -> Session:
        """Find an existing session by ID or create a new one.

        If a session ID is not provided, a new random UUID is generated.
        When an existing session is fetched, its last active timestamp is reset to now.
        """
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_active = datetime.now(timezone.utc)
            return session

        new_id = session_id or str(uuid.uuid4())
        session = Session(session_id=new_id)
        self._sessions[new_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve an existing session without modifying its last active timestamp."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Manually remove a session from memory immediately."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def get_history(self, session_id: str) -> list[Message]:
        """Fetch the recent chat history for a given session.

        This trims the list to respect the configured `window_size` (which limits
        the maximum number of recent QA pairs to return) preventing infinite context growth.
        """
        session = self._sessions.get(session_id)
        if not session:
            return []

        messages = session.messages
        window = self.config.window_size * 2  # pairs -> individual messages

        if len(messages) > window:
            messages = messages[-window:]

        return messages

    def get_history_for_condense(self, session_id: str) -> list[dict[str, str]]:
        """Returns history formatted for the condense prompt template.

        This reformats the active `window_size` Message objects into a plain dictionary
        format {"role": ..., "content": ...} required by Jinja templating.
        """
        messages = self.get_history(session_id)
        return [{"role": m.role, "content": m.content} for m in messages]

    async def start_gc(self):
        """Background asyncio loop that periodically purges old, inactive sessions.

        Intended to run as a long-lived task initialized during application startup.
        Will sleep for the configured GC interval between cleanups.
        """
        logger.info("Session GC started")
        try:
            while True:
                await asyncio.sleep(self.config.session_gc_interval_seconds)
                self._cleanup_expired()
        except asyncio.CancelledError:
            logger.info("Session GC cancelled")
            raise

    def _cleanup_expired(self):
        """Synchronous implementation of the GC loop deleting old sessions.

        Iterates over the dict deciding which entries have eclipsed the `session_timeout_minutes`
        limit and evicts them from the manager's memory.
        """
        now = datetime.now(timezone.utc)
        timeout = self.config.session_timeout_minutes * 60

        expired = [
            sid for sid, session in self._sessions.items()
            if (now - session.last_active).total_seconds() > timeout
        ]

        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.info(f"GC removed {len(expired)} expired sessions")
