import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.config.manager import ConversationConfig

logger = logging.getLogger(__name__)

# Words that only make sense with the previous turn ("Was ist sein Ziel?", "Wie funktioniert das?")
_REFERENCE_WORDS = frozenset({
    "er", "ihn", "ihm", "sie", "ihr", "ihre", "ihren", "ihrem", "ihrer", "ihres", "ihnen",
    "sein", "seine", "seinen", "seinem", "seiner", "seines", "dessen", "deren",
    "dies", "diese", "diesen", "diesem", "dieser", "dieses",
    "dort", "dorthin", "dabei", "dazu", "davon", "damit", "darauf", "daran", "dafür",
    "dagegen", "darüber", "darunter", "darin", "daraus", "dahinter", "deshalb", "deswegen",
    "genannte", "genannten", "vorherige", "vorherigen", "letzte", "letzten",
})
_LEADING_CONNECTORS = frozenset({"und", "aber", "oder", "auch", "sowie"})
# "das"/"es" are articles or idioms ("Was ist das Schicksalsdeck?", "Gibt es...") — referential only at the end
_TRAILING_REFERENCES = frozenset({"das", "es"})
_WORD_RE = re.compile(r"[\wäöüß]+", re.IGNORECASE)


def needs_condense(question: str) -> bool:
    """Heuristic: does this question depend on the previous turn and need rewriting?

    False positives only cost an extra LLM call; standalone questions skip the
    rewrite so retrieval runs on exactly what the user typed.
    """
    words = _WORD_RE.findall(question.lower())
    if len(words) <= 2:
        return True
    if words[0] in _LEADING_CONNECTORS or words[-1] in _TRAILING_REFERENCES:
        return True
    return any(w in _REFERENCE_WORDS for w in words)


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Session:
    """Represents an active or cached chat session with history and token usage tracking."""
    session_id: str
    access_role: str = "game_master"  # sessions never cross roles — no game-master history in player context
    messages: list[Message] = field(default_factory=list)
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))
    usage_totals: dict = field(default_factory=lambda: {
        "tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0,
    })

    def add_message(self, role: str, content: str):
        """Append a message and bump ``last_active``."""
        self.messages.append(Message(role=role, content=content))
        self.last_active = datetime.now(UTC)

    def add_usage(self, usage: dict):
        """Increment the session-wide token usage counters by the values provided."""
        for k in ("tokens_in", "tokens_out", "tokens_thinking"):
            self.usage_totals[k] += int(usage.get(k, 0) or 0)


class ConversationManager:
    """In-memory registry of chat sessions with a background GC for idle entries."""
    def __init__(self, config: ConversationConfig):
        """Initialize an empty session store using the given conversation config."""
        self.config = config
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str | None = None, access_role: str = "game_master") -> Session:
        """Find an existing session by ID or create a new one.

        If a session ID is not provided, a new random UUID is generated.
        When an existing session is fetched, its last active timestamp is reset to now.
        A session of another role is never reused: the request gets a fresh session with a
        new id, so e.g. a player can't condense follow-ups against a game-master history.
        """
        existing = self._sessions.get(session_id) if session_id else None
        if existing and existing.access_role == access_role:
            existing.last_active = datetime.now(UTC)
            return existing

        new_id = session_id if session_id and not existing else str(uuid.uuid4())
        session = Session(session_id=new_id, access_role=access_role)
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
        """Return the last ``window_size`` QA pairs within ``max_context_tokens``, or [] if unknown."""
        session = self._sessions.get(session_id)
        if not session:
            return []

        messages = session.messages
        window = self.config.window_size * 2  # pairs -> individual messages

        if len(messages) > window:
            messages = messages[-window:]

        # Hard token limit: drop oldest pairs until the history fits (~3.5 chars/token for German)
        while messages and sum(len(m.content) for m in messages) / 3.5 > self.config.max_context_tokens:
            messages = messages[2:]

        return messages

    def get_history_for_condense(self, session_id: str) -> list[dict[str, str]]:
        """Same as ``get_history`` but as plain {role, content} dicts for Jinja templates."""
        messages = self.get_history(session_id)
        return [{"role": m.role, "content": m.content} for m in messages]

    def get_answer_history(self, session_id: str) -> list[dict[str, str]]:
        """The last ``answer_history_pairs`` QA pairs for the answer prompt, answers cut at
        ``answer_history_max_chars``. Kept short on purpose: it only resolves references to the
        previous answer ("fass das kürzer zusammen") — the sources stay the only facts."""
        session = self._sessions.get(session_id)
        pairs = self.config.answer_history_pairs
        if not session or pairs <= 0:
            return []
        limit = self.config.answer_history_max_chars
        history = []
        for m in session.messages[-2 * pairs:]:
            content = m.content
            if m.role == "assistant" and len(content) > limit:
                content = content[:limit].rstrip() + " […]"
            history.append({"role": m.role, "content": content})
        return history

    async def start_gc(self):
        """Long-lived task that evicts idle sessions every ``session_gc_interval_seconds``."""
        logger.info("Session GC started")
        try:
            while True:
                await asyncio.sleep(self.config.session_gc_interval_seconds)
                self._cleanup_expired()
        except asyncio.CancelledError:
            logger.info("Session GC cancelled")
            raise

    def _cleanup_expired(self):
        """Drop sessions whose ``last_active`` exceeds ``session_timeout_minutes``."""
        now = datetime.now(UTC)
        timeout = self.config.session_timeout_minutes * 60

        expired = [
            sid for sid, session in self._sessions.items()
            if (now - session.last_active).total_seconds() > timeout
        ]

        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.info(f"GC removed {len(expired)} expired sessions")
