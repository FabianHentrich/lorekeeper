"""Integration tests for the FastAPI HTTP layer.

These tests do NOT spin up the real lifespan (no models, no ChromaDB, no LLM).
Instead, they build a bare FastAPI app with the router attached, then inject
fakes into ``src.main`` module globals — which ``routes._get_services()``
imports lazily on every call.

The goal is to verify the HTTP contract: request/response shapes, SSE event
structure, session_id propagation, error codes, and provider switching.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import AsyncGenerator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.main as main_module
from src.api.routes import router
from src.retrieval.retriever import RetrievedChunk


# ─── Fakes ──────────────────────────────────────────────────────────────────────

@dataclass
class FakeMessage:
    """Mock stub for memory-based message history elements."""
    role: str
    content: str


@dataclass
class FakeSession:
    """Mock stub indicating a single chat session scope."""
    session_id: str
    messages: list[FakeMessage] = field(default_factory=list)
    usage_totals: dict = field(default_factory=lambda: {
        "tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0,
    })

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(FakeMessage(role=role, content=content))

    def add_usage(self, usage: dict) -> None:
        for k in ("tokens_in", "tokens_out", "tokens_thinking"):
            self.usage_totals[k] += int(usage.get(k, 0) or 0)


class FakeConversationManager:
    """Mock implementation of session history persistence."""
    def __init__(self) -> None:
        self._sessions: dict[str, FakeSession] = {}
        self.config = SimpleNamespace(condense_question=False)

    def get_or_create_session(self, session_id: str | None) -> FakeSession:
        sid = session_id or "test-session-123"
        if sid not in self._sessions:
            self._sessions[sid] = FakeSession(session_id=sid)
        return self._sessions[sid]

    def get_session(self, session_id: str) -> FakeSession | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def get_history_for_condense(self, session_id: str) -> list:
        return []


class FakeLLMResponse:
    def __init__(self, content: str, model: str = "fake-model") -> None:
        self.content = content
        self.model = model


class FakeGenerator:
    """Mock stub that circumvents real calls to language model providers."""
    async def generate(self, system_prompt: str, qa_prompt: str) -> FakeLLMResponse:
        return FakeLLMResponse(content="Das ist eine Testantwort.")

    async def generate_stream(
        self, system_prompt: str, qa_prompt: str, stream_result=None
    ) -> AsyncGenerator[str, None]:
        for token in ["Das ", "ist ", "ein ", "Token-Stream."]:
            yield token

    async def condense_question(self, prompt: str, provider) -> str:
        return prompt


class FakePromptManager:
    """Mock stub bypassing template loading."""
    def get_system_prompt(self) -> str:
        return "Du bist ein hilfreicher Assistent."

    def render_qa(self, chunks, question: str) -> str:
        return f"Frage: {question}"

    def render_condense(self, history, question: str) -> str:
        return question

    def render_no_context(self, question: str) -> str:
        return "Keine relevanten Quellen gefunden."


class FakeRetriever:
    """Mock stub returning hardcoded relevance context."""
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or [
            RetrievedChunk(
                content="Arkenfeld ist ein Zeitmagier aus der Akademie.",
                source_file="Arkenfeld.md",
                document_type="markdown",
                heading="Arkenfeld",
                score=0.87,
                metadata={"source_path": "Lore/NPCs/Arkenfeld.md", "content_category": "npc"},
            )
        ]
        self.last_call: dict = {}

    async def retrieve(self, query: str, top_k=None, metadata_filters=None, top_k_rerank=None, max_per_source=None, hybrid=None):
        self.last_call = {
            "query": query,
            "top_k": top_k,
            "metadata_filters": metadata_filters,
            "top_k_rerank": top_k_rerank,
            "max_per_source": max_per_source,
            "hybrid": hybrid,
        }
        return self._chunks


class FakeVectorStore:
    """Mock stub bypassing ChromaDB."""
    def __init__(self, healthy: bool = True, chunk_count: int = 42) -> None:
        self._healthy = healthy
        self._count = chunk_count

    def health_check(self) -> bool:
        return self._healthy

    def count(self) -> int:
        return self._count


class FakeProvider:
    def __init__(self, healthy: bool = True, provider_name: str = "ollama") -> None:
        self._healthy = healthy
        self.provider = provider_name
        self.model = "qwen3:8b"

    async def health_check(self) -> bool:
        return self._healthy


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_services():
    """Inject fakes into ``src.main`` globals, yield handles, then restore."""
    originals = {
        "conversation_manager": main_module.conversation_manager,
        "generator": main_module.generator,
        "condense_provider": main_module.condense_provider,
        "prompt_manager": main_module.prompt_manager,
        "retriever": main_module.retriever,
        "vectorstore": main_module.vectorstore,
        "provider": main_module.provider,
        "config": main_module.config,
    }

    fake_config = SimpleNamespace(
        settings=SimpleNamespace(
            llm=SimpleNamespace(provider="ollama"),
            conversation=SimpleNamespace(),
            ingestion=SimpleNamespace(sources=[{"path": "/fake"}]),
        )
    )
    main_module.config = fake_config

    fakes = SimpleNamespace(
        conversation_manager=FakeConversationManager(),
        generator=FakeGenerator(),
        condense_provider=None,
        prompt_manager=FakePromptManager(),
        retriever=FakeRetriever(),
        vectorstore=FakeVectorStore(),
        provider=FakeProvider(),
    )

    main_module.conversation_manager = fakes.conversation_manager
    main_module.generator = fakes.generator
    main_module.condense_provider = fakes.condense_provider
    main_module.prompt_manager = fakes.prompt_manager
    main_module.retriever = fakes.retriever
    main_module.vectorstore = fakes.vectorstore
    main_module.provider = fakes.provider

    # Seed health cache (background loop doesn't run in tests)
    import time as _time
    from src.api import routes as routes_module
    routes_module._health_cache.update({"ts": _time.monotonic(), "chroma": True, "llm": True})
    routes_module._ingest_jobs.clear()

    yield fakes

    for name, value in originals.items():
        setattr(main_module, name, value)


@pytest.fixture
def client(fake_services):
    """A TestClient against a bare app (no lifespan) with the router attached."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ─── /query ────────────────────────────────────────────────────────────────────

def test_query_returns_answer_and_sources(client, fake_services):
    """Ensure standard query returns 200 OK and expected API payload schema."""
    response = client.post("/query", json={"question": "Wer ist Arkenfeld?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Das ist eine Testantwort."
    assert body["session_id"] == "test-session-123"
    assert body["model_used"] == "fake-model"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["file"] == "Arkenfeld.md"
    assert body["sources"][0]["score"] == pytest.approx(0.87)
    assert body["retrieval_scores"] == [pytest.approx(0.87)]
    assert body["latency_ms"] >= 0


def test_query_passes_metadata_filters_to_retriever(client, fake_services):
    """Ensure requested query constraints are passed unadulterated into the retrieval engine."""
    filters = {"content_category": {"$in": ["rules", "tool"]}}
    response = client.post(
        "/query",
        json={"question": "Was macht der Zeitmagier?", "metadata_filters": filters},
    )
    assert response.status_code == 200
    assert fake_services.retriever.last_call["metadata_filters"] == filters


def test_query_without_chunks_returns_no_context_message(client, fake_services):
    """Ensure the app falls back strictly to the 'no_context' prompt if semantic search finds nothing."""
    fake_services.retriever._chunks = []
    response = client.post("/query", json={"question": "Unbekanntes Thema?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Keine relevanten Quellen gefunden."
    assert body["sources"] == []
    assert body["model_used"] == "none"


def test_query_llm_failure_returns_503(client, fake_services):
    """Ensure that backend LLM crashes bubble up as 503 instead of silent failures."""
    async def _boom(*args, **kwargs):
        raise RuntimeError("Ollama down")

    fake_services.generator.generate = _boom  # type: ignore[method-assign]
    response = client.post("/query", json={"question": "egal"})
    assert response.status_code == 503
    assert "LLM nicht erreichbar" in response.json()["detail"]


def test_query_persists_messages_across_calls(client, fake_services):
    """Ensure the user's session identifier correctly retrieves the linked history payload."""
    r1 = client.post("/query", json={"question": "Erste Frage"})
    sid = r1.json()["session_id"]
    r2 = client.post("/query", json={"question": "Zweite Frage", "session_id": sid})
    assert r2.status_code == 200
    session = fake_services.conversation_manager.get_session(sid)
    # Two user + two assistant messages
    assert len(session.messages) == 4
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Erste Frage"
    assert session.messages[2].content == "Zweite Frage"


# ─── /query/stream ─────────────────────────────────────────────────────────────

def _parse_sse(raw: str) -> list[dict]:
    """Helper to convert raw Server-Sent Event text into dict objects."""
    events = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_query_stream_yields_tokens_and_done_event(client, fake_services):
    """Ensure generating streams provide granular JSON events ending with a structured summary."""
    with client.stream("POST", "/query/stream", json={"question": "Hi"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.read().decode("utf-8")

    events = _parse_sse(body)
    token_events = [e for e in events if e["type"] == "token"]
    done_events = [e for e in events if e["type"] == "done"]

    assert len(token_events) == 4
    assert "".join(e["content"] for e in token_events) == "Das ist ein Token-Stream."
    assert len(done_events) == 1

    done = done_events[0]
    assert done["session_id"] == "test-session-123"
    assert done["model_used"] == "qwen3:8b"
    assert len(done["sources"]) == 1
    assert done["sources"][0]["file"] == "Arkenfeld.md"


def test_query_stream_no_chunks_emits_fallback_token(client, fake_services):
    """Ensure missing context immediately yields the configured fallback response via stream."""
    fake_services.retriever._chunks = []
    with client.stream("POST", "/query/stream", json={"question": "???"}) as resp:
        body = resp.read().decode("utf-8")

    events = _parse_sse(body)
    assert any(
        e["type"] == "token" and "Keine relevanten Quellen" in e["content"]
        for e in events
    )
    assert any(e["type"] == "done" for e in events)


# ─── /health ───────────────────────────────────────────────────────────────────

def test_health_healthy(client, fake_services):
    """Ensure the health endpoint maps dependency statuses appropriately."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "healthy", "chromadb": True, "llm": True, "sources_configured": True}


def test_health_degraded_when_llm_down(client, fake_services):
    from src.api import routes as routes_module
    routes_module._health_cache["llm"] = False
    response = client.get("/health")
    assert response.json() == {"status": "degraded", "chromadb": True, "llm": False, "sources_configured": True}


# ─── /sessions ─────────────────────────────────────────────────────────────────

def test_get_session_not_found(client, fake_services):
    """Ensure invalid session lookups return a distinct 404."""
    response = client.get("/sessions/does-not-exist")
    assert response.status_code == 404


def test_delete_session_roundtrip(client, fake_services):
    """Verify standard CRUD session lifecycle interactions."""
    r1 = client.post("/query", json={"question": "Hallo"})
    sid = r1.json()["session_id"]

    r2 = client.get(f"/sessions/{sid}")
    assert r2.status_code == 200
    assert r2.json()["session_id"] == sid
    assert len(r2.json()["messages"]) == 2

    r3 = client.delete(f"/sessions/{sid}")
    assert r3.status_code == 200
    assert r3.json() == {"status": "deleted"}

    r4 = client.delete(f"/sessions/{sid}")
    assert r4.status_code == 404


# ─── /stats & /provider ────────────────────────────────────────────────────────

def test_stats(client, fake_services):
    """Ensure database counts are cleanly parsed and bubbled to the endpoint."""
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json() == {"chunk_count": 42}


def test_get_provider(client, fake_services):
    """Verify endpoint provides the active identity of the system's LLM engine."""
    response = client.get("/provider")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "ollama"
    assert body["model"] == "qwen3:8b"


def test_switch_provider_unknown_returns_400(client, fake_services):
    """Ensure trying to switch to an unimplemented provider model returns a 400 Bad Request."""
    response = client.post("/provider", json={"provider": "openai"})
    assert response.status_code == 400
    assert "Unbekannter Provider" in response.json()["detail"]


# ─── /ingest ───────────────────────────────────────────────────────────────────

def test_ingest_status_unknown_job_returns_404(client, fake_services):
    """Validate ingestion status endpoints reject unfamiliar background jobs."""
    response = client.get("/ingest/status/nonexistent-job-id")
    assert response.status_code == 404
