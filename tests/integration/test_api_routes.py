"""Integration tests for the FastAPI HTTP layer.

These tests do NOT spin up the real lifespan (no models, no ChromaDB, no LLM).
Instead, they build a bare FastAPI app with the router attached, then inject
fakes into ``src.main`` module globals — the route handlers read them from
``src.main`` on every call.

The goal is to verify the HTTP contract: request/response shapes, SSE event
structure, session_id propagation, error codes, and provider switching.
"""
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.main as main_module
from src.api.routes import router
from src.retrieval.retriever import RetrievedChunk
from tests.integration.conftest import parse_sse

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

    def get_or_create_session(self, session_id: str | None, access_role: str = "game_master") -> FakeSession:
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

    def get_answer_history(self, session_id: str) -> list:
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

    def render_qa(self, chunks, question: str, history=None) -> str:
        self.last_qa_question = question
        self.last_qa_history = history
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

    async def retrieve(self, query: str, top_k=None, metadata_filters=None, top_k_rerank=None, max_per_source=None,
                       hybrid=None, timings=None):
        if timings is not None:
            timings.update({"embed": 1.0, "search": 2.0})
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


def test_query_uses_condensed_question_in_qa_prompt(client, fake_services):
    """Ensure follow-ups reach the QA prompt in condensed form — it carries no chat history."""
    cm = fake_services.conversation_manager
    cm.config.condense_question = True
    cm.get_history_for_condense = lambda sid: [{"role": "user", "content": "Wer ist Arkenfeld?"}]

    async def _condense(prompt, provider):
        return "Wer ist der Vater von Arkenfeld?"

    fake_services.generator.condense_question = _condense

    response = client.post("/query", json={"question": "Wer ist sein Vater?"})
    assert response.status_code == 200
    assert fake_services.retriever.last_call["query"] == "Wer ist der Vater von Arkenfeld?"
    assert fake_services.prompt_manager.last_qa_question == "Wer ist der Vater von Arkenfeld?"


def test_query_skips_condense_for_standalone_question(client, fake_services):
    """Ensure a self-contained question is retrieved verbatim even with chat history."""
    cm = fake_services.conversation_manager
    cm.config.condense_question = True
    cm.get_history_for_condense = lambda sid: [{"role": "user", "content": "Wer ist Arkenfeld?"}]

    async def _condense(prompt, provider):
        raise AssertionError("condense must not run for a standalone question")

    fake_services.generator.condense_question = _condense

    response = client.post("/query", json={"question": "was ist überschaden"})
    assert response.status_code == 200
    assert fake_services.retriever.last_call["query"] == "was ist überschaden"
    assert response.json()["search_query"] == "was ist überschaden"


def test_query_stream_done_event_carries_condensed_search_query(client, fake_services):
    """Ensure the UI learns which rewritten question was actually searched."""
    cm = fake_services.conversation_manager
    cm.config.condense_question = True
    cm.get_history_for_condense = lambda sid: [{"role": "user", "content": "Wer ist Arkenfeld?"}]

    async def _condense(prompt, provider):
        return "Wer ist der Vater von Arkenfeld?"

    fake_services.generator.condense_question = _condense

    done = parse_sse(client.post("/query/stream", json={"question": "Wer ist sein Vater?"}).text)[-1]
    assert done["type"] == "done"
    assert done["search_query"] == "Wer ist der Vater von Arkenfeld?"


def test_query_falls_back_to_original_question_when_condense_fails(client, fake_services):
    """Ensure a failing condense LLM call degrades to the raw question instead of a 500."""
    cm = fake_services.conversation_manager
    cm.config.condense_question = True
    cm.get_history_for_condense = lambda sid: [{"role": "user", "content": "Wer ist Arkenfeld?"}]

    async def _condense(prompt, provider):
        raise ConnectionError("Ollama down")

    fake_services.generator.condense_question = _condense

    response = client.post("/query", json={"question": "Wer ist sein Vater?"})
    assert response.status_code == 200
    assert fake_services.retriever.last_call["query"] == "Wer ist sein Vater?"


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

def test_query_stream_yields_tokens_and_done_event(client, fake_services):
    """Ensure generating streams provide granular JSON events ending with a structured summary."""
    with client.stream("POST", "/query/stream", json={"question": "Hi"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.read().decode("utf-8")

    events = parse_sse(body)
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


def test_query_returns_stage_timings(client, fake_services):
    """Ensure /query reports retrieval stages, chunk count and prompt size for performance tracking."""
    timings = client.post("/query", json={"question": "Wo liegt Arkenfeld?"}).json()["timings"]
    assert timings["retrieval_stages_ms"] == {"embed": 1, "search": 2}
    assert timings["chunks"] == 1
    assert timings["prompt_chars"] > 0
    assert timings["total_ms"] >= timings["retrieval_ms"] >= 0
    assert "condense_ms" not in timings


def test_query_stream_done_event_carries_timings(client, fake_services):
    """Ensure the done event reports time to first token alongside the total."""
    done = parse_sse(client.post("/query/stream", json={"question": "Wo liegt Arkenfeld?"}).text)[-1]
    timings = done["timings"]
    assert 0 <= timings["first_token_ms"] <= timings["total_ms"]
    assert timings["chunks"] == 1


def test_query_reports_condense_time_for_follow_ups(client, fake_services):
    """Ensure a rewritten follow-up shows up as its own stage."""
    cm = fake_services.conversation_manager
    cm.config.condense_question = True
    cm.get_history_for_condense = lambda sid: [{"role": "user", "content": "Wer ist Arkenfeld?"}]

    async def _condense(prompt, provider):
        return "Wer ist der Vater von Arkenfeld?"

    fake_services.generator.condense_question = _condense

    timings = client.post("/query", json={"question": "Wer ist sein Vater?"}).json()["timings"]
    assert timings["condense_ms"] >= 0


@pytest.mark.parametrize("history, question, expected", [
    ([], "Was ist eine Marke?", ["stream"]),
    ([{"role": "user", "content": "Wer ist Arkenfeld?"}], "Was ist eine Marke?", ["stream"]),
    ([{"role": "user", "content": "Wer ist Arkenfeld?"}], "Wer ist sein Vater?", ["condense", "stream"]),
])
def test_llm_calls_per_request(client, fake_services, history, question, expected):
    """Performance guard (roadmap 5.2): one LLM call per answer, plus one only for a real follow-up.

    Each extra call costs a full LLM round-trip before the first token.
    """
    calls: list[str] = []
    gen = fake_services.generator
    original_stream = gen.generate_stream

    async def _condense(prompt, provider):
        calls.append("condense")
        return "Wer ist der Vater von Arkenfeld?"

    def _stream(*args, **kwargs):
        calls.append("stream")
        return original_stream(*args, **kwargs)

    gen.condense_question = _condense
    gen.generate_stream = _stream
    cm = fake_services.conversation_manager
    cm.config.condense_question = True
    cm.get_history_for_condense = lambda sid: history

    client.post("/query/stream", json={"question": question})

    assert calls == expected


def test_query_stream_reports_model_that_answered(client, fake_services):
    """Ensure model_used comes from the stream result (e.g. the fallback), not the primary provider."""
    class FallbackGenerator(FakeGenerator):
        async def generate_stream(self, system_prompt, qa_prompt, stream_result=None):
            stream_result.model = "gemini-2.5-flash"
            yield "Antwort"

    main_module.generator = FallbackGenerator()
    done = parse_sse(client.post("/query/stream", json={"question": "Hi"}).text)[-1]
    assert done["model_used"] == "gemini-2.5-flash"


def test_query_stream_no_chunks_emits_fallback_token(client, fake_services):
    """Ensure missing context immediately yields the configured fallback response via stream."""
    fake_services.retriever._chunks = []
    with client.stream("POST", "/query/stream", json={"question": "???"}) as resp:
        body = resp.read().decode("utf-8")

    events = parse_sse(body)
    assert any(
        e["type"] == "token" and "Keine relevanten Quellen" in e["content"]
        for e in events
    )
    assert any(e["type"] == "done" for e in events)



def test_query_stream_falls_back_to_original_question_when_condense_fails(client, fake_services):
    """Ensure a failing condense call still opens the SSE stream instead of erroring before it."""
    cm = fake_services.conversation_manager
    cm.config.condense_question = True
    cm.get_history_for_condense = lambda sid: [{"role": "user", "content": "Wer ist Arkenfeld?"}]

    async def _condense(prompt, provider):
        raise ConnectionError("Ollama down")

    fake_services.generator.condense_question = _condense

    with client.stream("POST", "/query/stream", json={"question": "Wer ist sein Vater?"}) as resp:
        assert resp.status_code == 200
        events = parse_sse(resp.read().decode("utf-8"))

    assert fake_services.retriever.last_call["query"] == "Wer ist sein Vater?"
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


def test_switch_provider_unknown_returns_422(client, fake_services):
    """An unknown provider is rejected by the schema (Literal) before anything is rebuilt."""
    before = main_module.provider
    response = client.post("/provider", json={"provider": "openai"})
    assert response.status_code == 422
    assert "openai" in str(response.json()["detail"])
    assert main_module.provider is before


# ─── /ingest ───────────────────────────────────────────────────────────────────

def test_ingest_status_unknown_job_returns_404(client, fake_services):
    """Validate ingestion status endpoints reject unfamiliar background jobs."""
    response = client.get("/ingest/status/nonexistent-job-id")
    assert response.status_code == 404


def test_ingest_rejected_while_job_running(client, fake_services):
    """Validate a second ingest is refused with 409 while one is still running."""
    from src.api import routes as routes_module
    from src.api.schemas import IngestStatusResponse
    routes_module._ingest_jobs["busy"] = IngestStatusResponse(job_id="busy", status="running")

    assert client.post("/ingest").status_code == 409


def test_reindex_rejected_while_job_running(client, fake_services):
    """Validate a per-source reindex is refused with 409 while another job is running."""
    from src.api import routes as routes_module
    from src.api.schemas import IngestStatusResponse
    main_module.config.settings.ingestion.sources = [SimpleNamespace(id="vault")]
    routes_module._ingest_jobs["busy"] = IngestStatusResponse(job_id="busy", status="queued")

    assert client.post("/sources/vault/reindex").status_code == 409


# ─── /config ───────────────────────────────────────────────────────────────────

@pytest.fixture
def config_client(tmp_path):
    """TestClient wired to a real ConfigManager backed by a tmp settings.yaml.

    We need the real ConfigManager here (not a SimpleNamespace fake) so the
    /config routes can exercise the editable_snapshot/save_settings code paths.
    """
    from src.config.manager import ConfigManager

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")

    original_config = main_module.config
    main_module.config = ConfigManager(
        settings_path=settings_path,
        prompts_path=tmp_path / "p.yaml",
        sources_path=tmp_path / "s.yaml",
    )

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # UI writes go to the settings.local.yaml sidecar, not settings.yaml
    yield client, tmp_path / "settings.local.yaml", main_module.config

    main_module.config = original_config


def test_get_config_returns_editable_snapshot(config_client):
    """GET /config must return the expected top-level sections without secrets."""
    client, _, _ = config_client
    response = client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) >= {"retrieval", "llm", "conversation", "chunking", "embeddings", "vectorstore"}
    assert "api_key_env" not in body["llm"]["gemini"]


def test_put_config_persists_and_mutates(config_client):
    """PUT /config must update live settings and write to the settings.local.yaml sidecar."""
    import yaml as _yaml
    client, settings_path, cm = config_client

    response = client.put("/config", json={"retrieval": {"top_k": 27}})
    assert response.status_code == 200
    assert response.json()["retrieval"]["top_k"] == 27

    assert cm.settings.retrieval.top_k == 27
    reloaded = _yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert reloaded["retrieval"]["top_k"] == 27


def test_put_config_rejects_invalid_type(config_client):
    """Invalid types must yield a 422 and leave live settings untouched."""
    client, _, cm = config_client
    before = cm.settings.retrieval.top_k

    response = client.put("/config", json={"retrieval": {"top_k": "banana"}})
    assert response.status_code == 422
    assert cm.settings.retrieval.top_k == before


def test_put_config_drops_disallowed_keys(config_client):
    """Unknown keys must be ignored silently without 4xx."""
    client, _, cm = config_client
    original_embed_model = cm.settings.embeddings.model

    response = client.put("/config", json={"embeddings": {"model": "evil/model"}})
    assert response.status_code == 200
    assert cm.settings.embeddings.model == original_embed_model


def test_put_config_filters_disallowed_inner_keys(config_client):
    """Unknown keys nested inside an allowed section must be stripped by the allow-list."""
    import yaml as _yaml
    client, settings_path, cm = config_client

    response = client.put("/config", json={
        "retrieval": {"top_k": 11, "some_secret": "x"},
    })
    assert response.status_code == 200
    assert cm.settings.retrieval.top_k == 11

    reloaded = _yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert "some_secret" not in reloaded.get("retrieval", {})


# ─── Answer feedback (roadmap 3.4) ──────────────────────────────────────────────

def test_feedback_is_appended_as_json_lines(client, tmp_path, monkeypatch):
    """Each rating is one line; a changed or withdrawn rating is a new line for the same message_id."""
    import src.api.routes as routes_module
    path = tmp_path / "logs" / "feedback.jsonl"
    monkeypatch.setattr(routes_module, "_FEEDBACK_PATH", path)
    body = {"message_id": "m1", "rating": "down", "role": "player", "question": "Was ist Überschaden?",
            "answer": "…", "sources": ["pnp_regelbuch.pdf"], "model": "qwen3:14b"}

    assert client.post("/feedback", json=body).json() == {"saved": True}
    client.post("/feedback", json={**body, "rating": None})

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(e["message_id"], e["rating"]) for e in lines] == [("m1", "down"), ("m1", None)]
    assert lines[0]["question"] == "Was ist Überschaden?" and lines[0]["role"] == "player"
    assert lines[0]["ts"]


def test_feedback_rejects_unknown_ratings(client, tmp_path, monkeypatch):
    import src.api.routes as routes_module
    monkeypatch.setattr(routes_module, "_FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    assert client.post("/feedback", json={"message_id": "m1", "rating": "meh"}).status_code == 422
    assert not (tmp_path / "feedback.jsonl").exists()


def test_feedback_list_keeps_the_last_rating_per_answer(client, tmp_path, monkeypatch):
    """Changed ratings count once, withdrawn ones disappear, broken lines are skipped, newest first."""
    import src.api.routes as routes_module
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(routes_module, "_FEEDBACK_PATH", path)
    lines = [
        {"ts": "2026-10-12T10:00:00", "message_id": "a", "rating": "up", "question": "A"},
        {"ts": "2026-10-12T10:01:00", "message_id": "a", "rating": "down", "question": "A"},   # changed
        {"ts": "2026-10-12T10:02:00", "message_id": "b", "rating": "up", "question": "B"},
        {"ts": "2026-10-12T10:03:00", "message_id": "c", "rating": "down", "question": "C"},
        {"ts": "2026-10-12T10:04:00", "message_id": "c", "rating": None, "question": "C"},     # withdrawn
    ]
    path.write_text("\n".join(json.dumps(e) for e in lines) + "\n{kaputt\n", encoding="utf-8")

    listed = client.get("/feedback").json()

    assert [(e["message_id"], e["rating"]) for e in listed] == [("b", "up"), ("a", "down")]


def test_feedback_list_without_file_is_empty(client, tmp_path, monkeypatch):
    import src.api.routes as routes_module
    monkeypatch.setattr(routes_module, "_FEEDBACK_PATH", tmp_path / "nope.jsonl")
    assert client.get("/feedback").json() == []


def test_answer_prompt_sees_only_the_last_pair(client, fake_services):
    """The QA prompt carries the last question/answer (real template), not older turns and not on the first question."""
    from pathlib import Path

    from src.config.manager import ConversationConfig
    from src.conversation.manager import ConversationManager
    from src.prompts.manager import PromptManager

    main_module.conversation_manager = ConversationManager(ConversationConfig(condense_question=False))
    main_module.prompt_manager = PromptManager(prompts_path=Path("config/prompts.yaml"))
    prompts: list[str] = []
    answers = iter(["Antwort eins.", "Antwort zwei.", "Antwort drei."])

    async def _generate(system_prompt, qa_prompt):
        prompts.append(qa_prompt)
        return FakeLLMResponse(content=next(answers))

    fake_services.generator.generate = _generate

    sid = client.post("/query", json={"question": "Frage eins?"}).json()["session_id"]
    client.post("/query", json={"question": "Frage zwei?", "session_id": sid})
    client.post("/query", json={"question": "Und kürzer?", "session_id": sid})

    assert "Bisheriger Chat" not in prompts[0]
    assert "Nutzer: Frage eins?" in prompts[1] and "LoreKeeper: Antwort eins." in prompts[1]
    assert "Nutzer: Frage zwei?" in prompts[2] and "LoreKeeper: Antwort zwei." in prompts[2]
    assert "Frage eins?" not in prompts[2]
    # Chat first, then sources, the question last
    assert prompts[2].index("Bisheriger Chat") < prompts[2].index("### Quellen") < prompts[2].index("### Frage")


@pytest.mark.parametrize("bad", [{"$and": []}, {"source_id": None}, {"source_id": ["a"]}])
def test_query_rejects_malformed_filters_with_422(client, fake_services, bad):
    """ChromaDB would raise a ValueError (→ 500) — the API answers 422 with the reason instead."""
    fake_services.config = main_module.config
    from src.config.manager import SourceConfig
    main_module.config.settings.ingestion.sources = [SourceConfig(id="r", path="x", group="rules", player_visible=True)]
    for path in ("/query", "/query/stream"):
        resp = client.post(path, json={"question": "x", "metadata_filters": bad})
        assert resp.status_code == 422, (path, resp.text)



# ─── LLM rebuild (roadmap 11.0 P2) ─────────────────────────────────────────────

@pytest.fixture
def llm_builds(monkeypatch):
    """Record every ProviderFactory.build_all call and hand out fresh fakes (fallback per settings)."""
    from src.generation.provider_factory import ProviderFactory
    calls = []
    for name in ("provider", "condense_provider", "generator"):   # install_llm swaps these — restore after
        monkeypatch.setattr(main_module, name, getattr(main_module, name))

    def build_all(settings):
        calls.append(settings.llm.fallback_enabled)
        gen = FakeGenerator()
        gen.fallback_provider = FakeProvider(provider_name="gemini") if settings.llm.fallback_enabled else None
        return FakeProvider(provider_name=settings.llm.provider), None, gen

    monkeypatch.setattr(ProviderFactory, "build_all", staticmethod(build_all))
    return calls


def test_fallback_toggle_on_the_settings_page_takes_effect(config_client, llm_builds, monkeypatch):
    """The Generator fixes its fallback at construction — saving the toggle must rebuild it."""
    client, _, _ = config_client
    monkeypatch.setattr(main_module, "generator", FakeGenerator())

    assert client.put("/config", json={"llm": {"fallback_enabled": True}}).status_code == 200
    assert llm_builds == [True]
    assert main_module.generator.fallback_provider is not None

    assert client.put("/config", json={"llm": {"fallback_enabled": False}}).status_code == 200
    assert main_module.generator.fallback_provider is None


def test_sampling_changes_need_no_rebuild(config_client, llm_builds):
    """Temperature & co. are read per call — no rebuild (and no health check) for them."""
    client, _, _ = config_client
    assert client.put("/config", json={"llm": {"ollama": {"temperature": 0.5}}}).status_code == 200
    assert llm_builds == []


def test_gemini_key_reaches_the_fallback_while_ollama_is_active(config_client, llm_builds, monkeypatch):
    """Before, only an active Gemini was rebuilt — a fallback Gemini kept running without the key."""
    from src.generation.providers import gemini
    client, _, config = config_client
    config.settings.llm.fallback_enabled = True
    try:
        assert client.post("/provider/gemini/key", json={"api_key": "AIza-test"}).status_code == 200
    finally:
        gemini.set_runtime_api_key(None)

    assert llm_builds == [True]
    assert main_module.provider.provider == "ollama"
    assert main_module.generator.fallback_provider is not None
