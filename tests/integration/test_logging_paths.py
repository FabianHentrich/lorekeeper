"""Failures must end up in the log (roadmap 10.0, package A).

The in-memory job registry, the HTTP response and the console window are gone after a restart —
the log file is the only record of why something failed on a game night.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import src.main as main_module
from src.api import background, query_routes, sources_routes, system_routes
from src.api.schemas import QueryRequest
from tests.integration import test_api_routes as api


def test_unhandled_route_exception_gets_a_log_line(caplog):
    """The request middleware writes a line even when the route raises (the traceback follows from uvicorn)."""
    app = FastAPI()
    app.middleware("http")(main_module.request_logging)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaputt")

    with caplog.at_level(logging.INFO, logger="src.main"):
        r = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert r.status_code == 500
    assert any("GET /boom → unhandled exception" in rec.getMessage() for rec in caplog.records)


async def test_failed_ingest_job_is_logged_with_traceback(caplog):
    """A reindex that dies after deleting the source's chunks must say why — in the log, not only in the job."""
    sources_routes._ingest_jobs.clear()

    def run(progress_cb):
        raise RuntimeError("chroma locked")

    with caplog.at_level(logging.INFO, logger="src.api.sources_routes"):
        job_id = sources_routes._start_ingest_job(run, "reindex vault")
        await asyncio.gather(*background._background_tasks)

    messages = [rec.getMessage() for rec in caplog.records]
    assert f"Ingest job {job_id} started (reindex vault)" in messages
    failed = next(rec for rec in caplog.records if rec.getMessage() == f"Ingest job {job_id} failed (reindex vault)")
    assert failed.exc_info and "chroma locked" in str(failed.exc_info[1])
    assert sources_routes._ingest_jobs[job_id].status == "error"
    sources_routes._ingest_jobs.clear()


async def test_stream_aborted_by_the_client_is_logged(fake_services, caplog):
    """Closing the stream after the first token (page closed, UI timeout) leaves a line instead of silence."""
    response = await query_routes.query_stream(QueryRequest(question="Was ist Arkenfeld?"))
    events = response.body_iterator

    with caplog.at_level(logging.INFO, logger="src.api.query_routes"):
        await events.__anext__()
        await events.aclose()

    assert any(rec.getMessage().startswith("Stream aborted by the client") for rec in caplog.records)


@pytest.fixture
def fake_services(monkeypatch):
    """The fakes of the API route tests, injected into ``src.main``."""
    monkeypatch.setattr(main_module, "config", SimpleNamespace(settings=SimpleNamespace(
        ingestion=SimpleNamespace(sources=[]))))
    monkeypatch.setattr(main_module, "conversation_manager", api.FakeConversationManager())
    monkeypatch.setattr(main_module, "generator", api.FakeGenerator())
    monkeypatch.setattr(main_module, "prompt_manager", api.FakePromptManager())
    monkeypatch.setattr(main_module, "retriever", api.FakeRetriever())
    monkeypatch.setattr(main_module, "provider", api.FakeProvider())


# ─── Package B: important events ───────────────────────────────────────────────

def test_health_logs_only_changes(monkeypatch, caplog):
    """A line per change (down = WARNING, recovered = INFO), not one every 60 s."""
    monkeypatch.setattr(system_routes, "_health_cache", {"chroma": False, "llm": False})
    with caplog.at_level(logging.INFO, logger="src.api.system_routes"):
        system_routes.update_health(True, True)
        system_routes.update_health(True, True)
        system_routes.update_health(True, False)

    changes = [r for r in caplog.records if r.getMessage().startswith("Health changed")]
    assert [r.levelno for r in changes] == [logging.INFO, logging.WARNING]
    assert "llm=False (was chroma=True llm=True)" in changes[1].getMessage()


async def test_answer_lines_carry_session_role_and_model(fake_services, caplog):
    """With several people asking at once, session/role/model tie the lines of one question together."""
    response = await query_routes.query_stream(QueryRequest(question="Was ist Arkenfeld?", role="game_master"))
    with caplog.at_level(logging.INFO, logger="src.api.query_routes"):
        async for _ in response.body_iterator:
            pass

    timings = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Query timings"))
    assert "session=test-ses role=game_master model=qwen3:8b" in timings


def test_timings_line_carries_tokens_and_output_rate(caplog):
    """17.10.: generation was 3–4× slower with the same prompts (model partly in shared memory) — the rate in
    the log shows that at a glance. Without token counts (provider reports none) the line stays as before."""
    with caplog.at_level(logging.INFO, logger="src.api.query_routes"):
        query_routes._log_timings({"total_ms": 11000}, "session=s role=r", "qwen3:14b",
                                  {"tokens_in": 3000, "tokens_out": 200}, 10000)
        query_routes._log_timings({"total_ms": 900}, "session=s role=r", "none")

    with_tokens, without = (r.getMessage() for r in caplog.records)
    assert with_tokens.endswith("tokens_in=3000 tokens_out=200 20.0tok/s")
    assert "tokens_" not in without


async def test_gemini_key_change_is_logged_without_the_key(monkeypatch, caplog):
    """Set/clear shows up in the log; the key itself never does."""
    from src.generation.provider_factory import ProviderFactory
    from src.generation.providers import gemini
    monkeypatch.setattr(main_module, "config", SimpleNamespace(settings=SimpleNamespace(
        llm=SimpleNamespace(provider="ollama"))))
    monkeypatch.setattr(ProviderFactory, "build_all", staticmethod(
        lambda settings: (api.FakeProvider(), None, api.FakeGenerator())))
    try:
        with caplog.at_level(logging.INFO, logger="src.api.system_routes"):
            await system_routes.set_gemini_key({"api_key": "AIza-SECRET"})
            await system_routes.set_gemini_key({"api_key": None})
    finally:
        gemini.set_runtime_api_key(None)

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Gemini runtime key set") for m in messages)
    assert any(m.startswith("Gemini runtime key cleared") for m in messages)
    assert not any("SECRET" in m for m in messages)


# ─── Package C: noise, the question once ───────────────────────────────────────

def _question_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and r.getMessage().startswith("Question (")]


async def test_the_question_is_logged_once_at_info(fake_services, caplog):
    """One line carries the question — the anchor for the retrieval, timings and error lines after it."""
    with caplog.at_level(logging.INFO):
        await query_routes.query(QueryRequest(question="Was ist Arkenfeld?"))

    assert _question_lines(caplog) == ["Question (session=test-ses role=game_master): 'Was ist Arkenfeld?'"]
    assert sum("Was ist Arkenfeld?" in r.getMessage() for r in caplog.records if r.levelno >= logging.INFO) == 1


async def test_a_condensed_follow_up_shows_both_forms_in_one_line(fake_services, monkeypatch, caplog):
    cm, gen = main_module.conversation_manager, main_module.generator
    monkeypatch.setattr(cm, "config", SimpleNamespace(condense_question=True))
    monkeypatch.setattr(cm, "get_history_for_condense", lambda sid: [{"role": "user", "content": "Wer ist Arkenfeld?"}],
                        raising=False)

    async def condense(prompt, provider):
        return "Wer ist der Vater von Arkenfeld?"

    monkeypatch.setattr(gen, "condense_question", condense)
    with caplog.at_level(logging.INFO):
        await query_routes.query(QueryRequest(question="Wer ist sein Vater?"))

    assert _question_lines(caplog) == [
        "Question (session=test-ses role=game_master): 'Wer ist sein Vater?' → condensed 'Wer ist der Vater von Arkenfeld?'"
    ]


@pytest.mark.parametrize("method, path, status, level", [
    ("GET", "/ingest/status/x", 200, logging.DEBUG),    # polled every 1.5 s
    ("POST", "/ingest", 200, logging.INFO),
    ("GET", "/eval/results/x", 404, logging.INFO),
    ("POST", "/query", 503, logging.WARNING),
])
def test_request_log_level(method, path, status, level, caplog):
    """Successful GETs (page loads, job polling) go to DEBUG; changes and errors stay visible."""
    app = FastAPI()
    app.middleware("http")(main_module.request_logging)
    app.add_api_route(path, lambda: JSONResponse({}, status_code=status), methods=[method])

    with caplog.at_level(logging.DEBUG, logger="src.main"):
        TestClient(app).request(method, path)

    line = next(r for r in caplog.records if r.getMessage().startswith(f"{method} {path}"))
    assert line.levelno == level


def test_stream_request_line_says_it_only_measures_the_start(caplog):
    """/query/stream returns before the answer is generated — the line must not look like the answer time."""
    app = FastAPI()
    app.middleware("http")(main_module.request_logging)
    app.add_api_route("/query/stream", lambda: {}, methods=["POST"])

    with caplog.at_level(logging.INFO, logger="src.main"):
        TestClient(app).post("/query/stream")

    assert any("→ 200 (stream started after" in r.getMessage() for r in caplog.records)


async def test_started_ollama_writes_into_its_log_file(tmp_path, monkeypatch):
    """Ollama's own output (crash reason) goes to logs/ollama.log instead of DEVNULL."""
    log = tmp_path / "ollama.log"
    monkeypatch.setattr(main_module, "_OLLAMA_LOG", log)
    monkeypatch.setattr(main_module.shutil, "which", lambda name: "ollama.exe")
    started = {}

    class FakePopen:
        pid = 1

        def __init__(self, args, stdout, stderr, creationflags=0):
            started.update(stdout=stdout, stderr=stderr, creationflags=creationflags)

    calls = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, timeout):
            calls.append(url)
            if len(calls) == 1:
                raise ConnectionError("not running yet")
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(main_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(main_module.httpx, "AsyncClient", FakeClient)

    proc = await main_module._ensure_ollama("http://127.0.0.1:11434")

    assert isinstance(proc, FakePopen)
    assert started["stdout"].name == str(log)
    assert started["stderr"] == main_module.subprocess.STDOUT
    # Own process group: the dev reloader's console Ctrl+C must not kill Ollama mid-answer (12.3)
    assert started["creationflags"] & main_module.subprocess.CREATE_NEW_PROCESS_GROUP
    assert log.exists()


# ─── Roadmap 8.0: waiting notice ───────────────────────────────────────────────

async def _events(request) -> list[dict]:
    import json
    response = await query_routes.query_stream(request)
    return [json.loads(chunk[6:]) async for chunk in response.body_iterator]


async def test_answer_waiting_for_another_gets_a_notice_first(fake_services, monkeypatch):
    """Ollama answers one question at a time — the chat says why the answer hasn't started yet."""
    monkeypatch.setattr(query_routes, "_active_generations", 1)
    events = await _events(QueryRequest(question="Was ist Arkenfeld?"))

    assert events[0] == {"type": "status", "content": "Das LLM beantwortet gerade eine andere Frage — deine kommt danach."}
    assert events[1]["type"] == "token" and events[-1]["type"] == "done"
    assert query_routes._active_generations == 1  # back to the other answer's count after this one finished


async def test_no_notice_when_the_llm_is_free(fake_services):
    events = await _events(QueryRequest(question="Was ist Arkenfeld?"))
    assert [e["type"] for e in events if e["type"] == "status"] == []
    assert query_routes._active_generations == 0


async def test_count_drops_when_the_client_leaves(fake_services):
    """An aborted stream must not leave a phantom generation — every later question would show the notice."""
    response = await query_routes.query_stream(QueryRequest(question="Was ist Arkenfeld?"))
    events = response.body_iterator
    await events.__anext__()
    assert query_routes._active_generations == 1
    await events.aclose()
    assert query_routes._active_generations == 0
