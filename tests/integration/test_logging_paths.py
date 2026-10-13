"""Failures must end up in the log (roadmap 10.0, package A).

The in-memory job registry, the HTTP response and the console window are gone after a restart —
the log file is the only record of why something failed on a game night.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.main as main_module
from src.api import routes
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
    routes._ingest_jobs.clear()

    def run(progress_cb):
        raise RuntimeError("chroma locked")

    with caplog.at_level(logging.INFO, logger="src.api.routes"):
        job_id = routes._start_ingest_job(run, "reindex vault")
        await asyncio.gather(*routes._background_tasks)

    messages = [rec.getMessage() for rec in caplog.records]
    assert f"Ingest job {job_id} started (reindex vault)" in messages
    failed = next(rec for rec in caplog.records if rec.getMessage() == f"Ingest job {job_id} failed (reindex vault)")
    assert failed.exc_info and "chroma locked" in str(failed.exc_info[1])
    assert routes._ingest_jobs[job_id].status == "error"
    routes._ingest_jobs.clear()


async def test_stream_aborted_by_the_client_is_logged(fake_services, caplog):
    """Closing the stream after the first token (page closed, UI timeout) leaves a line instead of silence."""
    response = await routes.query_stream(QueryRequest(question="Was ist Arkenfeld?"))
    events = response.body_iterator

    with caplog.at_level(logging.INFO, logger="src.api.routes"):
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
    monkeypatch.setattr(routes, "_health_cache", {"chroma": False, "llm": False})
    with caplog.at_level(logging.INFO, logger="src.api.routes"):
        routes.update_health(True, True)
        routes.update_health(True, True)
        routes.update_health(True, False)

    changes = [r for r in caplog.records if r.getMessage().startswith("Health changed")]
    assert [r.levelno for r in changes] == [logging.INFO, logging.WARNING]
    assert "llm=False (was chroma=True llm=True)" in changes[1].getMessage()


async def test_answer_lines_carry_session_role_and_model(fake_services, caplog):
    """With several people asking at once, session/role/model tie the lines of one question together."""
    response = await routes.query_stream(QueryRequest(question="Was ist Arkenfeld?", role="game_master"))
    with caplog.at_level(logging.INFO, logger="src.api.routes"):
        async for _ in response.body_iterator:
            pass

    timings = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Query timings"))
    assert "session=test-ses role=game_master model=qwen3:8b" in timings


async def test_gemini_key_change_is_logged_without_the_key(monkeypatch, caplog):
    """Set/clear shows up in the log; the key itself never does."""
    from src.generation.providers import gemini
    monkeypatch.setattr(main_module, "config", SimpleNamespace(settings=SimpleNamespace(
        llm=SimpleNamespace(provider="ollama"))))
    try:
        with caplog.at_level(logging.INFO, logger="src.api.routes"):
            await routes.set_gemini_key({"api_key": "AIza-SECRET"})
            await routes.set_gemini_key({"api_key": None})
    finally:
        gemini.set_runtime_api_key(None)

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Gemini runtime key set") for m in messages)
    assert any(m.startswith("Gemini runtime key cleared") for m in messages)
    assert not any("SECRET" in m for m in messages)
