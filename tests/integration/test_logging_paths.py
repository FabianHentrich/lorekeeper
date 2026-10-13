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
