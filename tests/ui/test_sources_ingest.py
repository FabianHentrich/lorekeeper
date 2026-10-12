"""Ingest progress on the Sources page (roadmap 3.8/3.9): non-blocking, attachable to a running job."""
import pytest

from tests.ui import conftest
from tests.ui.conftest import open_page

RUNNING = {"job_id": "j1", "status": "running", "phase": "parsing", "documents_processed": 4,
           "documents_total": 10, "documents_skipped": 0, "chunks_created": 0, "chunks_updated": 0,
           "chunks_deleted": 0, "duration_seconds": 3.0, "errors": []}
DONE = {**RUNNING, "status": "done", "phase": "done", "documents_processed": 10,
        "documents_skipped": 7, "chunks_created": 12, "duration_seconds": 9.5}


@pytest.fixture
def ingest_backend(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/ingest", {"job_id": "j1", "status": "queued"})
    monkeypatch.setitem(conftest.PAYLOADS, "/ingest/status/j1", RUNNING)
    return fake_backend


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _texts(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_running_job_shows_progress_and_the_page_stays_rendered(ingest_backend):
    """The old poll loop blocked the script — everything below the button was missing until the job ended."""
    at = _click(open_page("sources").run(), "⚡ Nur Änderungen indizieren")

    assert "POST /ingest" in ingest_backend.calls
    assert at.session_state["_ingest_job"]["id"] == "j1"
    assert "Parsing & Chunking: **4** / 10 Dokumente" in _texts(at)
    assert "Neue Source hinzufügen" in [h.value for h in at.subheader]   # rest of the page is there
    assert next(b for b in at.button if b.label == "⚡ Nur Änderungen indizieren").disabled


def test_progress_survives_a_rerun(ingest_backend):
    """Any widget click reruns the page — the tracked job keeps being shown instead of being lost."""
    at = _click(open_page("sources").run(), "⚡ Nur Änderungen indizieren")
    at.run()

    assert at.session_state["_ingest_job"]["id"] == "j1"
    assert "Parsing & Chunking" in _texts(at)


def test_finished_job_shows_result_and_stops_tracking(ingest_backend, monkeypatch):
    at = open_page("sources")
    at.session_state["_ingest_job"] = {"id": "j1", "label": "Inkrementelle Indizierung", "failures": 0}
    monkeypatch.setitem(conftest.PAYLOADS, "/ingest/status/j1", DONE)
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    assert "_ingest_job" not in at.session_state
    assert "**3** Dokumente indiziert · **7** unverändert" in _texts(at)
    assert not next(b for b in at.button if b.label == "⚡ Nur Änderungen indizieren").disabled


def test_conflict_attaches_to_the_running_job(ingest_backend, monkeypatch):
    """409 used to end in a warning only; now the page follows the job that is already running."""
    ingest_backend.errors["POST /ingest"] = (409, {"detail": "Eine Indexierung läuft bereits"})
    monkeypatch.setitem(conftest.PAYLOADS, "/ingest/active", {**RUNNING, "job_id": "j7"})
    monkeypatch.setitem(conftest.PAYLOADS, "/ingest/status/j7", {**RUNNING, "job_id": "j7"})

    at = _click(open_page("sources").run(), "⚡ Nur Änderungen indizieren")

    assert "GET /ingest/active" in ingest_backend.calls
    assert at.session_state["_ingest_job"] == {"id": "j7", "label": "Bereits laufende Indizierung", "failures": 0}
    assert "Parsing & Chunking" in _texts(at)


def test_unreachable_backend_gives_up_after_the_failure_limit(ingest_backend):
    at = open_page("sources")
    at.session_state["_ingest_job"] = {"id": "j1", "label": "Reindex", "failures": 29}
    ingest_backend.fail_paths.add("/ingest/status/j1")
    at.run()

    assert not at.exception, [e.message for e in at.exception]
    assert "_ingest_job" not in at.session_state
    assert any("Status konnte nicht abgefragt werden" in w.value for w in at.warning)
