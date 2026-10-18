"""Evaluation page: tracking an eval job (roadmap 7.4 A5, 13.6).

The job lives in session state and a fragment polls it once per run — so a test can preset the state
a page switch leaves behind and check what the page shows when the user comes back.
"""
from tests.ui import conftest
from tests.ui.conftest import open_page

_JOB = {"id": "j1", "kind": "e2e", "label": "End-to-End Evaluation", "failures": 0}


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


def _tracked(job: dict | None = None):
    at = open_page("evaluation")
    at.session_state["_eval_job"] = dict(job or _JOB)
    return at


def test_running_job_survives_a_page_switch_with_its_progress(fake_backend, monkeypatch):
    """Coming back to the page: progress from the backend, start buttons locked, all tabs still rendered."""
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/status/j1",
                        {"job_id": "j1", "eval_type": "e2e", "status": "running", "progress": 3, "total": 274})
    at = _ok(_tracked().run())

    assert "Frage **3** / 274" in [m.value for m in at.markdown]
    assert at.button(key="e2e_run").disabled and at.button(key="ret_run").disabled
    assert at.session_state["_eval_job"]["id"] == "j1"
    # the old blocking loop never got to the tabs after the running one — the last tab must exist
    assert any(m.value.startswith("Alle gespeicherten Chunks einer Datei") for m in at.markdown)


def test_polling_gives_up_after_repeated_failures(fake_backend):
    fake_backend.fail_paths.add("/eval/status/j1")

    at = _ok(_tracked().run())
    assert any("Status wird abgefragt" in i.value for i in at.info)   # one failure: keep trying
    assert at.session_state["_eval_job"]["failures"] == 1

    at = _ok(_tracked({**_JOB, "failures": 29}).run())
    assert "_eval_job" not in at.session_state
    assert any("Status konnte nicht abgefragt werden" in e.value for e in at.error)


def test_unknown_job_after_a_backend_restart_is_final(fake_backend):
    fake_backend.errors["GET /eval/status/j1"] = (404, {"detail": "Job nicht gefunden"})

    at = _ok(_tracked().run())

    assert "_eval_job" not in at.session_state
    assert any("wurde das Backend neu gestartet" in e.value for e in at.error)


def test_start_during_a_running_eval_attaches_to_it(fake_backend, monkeypatch):
    """409 → /eval/active: the page follows the running job instead of only refusing (13.6)."""
    fake_backend.errors["POST /eval/run"] = (409, {"detail": "Es läuft bereits eine Evaluation"})
    at = _ok(open_page("evaluation").run())
    running = {"job_id": "j9", "eval_type": "e2e", "status": "running", "progress": 5, "total": 274}
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/active", running)
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/status/j9", running)

    _ok(at.button(key="ret_run").click().run())

    job = at.session_state["_eval_job"]
    assert (job["id"], job["kind"], job["label"]) == ("j9", "e2e", "End-to-End Evaluation (schon laufend)")
    assert "Frage **5** / 274" in [m.value for m in at.markdown]


def test_a_new_session_attaches_to_a_running_job(fake_backend, monkeypatch):
    """After a browser reload (new session) the running job is picked up on the first render."""
    running = {"job_id": "j9", "eval_type": "retrieval", "status": "running", "progress": 1, "total": 274}
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/active", running)
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/status/j9", running)

    at = _ok(open_page("evaluation").run())

    assert at.session_state["_eval_job"]["label"] == "Retrieval Evaluation (schon laufend)"
    assert "Frage **1** / 274" in [m.value for m in at.markdown]
