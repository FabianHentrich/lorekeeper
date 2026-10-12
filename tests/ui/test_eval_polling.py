"""Evaluation page: polling an eval job gives up when the backend is gone (roadmap 7.4 A5)."""
import time

from tests.ui import conftest
from tests.ui.conftest import open_page


def test_polling_stops_when_the_backend_is_unreachable(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/run", {"job_id": "j1", "status": "queued"})
    fake_backend.fail_paths.add("/eval/status/j1")
    real_sleep = time.sleep   # the loop waits 1 s per try; a no-op would starve AppTest's own wait loop
    monkeypatch.setattr(time, "sleep", lambda s: real_sleep(0.001))

    at = open_page("evaluation")
    at.run()
    at.button(key="ret_run").click().run()

    assert not at.exception, [e.message for e in at.exception]
    assert fake_backend.calls.count("GET /eval/status/j1") == 45
    assert any("Status konnte nicht abgefragt werden" in e.value for e in at.error)
