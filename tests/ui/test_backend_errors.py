"""Shared backend error handling (ui/backend.py) and the eval slider defaults (roadmap 7.4, package D)."""
import copy

from tests.ui import conftest
from tests.ui.conftest import open_page


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


def _slider(at, key):
    return next(s for s in at.slider if s.key == key)


def test_eval_sliders_start_at_the_configured_values(fake_backend, monkeypatch):
    """Untouched sliders must measure what the chat uses — the values from settings.yaml, not 15/8/3."""
    cfg = copy.deepcopy(conftest.PAYLOADS["/config"])
    cfg["retrieval"]["top_k"] = 20
    cfg["retrieval"]["reranking"].update(top_k_rerank=6, max_per_source=2)
    monkeypatch.setitem(conftest.PAYLOADS, "/config", cfg)

    at = _ok(open_page("evaluation").run())

    assert _slider(at, "ret_top_k").value == 20
    assert _slider(at, "ret_top_k_rerank").value == 6
    assert _slider(at, "ret_max_per_source").value == 2


def test_eval_already_running_is_a_warning(fake_backend):
    """A 409 from /eval/run is a warning, not an error — recognised via BackendError.status_code."""
    fake_backend.errors["POST /eval/run"] = (409, {"detail": "Es läuft bereits eine Evaluation"})
    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="ret_run").click().run())

    assert "Es läuft bereits eine Evaluation" in [w.value for w in at.warning]
    assert not at.error


def test_failed_result_delete_shows_the_backend_reason(fake_backend, monkeypatch):
    """The delete used to ignore the status code — a failed delete looked like a success."""
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results", [{
        "filename": "retrieval_1.json", "timestamp": "t", "eval_type": "retrieval",
        "hit_rate": 0.5, "total_questions": 2, "config": {},
    }])
    fake_backend.errors["DELETE /eval/results/retrieval_1.json"] = (404, {"detail": "Ergebnis nicht gefunden"})
    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="del_retrieval_1.json").click().run())

    assert any("Ergebnis nicht gefunden" in e.value for e in at.error), [e.value for e in at.error]


def test_variant_names_are_url_encoded_and_a_failed_delete_is_shown(fake_backend, monkeypatch):
    """A name with a space reaches the backend encoded; a refused delete shows the backend's detail."""
    variant = {"name": "Meine Variante", "description": "", "prompts": conftest.PAYLOADS["/prompts/active"]["prompts"]}
    monkeypatch.setitem(conftest.PAYLOADS, "/prompts/variants",
                        [{"name": "Meine Variante", "description": "", "is_active": False}])
    monkeypatch.setitem(conftest.PAYLOADS, "/prompts/variants/Meine%20Variante", variant)
    fake_backend.errors["DELETE /prompts/variants/Meine%20Variante"] = (500, {"detail": "Datei gesperrt"})

    at = _ok(open_page("prompts").run())
    assert "GET /prompts/variants/Meine%20Variante" in fake_backend.calls
    _ok(at.button(key="del_Meine Variante").click().run())

    assert any("500: Datei gesperrt" in e.value for e in at.error), [e.value for e in at.error]


# ─── Reads go through get_json (roadmap 11.0 P5) ────────────────────────────────

def test_failed_golden_set_read_never_overwrites_the_golden_set(fake_backend, monkeypatch):
    """"Add to golden set" reads the set, appends, writes it back. A failed read used to be taken as
    data ({"detail": …} → no pairs) — the PUT then replaced the whole golden set with one question."""
    from tests.ui.test_question_preview import PREVIEW
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/preview", PREVIEW)
    at = _ok(open_page("evaluation").run())
    at.text_input(key="test_question").set_value("Was ist Arkenfeld?").run()
    _ok(at.button(key="test_run").click().run())

    fake_backend.errors["GET /eval/qa-pairs"] = (500, {"detail": "Datei gesperrt"})
    at.text_input(key="test_add_id_1").set_value("m099").run()
    _ok(at.button(key="test_add_btn_1").click().run())

    assert "PUT /eval/qa-pairs" not in fake_backend.calls
    assert any("500: Datei gesperrt" in e.value for e in at.error), [e.value for e in at.error]


def test_eval_job_unknown_to_the_backend_ends_polling_at_once(fake_backend, monkeypatch):
    """404 (job registry gone after a backend restart) is final — not 45 retries ending in "unreachable"."""
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/run", {"job_id": "j1", "status": "queued"})
    fake_backend.errors["GET /eval/status/j1"] = (404, {"detail": "Job nicht gefunden"})
    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="ret_run").click().run())

    assert fake_backend.calls.count("GET /eval/status/j1") == 1
    assert any("Job nicht gefunden" in e.value for e in at.error), [e.value for e in at.error]


def test_ingest_job_unknown_to_the_backend_ends_tracking(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/ingest", {"job_id": "j1", "status": "queued"})
    fake_backend.errors["GET /ingest/status/j1"] = (404, {"detail": "Job nicht gefunden"})
    at = _ok(open_page("sources").run())
    _ok(next(b for b in at.button if b.label == "⚡ Nur Änderungen indizieren").click().run())

    assert "_ingest_job" not in at.session_state
    assert fake_backend.calls.count("GET /ingest/status/j1") == 1
