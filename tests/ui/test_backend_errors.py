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
