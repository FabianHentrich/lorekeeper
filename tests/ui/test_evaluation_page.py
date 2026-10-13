"""Evaluation page: golden set save, eval runs with their result views, A/B and result comparison
(roadmap 9.0 — page had 56 % coverage). The backend is the fake from tests/ui/conftest.py."""
import pytest

from tests.ui import conftest
from tests.ui.conftest import open_page

RETRIEVAL_REPORT = {
    "timestamp": "t", "hit_rate": 0.5, "total_questions": 2, "section_hit_rate": 0.0, "section_total": 1,
    "config": {"top_k": 15, "top_k_rerank": 8, "max_per_source": 3, "hybrid": None},
    "breakdown_by_source_type": {"pdf": {"hit_rate": 0.5, "hits": 1, "total": 2}},
    "details": [
        {"id": "p001", "question": "Was ist eine Marke?", "hit": True, "section_hit": False,
         "expected_section": "2.5", "expected_sources": ["regeln.pdf"], "retrieved_files": ["regeln.pdf"],
         "retrieved_headings": ["2.4 Proben"], "top_score": 0.9, "latency_ms": 40},
        {"id": "p002", "question": "Was ist Überschaden?", "hit": False, "expected_sources": ["regeln.pdf"],
         "retrieved_files": ["welt.md"], "top_score": 0.4, "latency_ms": 60},
    ],
    "errors": [{"id": "p003", "error": "timeout"}],
}
OTHER_REPORT = {**RETRIEVAL_REPORT, "hit_rate": 1.0, "details": [
    {**RETRIEVAL_REPORT["details"][0]}, {**RETRIEVAL_REPORT["details"][1], "hit": True},
]}
E2E_REPORT = {"hit_rate": 1.0, "answer_contains_rate": 0.5, "avg_latency_ms": 3200, "details": [
    {"question": "Was ist eine Marke?", "answer": "Eine Marke ist eine Ressource.", "hit": True,
     "answer_contains_match": False, "expected_sources": ["regeln.pdf"], "sources_retrieved": ["regeln.pdf"]},
]}


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.fixture
def jobs(fake_backend, monkeypatch):
    """/eval/run starts j1, which is already done and points at retrieval_1.json."""
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/run", {"job_id": "j1", "status": "queued"})
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/status/j1",
                        {"job_id": "j1", "status": "done", "progress": 2, "total": 2, "result_file": "retrieval_1.json"})
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results/retrieval_1.json", RETRIEVAL_REPORT)
    return fake_backend


def test_golden_set_save_sends_the_table_back(fake_backend):
    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="gs_save").click().run())

    [body] = fake_backend.bodies["PUT /eval/qa-pairs"]
    [pair] = body["pairs"]
    original = conftest.PAYLOADS["/eval/qa-pairs"]["pairs"][0]
    assert {k: pair[k] for k in original} == original     # list fields survive the comma-separated cells
    assert any("1 QA-Paare gespeichert" in s.value for s in at.success)


def test_retrieval_run_shows_metrics_hits_misses_and_errors(jobs):
    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="ret_run").click().run())

    assert jobs.bodies["POST /eval/run"][0]["eval_type"] == "retrieval"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Hit Rate"] == "50.0%" and metrics["Fragen"] == "2"
    assert metrics["Config"] == "K=15 R=8 C=3 H=default"
    labels = [e.label for e in at.expander]
    assert any(label.startswith("⚠️ [p001]") for label in labels)   # right file, wrong section
    assert any(label.startswith("❌ [p002]") for label in labels)
    assert "1 Fehler aufgetreten" in [w.value for w in at.warning]


def test_ab_run_sends_vector_then_hybrid(jobs):
    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="ret_ab").click().run())

    assert [b["hybrid"] for b in jobs.bodies["POST /eval/run"]] == [False, True]
    assert "A/B-Vergleich" in [h.value for h in at.subheader]
    assert "Keine Unterschiede bei Hit/Miss zwischen Vektor und Hybrid." in [i.value for i in at.info]


def test_e2e_run_shows_answer_metrics(jobs, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results/retrieval_1.json", E2E_REPORT)
    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="e2e_run").click().run())

    assert jobs.bodies["POST /eval/run"] == [{"eval_type": "e2e"}]
    metrics = {m.label: m.value for m in at.metric}
    assert (metrics["Hit Rate"], metrics["Answer Contains"], metrics["Ø Latenz"]) == ("100.0%", "50.0%", "3200 ms")


def test_comparing_two_stored_results_lists_the_flipped_questions(fake_backend, monkeypatch):
    summary = {"timestamp": "t", "eval_type": "retrieval", "hit_rate": 0.5, "total_questions": 2, "config": {}}
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results", [
        {**summary, "filename": "retrieval_2.json", "hit_rate": 1.0}, {**summary, "filename": "retrieval_1.json"},
    ])
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results/retrieval_1.json", RETRIEVAL_REPORT)
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results/retrieval_2.json", OTHER_REPORT)
    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="cmp_run").click().run())

    metrics = {m.label: m.value for m in at.metric}
    assert (metrics["Hit Rate A"], metrics["Hit Rate B"], metrics["Differenz"]) == ("100.0%", "50.0%", "-50.0%")
    assert "**1 Fragen mit geändertem Ergebnis:**" in [m.value for m in at.markdown]
