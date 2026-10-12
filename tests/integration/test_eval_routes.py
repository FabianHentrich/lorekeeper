"""Integration tests for the evaluation API endpoints.

Uses the same fake-injection pattern as test_api_routes.py — no real models,
no ChromaDB, no LLM.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.main as main_module
from src.api.eval_routes import _eval_jobs, eval_router

# ─── Fakes ──────────────────────────────────────────────────────────────────────

@dataclass
class FakeRetrievedChunk:
    """Mock stub for the search engine's context output."""
    content: str = "Test content about Arkenfeld."
    source_file: str = "Orte/Arkenfeld.md"
    document_type: str = "markdown"
    heading: str = "Arkenfeld > Overview"
    score: float = 0.85
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FakeRetriever:
    """Mock stub intercepting similarity searches."""
    last_filters: dict | None = None

    async def retrieve(self, query, top_k=None, metadata_filters=None,
                       top_k_rerank=None, max_per_source=None, hybrid=None):
        FakeRetriever.last_filters = metadata_filters
        return [FakeRetrievedChunk(metadata={"source_id": "vault"})]


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_eval_dir(tmp_path, monkeypatch):
    """Redirect QA path and results dir to a temp directory to protect workspace files."""
    import src.api.eval_routes as mod

    qa_path = tmp_path / "qa_pairs.yaml"
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    monkeypatch.setattr(mod, "_QA_PATH", qa_path)
    monkeypatch.setattr(mod, "_RESULTS_DIR", results_dir)

    return SimpleNamespace(qa_path=qa_path, results_dir=results_dir)


@pytest.fixture
def fake_retriever(monkeypatch):
    """Inject a static mock retriever into the main module namespace."""
    original = main_module.retriever
    main_module.retriever = FakeRetriever()
    yield
    main_module.retriever = original


@pytest.fixture
def client(tmp_eval_dir, fake_retriever):
    """A clear-state TestClient for the evaluation router."""
    _eval_jobs.clear()
    app = FastAPI()
    app.include_router(eval_router)
    return TestClient(app)


# ─── QA Pairs CRUD ──────────────────────────────────────────────────────────────

def test_get_qa_pairs_empty(client):
    """Ensure fetching pairs from an empty (new) temp file returns an empty list."""
    resp = client.get("/eval/qa-pairs")
    assert resp.status_code == 200
    assert resp.json()["pairs"] == []


def test_put_and_get_qa_pairs(client):
    """Verify writing configuration overwrites the YAML correctly and can be re-fetched."""
    pairs = [
        {
            "id": "t001",
            "question": "Was ist Arkenfeld?",
            "source_type": "markdown",
            "category": "location",
            "expected_sources": ["Orte/Arkenfeld.md"],
            "expected_answer_contains": ["Arkenfeld"],
            "notes": "test",
        }
    ]
    resp = client.put("/eval/qa-pairs", json={"pairs": pairs})
    assert resp.status_code == 200

    resp = client.get("/eval/qa-pairs")
    result = resp.json()["pairs"]
    assert len(result) == 1
    assert result[0]["id"] == "t001"
    assert result[0]["expected_sources"] == ["Orte/Arkenfeld.md"]


# ─── Preview ────────────────────────────────────────────────────────────────────

def test_preview(client):
    """Ensure retrieving context directly proxies to the internal retriever mechanics."""
    resp = client.post("/eval/preview", json={
        "question": "Was ist Arkenfeld?",
        "top_k": 10,
        "top_k_rerank": 5,
        "max_per_source": 3,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["source_file"] == "Orte/Arkenfeld.md"
    assert data["latency_ms"] >= 0


def test_preview_passes_filters_and_reports_source_id(client):
    """The live player-mode checks (tests/perf) preview with the player filter and verify source_id."""
    player_filter = {"source_id": {"$in": ["regelbuch"]}}
    data = client.post("/eval/preview", json={"question": "Was ist eine Marke?",
                                              "metadata_filters": player_filter}).json()

    assert FakeRetriever.last_filters == player_filter
    assert data["chunks"][0]["source_id"] == "vault"


# ─── Retrieval Eval Job ─────────────────────────────────────────────────────────

def test_retrieval_eval_run(client, tmp_eval_dir):
    """End-to-end integration mapping over evaluating the expected list of files."""
    # Write a minimal golden set
    import yaml
    pairs = [{"id": "t001", "question": "Was ist Arkenfeld?",
              "source_type": "markdown", "category": "location",
              "expected_sources": ["Orte/Arkenfeld.md"],
              "expected_answer_contains": [], "notes": ""}]
    tmp_eval_dir.qa_path.write_text(
        yaml.dump({"pairs": pairs}, allow_unicode=True), encoding="utf-8",
    )

    resp = client.post("/eval/run", json={
        "top_k": 10, "top_k_rerank": 5, "max_per_source": 3,
        "eval_type": "retrieval",
    })
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # Poll until done (with timeout)
    deadline = time.time() + 10
    while time.time() < deadline:
        status = client.get(f"/eval/status/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.2)

    assert status["status"] == "done"
    assert status["result_file"] is not None

    # Result should be loadable
    result = client.get(f"/eval/results/{status['result_file']}").json()
    assert result["total_questions"] == 1
    assert result["hit_rate"] == 1.0


def test_retrieval_eval_reports_section_hits(client, tmp_eval_dir):
    """Questions with expected_section get a section-level verdict next to the file-level hit."""
    import yaml
    pairs = [
        {"id": "t001", "question": "Was ist Arkenfeld?", "source_type": "markdown",
         "expected_sources": ["Orte/Arkenfeld.md"], "expected_section": "4.10"},
        {"id": "t002", "question": "Was ist Arkenfeld?", "source_type": "markdown",
         "expected_sources": ["Orte/Arkenfeld.md"]},
    ]
    tmp_eval_dir.qa_path.write_text(yaml.dump({"pairs": pairs}, allow_unicode=True), encoding="utf-8")

    job_id = client.post("/eval/run", json={"eval_type": "retrieval"}).json()["job_id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        status = client.get(f"/eval/status/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.2)

    result = client.get(f"/eval/results/{status['result_file']}").json()
    assert result["hit_rate"] == 1.0
    assert result["section_total"] == 1
    assert result["section_hit_rate"] == 0.0
    assert [m["id"] for m in result["section_misses"]] == ["t001"]
    assert result["details"][1]["section_hit"] is None


def test_reject_concurrent_eval(client, tmp_eval_dir):
    """409 when an eval is already running to prevent OOM errors and lock contention."""
    import yaml
    pairs = [{"id": "t001", "question": "test", "source_type": "markdown",
              "category": "x", "expected_sources": [], "expected_answer_contains": [],
              "notes": ""}]
    tmp_eval_dir.qa_path.write_text(
        yaml.dump({"pairs": pairs}, allow_unicode=True), encoding="utf-8",
    )

    # Inject a fake running job
    from src.api.eval_schemas import EvalJobStatus
    _eval_jobs["fake"] = EvalJobStatus(job_id="fake", status="running")

    resp = client.post("/eval/run", json={"eval_type": "retrieval"})
    assert resp.status_code == 409

    _eval_jobs.clear()


# ─── Results ────────────────────────────────────────────────────────────────────

def test_list_results_empty(client):
    """Fetching list from empty directory acts as an empty array, not 404."""
    resp = client.get("/eval/results")
    assert resp.status_code == 200
    assert resp.json() == []


def test_results_cap_at_3(tmp_eval_dir):
    """Cleanup should keep only the last 3 result files per type."""
    from src.api.eval_routes import _cleanup_results

    for i in range(5):
        f = tmp_eval_dir.results_dir / f"retrieval_20260410_00000{i}.json"
        f.write_text(json.dumps({"hit_rate": 0.9, "total_questions": 1}))

    _cleanup_results("retrieval")

    remaining = sorted(tmp_eval_dir.results_dir.glob("retrieval_*.json"))
    assert len(remaining) == 3
    # Should keep the newest (highest index)
    names = [f.name for f in remaining]
    assert "retrieval_20260410_000004.json" in names
    assert "retrieval_20260410_000003.json" in names
    assert "retrieval_20260410_000002.json" in names


def test_delete_result(client, tmp_eval_dir):
    """Ensure deleting an existing evaluation artifact sweeps the disk."""
    f = tmp_eval_dir.results_dir / "retrieval_test.json"
    f.write_text(json.dumps({"hit_rate": 0.5}))

    resp = client.delete("/eval/results/retrieval_test.json")
    assert resp.status_code == 200
    assert not f.exists()


def test_delete_nonexistent_result(client):
    """Ensure non-existent files trigger logical HTTP 404s."""
    resp = client.delete("/eval/results/nope.json")
    assert resp.status_code == 404


@pytest.mark.parametrize("name", ["..%5Csecret.json", "sub%5Csecret.json"])
def test_result_paths_cannot_escape_results_dir(client, tmp_eval_dir, name):
    """Ensure encoded backslashes can't read or delete files outside the results dir."""
    outside = tmp_eval_dir.results_dir.parent / "secret.json"
    outside.write_text(json.dumps({"hit_rate": 1.0}))

    assert client.get(f"/eval/results/{name}").status_code == 400
    assert client.delete(f"/eval/results/{name}").status_code == 400
    assert outside.exists()


# ─── Job start validation, e2e and error paths ──────────────────────────────────

_PAIR = {"id": "t001", "question": "Was ist Arkenfeld?", "source_type": "markdown", "category": "location",
         "expected_sources": ["Orte/Arkenfeld.md"], "expected_answer_contains": [], "notes": ""}


def _write_pairs(tmp_eval_dir, pairs):
    import yaml
    tmp_eval_dir.qa_path.write_text(yaml.dump({"pairs": pairs}, allow_unicode=True), encoding="utf-8")


def _wait(client, job_id: str) -> dict:
    deadline = time.time() + 10
    while time.time() < deadline:
        status = client.get(f"/eval/status/{job_id}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.05)
    pytest.fail(f"eval job {job_id} did not finish")


def test_run_without_golden_set_is_404(client):
    assert client.post("/eval/run", json={"eval_type": "retrieval"}).status_code == 404


def test_run_with_empty_golden_set_is_400(client, tmp_eval_dir):
    _write_pairs(tmp_eval_dir, [])
    r = client.post("/eval/run", json={"eval_type": "retrieval"})
    assert r.status_code == 400 and "empty" in r.json()["detail"]


def test_run_with_unknown_eval_type_is_400(client, tmp_eval_dir):
    _write_pairs(tmp_eval_dir, [_PAIR])
    r = client.post("/eval/run", json={"eval_type": "bogus"})
    assert r.status_code == 400 and "bogus" in r.json()["detail"]


def test_unknown_job_status_is_404(client):
    assert client.get("/eval/status/nope").status_code == 404


def test_retrieval_eval_failure_marks_job_as_error(client, tmp_eval_dir, monkeypatch):
    import evaluation.evaluate_retrieval as ev

    async def boom(**kwargs):
        raise RuntimeError("chroma locked")

    monkeypatch.setattr(ev, "run_evaluation_with_retriever", boom)
    _write_pairs(tmp_eval_dir, [_PAIR])

    status = _wait(client, client.post("/eval/run", json={"eval_type": "retrieval"}).json()["job_id"])

    assert (status["status"], status["error"]) == ("error", "chroma locked")
    assert list(tmp_eval_dir.results_dir.iterdir()) == []


def test_e2e_eval_saves_report_against_the_same_server(client, tmp_eval_dir, monkeypatch):
    """The e2e evaluator gets this server's base URL and its report is listed as e2e."""
    import evaluation.evaluate as ev
    seen = {}

    def fake_evaluate(api_url, qa_pairs):
        seen["url"], seen["n"] = api_url, len(qa_pairs)
        return {"timestamp": "t", "hit_rate": 0.5, "total_questions": 2, "config": {}}

    monkeypatch.setattr(ev, "evaluate", fake_evaluate)
    _write_pairs(tmp_eval_dir, [_PAIR, {**_PAIR, "id": "t002"}])

    status = _wait(client, client.post("/eval/run", json={"eval_type": "e2e"}).json()["job_id"])

    assert status["status"] == "done"
    assert status["progress"] == status["total"] == 2
    assert seen == {"url": "http://testserver", "n": 2}
    assert status["result_file"].startswith("e2e_")
    listed = client.get("/eval/results").json()
    assert [(r["eval_type"], r["hit_rate"]) for r in listed] == [("e2e", 0.5)]


def test_e2e_eval_failure_marks_job_as_error(client, tmp_eval_dir, monkeypatch):
    import evaluation.evaluate as ev

    def boom(api_url, qa_pairs):
        raise ConnectionError("backend down")

    monkeypatch.setattr(ev, "evaluate", boom)
    _write_pairs(tmp_eval_dir, [_PAIR])

    status = _wait(client, client.post("/eval/run", json={"eval_type": "e2e"}).json()["job_id"])

    assert (status["status"], status["error"]) == ("error", "backend down")


def test_list_results_skips_unreadable_files(client, tmp_eval_dir):
    (tmp_eval_dir.results_dir / "retrieval_20261012_100000.json").write_text(
        json.dumps({"timestamp": "t", "hit_rate": 0.9, "total_questions": 3}), encoding="utf-8")
    (tmp_eval_dir.results_dir / "retrieval_20261012_090000.json").write_text("{kaputt", encoding="utf-8")

    listed = client.get("/eval/results").json()

    assert [(r["filename"], r["eval_type"], r["hit_rate"]) for r in listed] == \
        [("retrieval_20261012_100000.json", "retrieval", 0.9)]


def test_get_missing_result_is_404(client):
    assert client.get("/eval/results/retrieval_nope.json").status_code == 404


def test_list_results_without_results_dir(client, tmp_eval_dir):
    tmp_eval_dir.results_dir.rmdir()
    assert client.get("/eval/results").json() == []
