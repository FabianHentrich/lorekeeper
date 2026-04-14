"""Integration tests for the evaluation API endpoints.

Uses the same fake-injection pattern as test_api_routes.py — no real models,
no ChromaDB, no LLM.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.main as main_module
from src.api.eval_routes import eval_router, _eval_jobs, _QA_PATH, _RESULTS_DIR


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
    async def retrieve(self, query, top_k=None, metadata_filters=None,
                       top_k_rerank=None, max_per_source=None, hybrid=None):
        return [FakeRetrievedChunk()]


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
    import src.api.eval_routes as mod

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
