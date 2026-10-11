"""Timing/size check (roadmap 5.2): the golden set against the live backend.

Runs the retrieval eval through the running backend (the embedded ChromaDB can't be
opened twice) and compares latency, context size and section hits to the baseline.
Skipped when no backend answers on 127.0.0.1:8000. Takes a few minutes.
"""
import statistics
import time

import pytest
import requests

pytestmark = pytest.mark.perf

API = "http://127.0.0.1:8000"


@pytest.fixture(scope="module")
def report():
    try:
        requests.get(f"{API}/health", timeout=3).raise_for_status()
    except requests.RequestException:
        pytest.skip(f"no backend on {API}")

    job_id = requests.post(f"{API}/eval/run", json={"eval_type": "retrieval"}, timeout=30).json()["job_id"]
    deadline = time.time() + 1200
    while time.time() < deadline:
        status = requests.get(f"{API}/eval/status/{job_id}", timeout=30).json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(5)
    assert status["status"] == "done", status
    return requests.get(f"{API}/eval/results/{status['result_file']}", timeout=30).json()


def test_retrieval_latency(report, baseline):
    """p50/p95 retrieval latency per question (×1.5 + 100 ms slack)."""
    latencies = sorted(d["latency_ms"] for d in report["details"] if "latency_ms" in d)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    baseline.check("golden_retrieval", "p50_ms", statistics.median(latencies), slack=100)
    baseline.check("golden_retrieval", "p95_ms", p95, slack=100)


def test_context_size(report, baseline):
    """Chunks handed to the LLM per question — catches context bloat (e.g. runaway section expansion)."""
    counts = [len(d.get("retrieved_files", [])) for d in report["details"]]
    baseline.check("golden_retrieval", "avg_chunks", statistics.mean(counts), factor=1.15)
    baseline.check("golden_retrieval", "max_chunks", max(counts), factor=1.0, slack=5)


def test_section_hits_do_not_drop(report, baseline):
    """Speed-ups must not cost retrieval quality: section hit rate may drop at most 2 points."""
    baseline.check("golden_retrieval", "section_hit_rate", report["section_hit_rate"],
                   factor=1.0, slack=0.02, higher_is_better=True)
