"""End-to-end eval script (evaluation/evaluate.py) against a mocked /query — no backend, no LLM."""
from types import SimpleNamespace

import pytest
import requests

from evaluation import evaluate as ev

PAIRS = [
    {"id": "m001", "question": "Wer regiert Arkenfeld?", "expected_sources": ["Orte/Arkenfeld.md"],
     "expected_answer_contains": ["Rat", "Magier"]},
    {"question": "Was kostet ein Schwert?", "expected_sources": ["Regeln/Waffen.md"],
     "expected_answer_contains": ["Gold"]},
    {"id": "x009", "question": "Kaputt?", "expected_sources": ["x.md"]},
]


def _response(answer, files):
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"answer": answer, "sources": [{"file": f} for f in files]},
    )


@pytest.fixture
def fake_query(monkeypatch):
    answers = {
        "Wer regiert Arkenfeld?": _response("Ein Rat aus sieben MAGIERN.", ["Orte\\Arkenfeld.md"]),
        "Was kostet ein Schwert?": _response("Das steht nicht in den Quellen.", ["Orte/Arkenfeld.md"]),
    }
    sent = []

    def post(url, json, timeout):
        sent.append((url, json))
        if json["question"] not in answers:
            raise requests.ConnectionError("backend down")
        return answers[json["question"]]

    monkeypatch.setattr(ev.requests, "post", post)
    return sent


def test_rates_count_hits_keywords_and_failures(fake_query):
    report = ev.evaluate("http://127.0.0.1:8000", PAIRS)

    assert fake_query[0] == ("http://127.0.0.1:8000/query", {"question": "Wer regiert Arkenfeld?"})
    assert report["total_questions"] == 3
    assert report["hit_rate"] == pytest.approx(1 / 3)          # Windows path normalised → hit
    assert report["answer_contains_rate"] == pytest.approx(1 / 3)  # keywords case-insensitive
    first, second, failed = report["details"]
    assert (first["hit"], first["answer_contains_match"]) == (True, True)
    assert (second["hit"], second["answer_contains_match"]) == (False, False)
    assert failed == {"id": "x009", "question": "Kaputt?", "error": "backend down"}


def test_details_keep_the_whole_answer_id_and_keywords(monkeypatch):
    """A 200-character excerpt couldn't show why a list answer failed (roadmap 3.15)."""
    long_answer = "Klassen: " + ", ".join(f"Klasse{i}" for i in range(100))  # ~1,000 characters
    monkeypatch.setattr(ev.requests, "post", lambda url, json, timeout: _response(long_answer, ["Orte/Arkenfeld.md"]))
    first, second, _ = ev.evaluate("http://127.0.0.1:8000", PAIRS)["details"]
    assert first["answer"] == long_answer
    assert (first["id"], first["expected_answer_contains"]) == ("m001", ["Rat", "Magier"])
    assert second["id"] == ""  # a pair without id


def test_average_latency_ignores_failed_requests(fake_query, monkeypatch):
    """A failed request has no latency — it must not pull the mean down."""
    clock = iter([0.0, 2.0, 10.0, 14.0, 20.0])            # 2 s and 4 s for the two answered questions
    monkeypatch.setattr(ev.time, "time", lambda: next(clock))

    report = ev.evaluate("http://x", PAIRS)

    assert report["avg_latency_ms"] == pytest.approx(3000)


def test_empty_golden_set_gives_zero_rates():
    report = ev.evaluate("http://x", [])
    assert (report["hit_rate"], report["answer_contains_rate"], report["avg_latency_ms"]) == (0, 0, 0)
