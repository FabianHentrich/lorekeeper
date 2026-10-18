"""
LoreKeeper Evaluation Script

End to end through the running backend's ``/query`` (LLM included): file hit rate, how often the
answer contains the expected keywords, latency.

Usage:
    python -m evaluation.evaluate \
        --qa-pairs evaluation/qa_pairs.yaml \
        --api-url http://127.0.0.1:8000 \
        --output evaluation/results/
"""

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime

import requests

from evaluation.evaluate_retrieval import load_qa_pairs, save_report, source_hit


def evaluate(api_url: str, qa_pairs: list[dict],
             progress_callback: Callable[[int, int], None] | None = None, timeout: float = 120) -> dict:
    """Run an end-to-end evaluation against a live API endpoint.

    Sends each question as a POST request to `/query`, measuring latency and
    testing if the retrieved sources and generated answer match the expected
    values defined in the evaluation dataset. A failed request counts as a miss (it stays in the
    denominator of both rates) and is listed with its error. Each detail keeps the whole answer, the
    question id and the expected keywords — a 200-character excerpt couldn't show why a list answer
    failed (roadmap 3.15).

    ``progress_callback(done, total)`` is called after every question, failed ones included — the backend
    job mirrors it into its status; without it the UI showed "0 / 274" for the whole run (roadmap 13.6).
    ``timeout`` is per question in seconds; the backend passes one derived from the LLM timeout.
    """
    results = []
    hits = 0
    answer_matches = 0
    total_latency = 0.0
    answered = 0

    for done, pair in enumerate(qa_pairs, 1):
        question = pair["question"]
        expected_contains = pair.get("expected_answer_contains", [])
        expected_sources = pair.get("expected_sources", [])

        start = time.time()
        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"question": question},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            results.append({"id": pair.get("id", ""), "question": question, "error": str(e)})
            if progress_callback:
                progress_callback(done, len(qa_pairs))
            continue

        latency = (time.time() - start) * 1000
        total_latency += latency
        answered += 1

        answer = data.get("answer", "")
        sources = [s["file"].replace("\\", "/") for s in data.get("sources", [])]

        hit = source_hit(expected_sources, sources)   # /query returns no source ids
        if hit:
            hits += 1

        # Answer Contains: all expected keywords present
        contains_all = all(kw.lower() in answer.lower() for kw in expected_contains)
        if contains_all:
            answer_matches += 1

        results.append({
            "id": pair.get("id", ""),
            "question": question,
            "answer": answer,
            "expected_answer_contains": expected_contains,
            "sources_retrieved": sources,
            "expected_sources": expected_sources,
            "hit": hit,
            "answer_contains_match": contains_all,
            "latency_ms": latency,
        })
        if progress_callback:
            progress_callback(done, len(qa_pairs))

    total = len(qa_pairs)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_questions": total,
        "hit_rate": hits / total if total else 0,
        "answer_contains_rate": answer_matches / total if total else 0,
        # Over answered questions only — a failed request has no latency and used to pull the mean down
        "avg_latency_ms": total_latency / answered if answered else 0,
        "details": results,
    }


def main():
    """CLI entry point for running the end-to-end evaluation standalone."""
    parser = argparse.ArgumentParser(description="LoreKeeper Evaluation")
    parser.add_argument("--qa-pairs", default="evaluation/qa_pairs.yaml")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="evaluation/results/")
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa_pairs)
    print(f"Evaluating {len(qa_pairs)} questions...")

    report = evaluate(args.api_url, qa_pairs)

    report_path = save_report(report, args.output, "e2e")   # e2e_* — the API lists and cleans up by prefix

    print(f"\n{'='*40}")
    print(f"Hit Rate@K:       {report['hit_rate']:.2%}")
    print(f"Answer Contains:  {report['answer_contains_rate']:.2%}")
    print(f"Avg Latency:      {report['avg_latency_ms']:.0f}ms")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    main()
