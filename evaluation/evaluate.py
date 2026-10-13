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
from datetime import UTC, datetime

import requests

from evaluation.evaluate_retrieval import load_qa_pairs, save_report, source_hit


def evaluate(api_url: str, qa_pairs: list[dict]) -> dict:
    """Run an end-to-end evaluation against a live API endpoint.

    Sends each question as a POST request to `/query`, measuring latency and
    testing if the retrieved sources and generated answer match the expected
    values defined in the evaluation dataset. A failed request counts as a miss (it stays in the
    denominator of both rates) and is listed with its error.
    """
    results = []
    hits = 0
    answer_matches = 0
    total_latency = 0.0
    answered = 0

    for pair in qa_pairs:
        question = pair["question"]
        expected_contains = pair.get("expected_answer_contains", [])
        expected_sources = pair.get("expected_sources", [])

        start = time.time()
        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"question": question},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            results.append({"question": question, "error": str(e)})
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
            "question": question,
            "answer": answer[:200],
            "sources_retrieved": sources,
            "expected_sources": expected_sources,
            "hit": hit,
            "answer_contains_match": contains_all,
            "latency_ms": latency,
        })

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
