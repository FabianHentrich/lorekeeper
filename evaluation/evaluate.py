"""
LoreKeeper Evaluation Script

Usage:
    python -m evaluation.evaluate \
        --qa-pairs evaluation/qa_pairs.yaml \
        --settings config/settings.yaml \
        --output evaluation/results/
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


def load_qa_pairs(path: str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data.get("pairs", [])


def evaluate(api_url: str, qa_pairs: list[dict]) -> dict:
    results = []
    hits = 0
    answer_matches = 0
    total_latency = 0

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

        answer = data.get("answer", "")
        sources = [s["file"].replace("\\", "/") for s in data.get("sources", [])]

        # Hit Rate: expected source filename in retrieved sources (path-separator agnostic)
        hit = any(
            any(exp.split("/")[-1] in src for src in sources)
            for exp in expected_sources
        )
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": total,
        "hit_rate": hits / total if total else 0,
        "answer_contains_rate": answer_matches / total if total else 0,
        "avg_latency_ms": total_latency / total if total else 0,
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description="LoreKeeper Evaluation")
    parser.add_argument("--qa-pairs", default="evaluation/qa_pairs.yaml")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", default="evaluation/results/")
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa_pairs)
    print(f"Evaluating {len(qa_pairs)} questions...")

    report = evaluate(args.api_url, qa_pairs)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*40}")
    print(f"Hit Rate@K:       {report['hit_rate']:.2%}")
    print(f"Answer Contains:  {report['answer_contains_rate']:.2%}")
    print(f"Avg Latency:      {report['avg_latency_ms']:.0f}ms")
    print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    main()
