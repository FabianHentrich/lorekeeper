"""
LoreKeeper Retrieval Evaluation

Evaluates retrieval quality directly against ChromaDB — no LLM call needed.
Measures Hit Rate@K per source_type and category.

Usage:
    python -m evaluation.evaluate_retrieval
    python -m evaluation.evaluate_retrieval --top-k 20 --top-k-rerank 10
    python -m evaluation.evaluate_retrieval --qa-pairs evaluation/qa_pairs.yaml --output evaluation/results/
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_qa_pairs(path: str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data.get("pairs", [])


async def run_evaluation_with_retriever(
    qa_pairs: list[dict],
    retriever,
    top_k: int,
    top_k_rerank: int,
    max_per_source: int = 3,
    progress_callback=None,
) -> dict:
    """Run retrieval evaluation using an existing Retriever instance.

    Called by the API route (shared retriever) and by run_evaluation() (own retriever).
    progress_callback(current, total) is called after each question.
    """
    results = []
    hits = 0
    total = len(qa_pairs)

    for idx, pair in enumerate(qa_pairs):
        question = pair["question"]
        expected_sources = pair.get("expected_sources", [])
        source_type = pair.get("source_type", "unknown")
        category = pair.get("category", "unknown")

        start = time.time()
        try:
            chunks = await retriever.retrieve(
                query=question,
                top_k=top_k,
                top_k_rerank=top_k_rerank,
                max_per_source=max_per_source,
            )
        except Exception as e:
            logger.error("Eval [%s/%s] id=%s error: %s", idx + 1, total, pair.get("id", "?"), e)
            results.append({
                "id": pair.get("id", "?"),
                "question": question,
                "source_type": source_type,
                "category": category,
                "error": str(e),
                "hit": False,
            })
            if progress_callback:
                progress_callback(idx + 1, total)
            continue

        latency_ms = (time.time() - start) * 1000
        retrieved_files = [c.source_file for c in chunks]

        # Hit: any expected source filename appears in retrieved source_files
        hit = any(
            any(exp.split("/")[-1] in rf for rf in retrieved_files)
            for exp in expected_sources
        )
        if hit:
            hits += 1

        top_score = round(chunks[0].score, 4) if chunks else 0.0
        logger.debug(
            "Eval [%s/%s] id=%s %s score=%.4f latency=%dms",
            idx + 1, total, pair.get("id", "?"),
            "HIT" if hit else "MISS", top_score, round(latency_ms),
        )

        results.append({
            "id": pair.get("id", "?"),
            "question": question,
            "source_type": source_type,
            "category": category,
            "expected_sources": expected_sources,
            "retrieved_files": retrieved_files,
            "hit": hit,
            "top_score": top_score,
            "latency_ms": round(latency_ms),
        })

        if progress_callback:
            progress_callback(idx + 1, total)

    hit_rate = hits / total if total else 0
    avg_latency = sum(r.get("latency_ms", 0) for r in results if "error" not in r) / max(len(results), 1)
    logger.info(
        "Retrieval eval complete: %d/%d hits (%.1f%%), Ø latency=%dms, top_k=%d, top_k_rerank=%d, max_per_source=%d",
        hits, total, hit_rate * 100, avg_latency, top_k, top_k_rerank, max_per_source,
    )

    # Per source_type breakdown
    breakdown: dict[str, dict] = {}
    for r in results:
        st = r["source_type"]
        if st not in breakdown:
            breakdown[st] = {"total": 0, "hits": 0}
        breakdown[st]["total"] += 1
        if r.get("hit"):
            breakdown[st]["hits"] += 1

    breakdown_rates = {
        st: {
            "hit_rate": round(v["hits"] / v["total"], 3) if v["total"] else 0,
            "hits": v["hits"],
            "total": v["total"],
        }
        for st, v in breakdown.items()
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": top_k,
            "top_k_rerank": top_k_rerank,
            "max_per_source": max_per_source,
        },
        "total_questions": total,
        "hit_rate": round(hit_rate, 3),
        "breakdown_by_source_type": breakdown_rates,
        "misses": [r for r in results if not r.get("hit") and "error" not in r],
        "errors": [r for r in results if "error" in r],
        "details": results,
    }


def run_evaluation(qa_pairs: list[dict], top_k: int, top_k_rerank: int,
                   max_per_source: int = 3) -> dict:
    """CLI-compatible sync wrapper. Creates its own Retriever."""
    from src.config.manager import config_manager
    from src.retrieval.embeddings import EmbeddingService
    from src.retrieval.vectorstore import VectorStoreService
    from src.retrieval.retriever import Retriever
    import asyncio

    settings = config_manager.settings
    cfg = settings.retrieval

    embedding_service = EmbeddingService(settings.embeddings)
    vectorstore = VectorStoreService(settings.vectorstore, embedding_service)
    retriever = Retriever(cfg, embedding_service, vectorstore)

    return asyncio.run(run_evaluation_with_retriever(
        qa_pairs, retriever, top_k, top_k_rerank, max_per_source,
    ))


def main():
    parser = argparse.ArgumentParser(description="LoreKeeper Retrieval Evaluation")
    parser.add_argument("--qa-pairs", default="evaluation/qa_pairs.yaml")
    parser.add_argument("--output", default="evaluation/results/")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-k-rerank", type=int, default=None)
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa_pairs)

    # Use config defaults if not overridden
    from src.config.manager import config_manager
    cfg = config_manager.settings.retrieval
    top_k = args.top_k or cfg.top_k
    top_k_rerank = args.top_k_rerank or cfg.reranking.top_k_rerank

    print(f"Evaluating {len(qa_pairs)} questions (top_k={top_k}, top_k_rerank={top_k_rerank})...")

    report = run_evaluation(qa_pairs, top_k, top_k_rerank)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*45}")
    print(f"Hit Rate (gesamt):  {report['hit_rate']:.1%}  ({sum(r['hit'] for r in report['details'])}/{report['total_questions']})")
    print()
    for st, stats in report["breakdown_by_source_type"].items():
        print(f"  {st:<12} {stats['hit_rate']:.1%}  ({stats['hits']}/{stats['total']})")
    print()
    if report["misses"]:
        print("Misses:")
        for m in report["misses"]:
            print(f"  [{m['id']}] {m['question'][:60]}")
            print(f"        erwartet: {m['expected_sources']}")
            print(f"        erhalten: {m['retrieved_files'][:3]}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
