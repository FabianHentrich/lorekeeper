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
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def section_hit(expected_section: str, headings: list[str]) -> bool:
    """True if any retrieved heading lies in the expected section.

    Headings look like "**4.10 Kampfunfähigkeit und Tod** > **Kampfunfähigkeit**";
    "4.10" matches that and sub-sections ("5.4" matches "5.4.1 Krieger") but not
    "4.1 Der Ablauf eines Kampfes" for "4.10". Unnumbered headings match by name:
    "Gesamtwertung" matches "Gesamtwertung (R3 vs Median aus 6 Profilen)".
    """
    for heading in headings:
        for part in heading.replace("*", "").split(" > "):
            part = part.strip()
            if (part == expected_section or part.startswith(expected_section + " ")
                    or part.startswith(expected_section + ".")):
                return True
    return False


def load_qa_pairs(path: str) -> list[dict]:
    """Parse the configured YAML file returning a list of dicts with questions and expected formats."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data.get("pairs", [])


async def run_evaluation_with_retriever(
    qa_pairs: list[dict],
    retriever,
    top_k: int,
    top_k_rerank: int,
    max_per_source: int = 3,
    hybrid: bool | None = None,
    progress_callback=None,
) -> dict:
    """Run retrieval evaluation using an existing Retriever instance.

    Called by the API route (shared retriever) and by run_evaluation() (own retriever).
    progress_callback(current, total) is called after each question.
    """
    results = []
    hits = 0
    section_hits = 0
    section_total = 0
    total = len(qa_pairs)

    for idx, pair in enumerate(qa_pairs):
        question = pair["question"]
        expected_sources = pair.get("expected_sources", [])
        expected_source_ids = pair.get("expected_source_ids", [])
        source_type = pair.get("source_type", "unknown")
        category = pair.get("category", "unknown")

        start = time.time()
        try:
            chunks = await retriever.retrieve(
                query=question,
                top_k=top_k,
                top_k_rerank=top_k_rerank,
                max_per_source=max_per_source,
                hybrid=hybrid,
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
        retrieved_headings = [c.heading for c in chunks]
        retrieved_source_ids = [c.metadata.get("source_id", "") for c in chunks]

        # Hit: any expected source filename appears in retrieved source_files.
        # expected_source_ids additionally pins the source — for equally named files
        # in different sources (e.g. two simulation llm_report.md).
        hit = any(
            any(exp.split("/")[-1] in rf for rf, sid in zip(retrieved_files, retrieved_source_ids, strict=True)
                if not expected_source_ids or sid in expected_source_ids)
            for exp in expected_sources
        )
        if hit:
            hits += 1

        # File-level hits are near-meaningless for one large PDF — every chunk is from it
        expected_section = pair.get("expected_section", "")
        sec_hit: bool | None = None
        if expected_section:
            section_total += 1
            sec_hit = section_hit(expected_section, retrieved_headings)
            if sec_hit:
                section_hits += 1

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
            "expected_source_ids": expected_source_ids,
            "retrieved_files": retrieved_files,
            "retrieved_source_ids": retrieved_source_ids,
            "retrieved_headings": retrieved_headings,
            "hit": hit,
            "expected_section": expected_section,
            "section_hit": sec_hit,
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
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "top_k": top_k,
            "top_k_rerank": top_k_rerank,
            "max_per_source": max_per_source,
            "hybrid": hybrid,
        },
        "total_questions": total,
        "hit_rate": round(hit_rate, 3),
        "section_hit_rate": round(section_hits / section_total, 3) if section_total else None,
        "section_total": section_total,
        "breakdown_by_source_type": breakdown_rates,
        "misses": [r for r in results if not r.get("hit") and "error" not in r],
        "section_misses": [r for r in results if r.get("section_hit") is False],
        "errors": [r for r in results if "error" in r],
        "details": results,
    }


def run_evaluation(qa_pairs: list[dict], top_k: int, top_k_rerank: int,
                   max_per_source: int = 3, hybrid: bool | None = None) -> dict:
    """CLI-compatible sync wrapper that instantiates its own Retriever.

    Loads configuration settings, creates the necessary Embedding and VectorStore dependencies,
    and runs the asynchronous evaluation process blocking the thread until completion.
    """
    import asyncio

    from src.config.manager import config_manager
    from src.retrieval.embeddings import EmbeddingService
    from src.retrieval.retriever import Retriever
    from src.retrieval.vectorstore import VectorStoreService

    settings = config_manager.settings
    cfg = settings.retrieval

    embedding_service = EmbeddingService(settings.embeddings)
    vectorstore = VectorStoreService(settings.vectorstore, embedding_service)
    retriever = Retriever(cfg, embedding_service, vectorstore)

    return asyncio.run(run_evaluation_with_retriever(
        qa_pairs, retriever, top_k, top_k_rerank, max_per_source, hybrid=hybrid,
    ))


def main():
    """CLI entry-point. Parses command line arguments to conduct a pure retrieval evaluation.
    Outputs a consolidated JSON report and a console summary detailing Hit Rate@K.
    """
    # Headings can contain emoji — the Windows console (cp1252) would crash on printing them
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="LoreKeeper Retrieval Evaluation")
    parser.add_argument("--qa-pairs", default="evaluation/qa_pairs.yaml")
    parser.add_argument("--output", default="evaluation/results/")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-k-rerank", type=int, default=None)
    parser.add_argument("--hybrid", dest="hybrid", action="store_true", default=None,
                        help="Force hybrid BM25+vector retrieval on (overrides config)")
    parser.add_argument("--no-hybrid", dest="hybrid", action="store_false",
                        help="Force pure vector retrieval (overrides config)")
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa_pairs)

    # Use config defaults if not overridden
    from src.config.manager import config_manager
    cfg = config_manager.settings.retrieval
    top_k = args.top_k or cfg.top_k
    top_k_rerank = args.top_k_rerank or cfg.reranking.top_k_rerank

    hybrid_label = "config" if args.hybrid is None else ("on" if args.hybrid else "off")
    print(f"Evaluating {len(qa_pairs)} questions "
          f"(top_k={top_k}, top_k_rerank={top_k_rerank}, hybrid={hybrid_label})...")

    report = run_evaluation(qa_pairs, top_k, top_k_rerank, hybrid=args.hybrid)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*45}")
    hits = sum(r['hit'] for r in report['details'])
    print(f"Hit Rate (gesamt):  {report['hit_rate']:.1%}  ({hits}/{report['total_questions']})")
    print()
    for st, stats in report["breakdown_by_source_type"].items():
        print(f"  {st:<12} {stats['hit_rate']:.1%}  ({stats['hits']}/{stats['total']})")
    if report["section_total"]:
        sec_hits = round(report["section_hit_rate"] * report["section_total"])
        print(f"  {'section':<12} {report['section_hit_rate']:.1%}  ({sec_hits}/{report['section_total']})")
    print()
    if report["misses"]:
        print("Misses:")
        for m in report["misses"]:
            print(f"  [{m['id']}] {m['question'][:60]}")
            print(f"        erwartet: {m['expected_sources']}")
            print(f"        erhalten: {m['retrieved_files'][:3]}")
    if report["section_misses"]:
        print("Section-Misses:")
        for m in report["section_misses"]:
            print(f"  [{m['id']}] {m['question'][:60]}")
            print(f"        erwartet: Abschnitt {m['expected_section']}")
            print(f"        erhalten: {m['retrieved_headings'][:3]}")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
