"""Keyword index for hybrid search: rank_bm25 over the whole collection, held in memory.

Built lazily on the first hybrid query, invalidated after every ingest job. ``_matches_filter`` must
evaluate the same where-subset as ChromaDB (``access.validate_filters`` admits nothing else) — otherwise
the player filter could leak through the keyword path. tests/integration/test_filter_contract.py checks
that both paths return the same.
"""
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokenizer; punctuation is dropped so "Arkenfeld?" matches "Arkenfeld.".

    Sufficient for German TTRPG content at this scale. No stemmer needed —
    the cross-encoder reranker compensates for morphological variation.
    """
    return _WORD_PATTERN.findall(text.lower())


def _matches_filter(metadata: dict, where: dict | None) -> bool:
    """Evaluate a ChromaDB-style where clause against a metadata dict.

    Supported operators: $eq, $ne, $in, $and, and bare equality.
    """
    if not where:
        return True

    for key, condition in where.items():
        if key == "$and":
            if not all(_matches_filter(metadata, sub) for sub in condition):
                return False
        elif isinstance(condition, dict):
            for op, val in condition.items():
                actual = metadata.get(key)
                if op == "$eq" and actual != val:
                    return False
                if op == "$ne" and actual == val:
                    return False
                if op == "$in" and actual not in val:
                    return False
        else:
            # Bare equality: {"field": "value"}
            if metadata.get(key) != condition:
                return False

    return True


@dataclass(frozen=True)
class _Snapshot:
    """One built index: documents, their token sets and the BM25 model — always replaced as a whole."""
    documents: list[dict[str, Any]]
    token_sets: list[set[str]]
    index: BM25Okapi | None


class BM25Index:
    """In-memory BM25 keyword index built lazily from the ChromaDB collection.

    Used as the lexical half of hybrid retrieval: results are merged with
    vector-search hits via Reciprocal Rank Fusion in ``Retriever``.

    Thread safety (queries run in the threadpool): the built index is one immutable ``_Snapshot``,
    published in a single assignment and read once per query. Building happens under a lock, so two
    first hybrid queries don't build twice into the same lists — that used to mix two corpora and map
    scores to the wrong documents (roadmap 11.0 #4). ``invalidate`` bumps a generation counter; a
    build that started before it is not published (it would contain deleted chunks).
    """

    def __init__(self):
        """Create an empty, not-yet-built index. ``ensure_built()`` populates it."""
        self._snapshot: _Snapshot | None = None
        self._generation = 0
        self._build_lock = threading.Lock()

    @property
    def is_built(self) -> bool:
        """True once the index has been populated (or verified empty)."""
        return self._snapshot is not None

    def ensure_built(self, vectorstore) -> None:
        """Build the index unless a current one exists; concurrent callers wait for one build."""
        if self._snapshot is not None:
            return
        with self._build_lock:
            if self._snapshot is None:
                self.build_from_vectorstore(vectorstore)

    def build_from_vectorstore(self, vectorstore) -> None:
        """Load all documents from ChromaDB and build the BM25 index (prefer ``ensure_built``)."""
        generation = self._generation
        data = vectorstore.get_all(include=["documents", "metadatas"])

        documents: list[dict[str, Any]] = []
        corpus: list[list[str]] = []
        for i, doc_id in enumerate((data or {}).get("ids") or []):
            content = data["documents"][i] or ""
            documents.append({"id": doc_id, "content": content, "metadata": data["metadatas"][i] or {}})
            corpus.append(_tokenize(content))

        snapshot = _Snapshot(
            documents=documents,
            token_sets=[set(toks) for toks in corpus],
            index=BM25Okapi(corpus) if corpus else None,
        )
        if generation != self._generation:
            logger.info("BM25 build discarded — the index was invalidated while it was built")
            return
        self._snapshot = snapshot
        logger.info(f"BM25 index built: {len(documents)} documents")

    def query(
        self,
        query_text: str,
        top_k: int = 15,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 search with optional metadata filter.

        Returns list[{id, content, metadata, score}] — same format as
        VectorStoreService.query() for easy merging.
        """
        snapshot = self._snapshot  # read once: invalidate() or a rebuild may replace it meanwhile
        if snapshot is None or snapshot.index is None:
            return []

        tokens = _tokenize(query_text)
        if not tokens:
            return []

        scores = snapshot.index.get_scores(tokens)
        query_tokens = set(tokens)

        # Pair scores with documents, filter by metadata, sort by score desc.
        # When IDF is 0 (term appears in ≥half the corpus — common on tiny
        # corpora) BM25 returns 0 for all docs. Fall back to token-overlap
        # count so exact matches are still retrievable.
        scored = []
        for idx, score in enumerate(scores):
            doc = snapshot.documents[idx]
            if score <= 0:
                overlap = len(query_tokens & snapshot.token_sets[idx])
                if overlap == 0:
                    continue
                score = float(overlap) * 1e-6  # positive but below any real BM25 score
            if not _matches_filter(doc["metadata"], where):
                continue
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": score,
            }
            for score, doc in scored[:top_k]
        ]

    def invalidate(self) -> None:
        """Mark the index as stale (after anything that changed the collection). The next hybrid query
        rebuilds it; a build already running is discarded."""
        self._generation += 1
        self._snapshot = None
