import logging
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing.

    Sufficient for German TTRPG content at this scale. No stemmer needed —
    the cross-encoder reranker compensates for morphological variation.
    """
    return text.lower().split()


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


class BM25Index:
    def __init__(self):
        self._index: BM25Okapi | None = None
        self._documents: list[dict[str, Any]] = []
        self._doc_token_sets: list[set[str]] = []
        self._is_built = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    def build_from_vectorstore(self, vectorstore) -> None:
        """Load all documents from ChromaDB and build the BM25 index."""
        collection = vectorstore._get_collection()
        data = collection.get(include=["documents", "metadatas"])

        if not data or not data.get("ids"):
            self._index = None
            self._documents = []
            self._doc_token_sets = []
            self._is_built = True
            logger.info("BM25 index built (empty — no documents in vectorstore)")
            return

        self._documents = []
        corpus: list[list[str]] = []

        for i, doc_id in enumerate(data["ids"]):
            content = data["documents"][i] or ""
            metadata = data["metadatas"][i] or {}

            self._documents.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
            })
            corpus.append(_tokenize(content))

        self._index = BM25Okapi(corpus)
        self._doc_token_sets = [set(toks) for toks in corpus]
        self._is_built = True
        logger.info(f"BM25 index built: {len(self._documents)} documents")

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
        if not self._is_built or self._index is None or not self._documents:
            return []

        tokens = _tokenize(query_text)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        query_tokens = set(tokens)

        # Pair scores with documents, filter by metadata, sort by score desc.
        # When IDF is 0 (term appears in ≥half the corpus — common on tiny
        # corpora) BM25 returns 0 for all docs. Fall back to token-overlap
        # count so exact matches are still retrievable.
        scored = []
        for idx, score in enumerate(scores):
            doc = self._documents[idx]
            if score <= 0:
                overlap = len(query_tokens & self._doc_token_sets[idx])
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
        """Mark index as stale. Next hybrid query will rebuild."""
        self._is_built = False
        self._index = None
        self._documents = []
        self._doc_token_sets = []
