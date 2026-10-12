"""Contract between the filter check and the two search paths (review 7.2, N1/N2).

Every filter ``validate_filters`` lets through must be accepted by ChromaDB *and* select exactly
the same chunks in the BM25 path — otherwise hybrid search would silently ignore a condition or a
request would end in a 500.
"""
import uuid

import chromadb
import pytest

from src.api.access import validate_filters
from src.retrieval.bm25_index import _matches_filter

DOCS = [
    {"source_id": "a", "group": "lore", "content_category": "npc", "chunk_index": 0},
    {"source_id": "a", "group": "lore", "content_category": "location", "chunk_index": 1},
    {"source_id": "b", "group": "rules", "content_category": "rules", "chunk_index": 0},
    {"source_id": "c", "group": "simulation", "content_category": "report", "chunk_index": 2},
]

ACCEPTED = [
    {"group": "rules"},
    {"source_id": {"$eq": "a"}},
    {"group": {"$ne": "simulation"}},
    {"chunk_index": {"$in": [0, 2]}},
    {"group": {"$in": ["lore", "rules"]}, "content_category": {"$in": ["npc", "rules"]}},
    {"$and": [{"source_id": "a"}, {"chunk_index": {"$ne": 0}}]},
    {"$and": [{"group": {"$in": ["lore", "rules"]}}, {"$and": [{"source_id": {"$ne": "b"}}, {"chunk_index": 0}]}]},
]


@pytest.fixture(scope="module")
def collection():
    col = chromadb.EphemeralClient().create_collection(f"contract_{uuid.uuid4().hex}")
    col.add(ids=[str(i) for i in range(len(DOCS))], documents=[f"doc {i}" for i in range(len(DOCS))],
            embeddings=[[1.0, float(i)] for i in range(len(DOCS))], metadatas=DOCS)
    return col


def _as_where(filters: dict) -> dict:
    """What the retriever builds: each top-level key becomes its own condition."""
    conditions = [{k: v} for k, v in filters.items()]
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@pytest.mark.parametrize("filters", ACCEPTED)
def test_accepted_filters_select_the_same_chunks_in_chromadb_and_bm25(collection, filters):
    validate_filters(filters)   # precondition: the check lets it through
    where = _as_where(filters)

    chroma_ids = set(collection.get(where=where)["ids"])
    bm25_ids = {str(i) for i, meta in enumerate(DOCS) if _matches_filter(meta, where)}

    assert chroma_ids == bm25_ids, (where, chroma_ids, bm25_ids)
