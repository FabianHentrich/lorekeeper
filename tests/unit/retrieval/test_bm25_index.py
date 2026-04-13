import pytest

from src.retrieval.bm25_index import BM25Index, _matches_filter, _tokenize


class TestTokenize:
    def test_lowercase_split(self):
        assert _tokenize("Arkenfeld ist GROSS") == ["arkenfeld", "ist", "gross"]

    def test_empty(self):
        assert _tokenize("") == [""]


class TestMatchesFilter:
    def test_no_filter(self):
        assert _matches_filter({"group": "lore"}, None) is True

    def test_bare_equality(self):
        meta = {"group": "lore", "source_file": "a.md"}
        assert _matches_filter(meta, {"group": "lore"}) is True
        assert _matches_filter(meta, {"group": "rules"}) is False

    def test_eq_operator(self):
        meta = {"group": "lore"}
        assert _matches_filter(meta, {"group": {"$eq": "lore"}}) is True
        assert _matches_filter(meta, {"group": {"$eq": "rules"}}) is False

    def test_ne_operator(self):
        meta = {"document_type": "markdown"}
        assert _matches_filter(meta, {"document_type": {"$ne": "image"}}) is True
        assert _matches_filter(meta, {"document_type": {"$ne": "markdown"}}) is False

    def test_in_operator(self):
        meta = {"group": "lore"}
        assert _matches_filter(meta, {"group": {"$in": ["lore", "rules"]}}) is True
        assert _matches_filter(meta, {"group": {"$in": ["adventure"]}}) is False

    def test_and_operator(self):
        meta = {"group": "lore", "document_type": "markdown"}
        where = {"$and": [
            {"group": "lore"},
            {"document_type": {"$ne": "image"}},
        ]}
        assert _matches_filter(meta, where) is True

        where_fail = {"$and": [
            {"group": "rules"},
            {"document_type": {"$ne": "image"}},
        ]}
        assert _matches_filter(meta, where_fail) is False

    def test_missing_field_fails(self):
        meta = {"group": "lore"}
        assert _matches_filter(meta, {"category": "npc"}) is False


class TestBM25Index:
    def _build_index(self):
        idx = BM25Index()
        # Simulate what build_from_vectorstore does, but with manual data
        from rank_bm25 import BM25Okapi

        docs = [
            {"id": "1", "content": "Arkenfeld ist eine Stadt im Norden", "metadata": {"group": "lore", "document_type": "markdown", "source_file": "a.md"}},
            {"id": "2", "content": "Das Langschwert kostet 15 Gold", "metadata": {"group": "rules", "document_type": "markdown", "source_file": "b.md"}},
            {"id": "3", "content": "Ein Bild der Karte", "metadata": {"group": "lore", "document_type": "image", "source_file": "c.png"}},
            {"id": "4", "content": "Arkenfeld wurde von Elfen gegründet", "metadata": {"group": "lore", "document_type": "markdown", "source_file": "d.md"}},
        ]
        idx._documents = docs
        corpus = [_tokenize(d["content"]) for d in docs]
        idx._index = BM25Okapi(corpus)
        idx._is_built = True
        return idx

    def test_query_returns_results(self):
        idx = self._build_index()
        results = idx.query("Arkenfeld", top_k=5)
        assert len(results) >= 1
        # Both Arkenfeld docs should be returned
        ids = [r["id"] for r in results]
        assert "1" in ids
        assert "4" in ids

    def test_query_ranked_by_score(self):
        idx = self._build_index()
        results = idx.query("Langschwert Gold", top_k=5)
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_metadata_filter_applied(self):
        idx = self._build_index()
        results = idx.query("Arkenfeld", top_k=5, where={"group": "rules"})
        # Arkenfeld docs are group=lore, so they should be filtered out
        assert all(r["metadata"]["group"] == "rules" for r in results)

    def test_exclude_image_filter(self):
        idx = self._build_index()
        results = idx.query("Karte Bild", top_k=5, where={"document_type": {"$ne": "image"}})
        assert all(r["metadata"]["document_type"] != "image" for r in results)

    def test_empty_index_returns_empty(self):
        idx = BM25Index()
        results = idx.query("anything", top_k=5)
        assert results == []

    def test_is_built_default_false(self):
        idx = BM25Index()
        assert idx.is_built is False

    def test_invalidate(self):
        idx = self._build_index()
        assert idx.is_built is True
        idx.invalidate()
        assert idx.is_built is False
        assert idx.query("anything") == []

    def test_top_k_limits_results(self):
        idx = self._build_index()
        results = idx.query("Arkenfeld", top_k=1)
        assert len(results) <= 1

    def test_query_result_format(self):
        idx = self._build_index()
        results = idx.query("Arkenfeld", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "metadata" in r
        assert "score" in r
        assert isinstance(r["score"], float)
