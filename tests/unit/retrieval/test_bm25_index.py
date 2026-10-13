
from src.retrieval.bm25_index import BM25Index, _matches_filter, _tokenize


class TestTokenize:
    """Test suite validating keyword extraction logic for BM25 term indexing."""

    def test_lowercase_split(self):
        """Verify standard inputs are correctly lowercased and segmented by whitespace."""
        assert _tokenize("Arkenfeld ist GROSS") == ["arkenfeld", "ist", "gross"]

    def test_empty(self):
        """Verify empty strings map safely to void arrays."""
        assert _tokenize("") == []

    def test_strips_punctuation(self):
        """Verify punctuation is dropped so query 'Arkenfeld?' matches document 'Arkenfeld.'."""
        assert _tokenize("Wer ist Arkenfeld?") == ["wer", "ist", "arkenfeld"]
        assert _tokenize("(Arkenfeld), Größe: 3.") == ["arkenfeld", "größe", "3"]


class TestMatchesFilter:
    """Test suite validating subset ChromaDB pseudo-query filtering implementations."""

    def test_no_filter(self):
        """Verify omitting where-clauses permits everything."""
        assert _matches_filter({"group": "lore"}, None) is True

    def test_bare_equality(self):
        """Verify implicit direct equality shorthand resolves true/false correctly."""
        meta = {"group": "lore", "source_file": "a.md"}
        assert _matches_filter(meta, {"group": "lore"}) is True
        assert _matches_filter(meta, {"group": "rules"}) is False

    def test_eq_operator(self):
        """Verify explicit `$eq` operator parsing."""
        meta = {"group": "lore"}
        assert _matches_filter(meta, {"group": {"$eq": "lore"}}) is True
        assert _matches_filter(meta, {"group": {"$eq": "rules"}}) is False

    def test_ne_operator(self):
        """Verify negation `$ne` operators block matching values."""
        meta = {"document_type": "markdown"}
        assert _matches_filter(meta, {"document_type": {"$ne": "image"}}) is True
        assert _matches_filter(meta, {"document_type": {"$ne": "markdown"}}) is False

    def test_in_operator(self):
        """Verify array subset inclusion paths via `$in`."""
        meta = {"group": "lore"}
        assert _matches_filter(meta, {"group": {"$in": ["lore", "rules"]}}) is True
        assert _matches_filter(meta, {"group": {"$in": ["adventure"]}}) is False

    def test_and_operator(self):
        """Verify composite logical `$and` grouping requirements strictly constrain matches."""
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
        """Verify searching for fields entirely absent from metadata rejects the document."""
        meta = {"group": "lore"}
        assert _matches_filter(meta, {"category": "npc"}) is False


class _FakeStore:
    """Stand-in for VectorStoreService.get_all()."""

    def __init__(self, docs, on_read=None):
        self.docs, self.on_read, self.reads = docs, on_read, 0

    def get_all(self, include):
        self.reads += 1
        if self.on_read:
            self.on_read()
        return {"ids": [d["id"] for d in self.docs], "documents": [d["content"] for d in self.docs],
                "metadatas": [d["metadata"] for d in self.docs]}


class TestBM25Index:
    """Test suite verifying lexical scoring indexes and background cache lifecycle models."""

    def _build_index(self):
        """
        Helper method manually generating a simulated BM25 space to bypass
        expensive vector database crawling tests.
        """
        idx = BM25Index()
        docs = [
            {"id": "1", "content": "Arkenfeld ist eine Stadt im Norden", "metadata": {"group": "lore", "document_type": "markdown", "source_file": "a.md"}},
            {"id": "2", "content": "Das Langschwert kostet 15 Gold", "metadata": {"group": "rules", "document_type": "markdown", "source_file": "b.md"}},
            {"id": "3", "content": "Ein Bild der Karte", "metadata": {"group": "lore", "document_type": "image", "source_file": "c.png"}},
            {"id": "4", "content": "Arkenfeld wurde von Elfen gegründet", "metadata": {"group": "lore", "document_type": "markdown", "source_file": "d.md"}},
        ]
        idx.ensure_built(_FakeStore(docs))
        return idx

    def test_query_returns_results(self):
        """Verify the index maps semantic strings correctly to their underlying document ids."""
        idx = self._build_index()
        results = idx.query("Arkenfeld", top_k=5)
        assert len(results) >= 1
        # Both Arkenfeld docs should be returned
        ids = [r["id"] for r in results]
        assert "1" in ids
        assert "4" in ids

    def test_query_ranked_by_score(self):
        """Verify multi-term returns sort appropriately descending by TF-IDF scoring."""
        idx = self._build_index()
        results = idx.query("Langschwert Gold", top_k=5)
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_metadata_filter_applied(self):
        """Verify BM25 searches accurately restrict results mirroring Chroma's 'where' clauses."""
        idx = self._build_index()
        results = idx.query("Arkenfeld", top_k=5, where={"group": "rules"})
        # Arkenfeld docs are group=lore, so they should be filtered out
        assert all(r["metadata"]["group"] == "rules" for r in results)

    def test_exclude_image_filter(self):
        """Verify `$ne` logic filters apply during BM25 evaluations natively mapping image suppressions."""
        idx = self._build_index()
        results = idx.query("Karte Bild", top_k=5, where={"document_type": {"$ne": "image"}})
        assert all(r["metadata"]["document_type"] != "image" for r in results)

    def test_empty_index_returns_empty(self):
        """Verify queries across unpopulated or empty index instances degrade safely."""
        idx = BM25Index()
        results = idx.query("anything", top_k=5)
        assert results == []

    def test_is_built_default_false(self):
        """Verify new indexes declare invalidation bounds accurately until populated."""
        idx = BM25Index()
        assert idx.is_built is False

    def test_invalidate(self):
        """Verify caching controls toggle invalidation appropriately flushing active states."""
        idx = self._build_index()
        assert idx.is_built is True
        idx.invalidate()
        assert idx.is_built is False
        assert idx.query("anything") == []

    def test_top_k_limits_results(self):
        """Verify length constrictions cap array yields accordingly."""
        idx = self._build_index()
        results = idx.query("Arkenfeld", top_k=1)
        assert len(results) <= 1

    def test_query_result_format(self):
        """Verify the structural composition of search hits for RRF pipeline integration."""
        idx = self._build_index()
        results = idx.query("Arkenfeld", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "metadata" in r
        assert "score" in r
        assert isinstance(r["score"], float)


class TestConcurrentBuild:
    """Two first hybrid queries at once must not build into the same lists (roadmap 11.0 #4)."""

    def test_parallel_first_queries_build_once_and_find_the_right_document(self):
        import sys
        import threading

        docs = [{"id": f"id{i}", "content": f"wort{i} text", "metadata": {}} for i in range(300)]
        store = _FakeStore(docs)
        idx = BM25Index()
        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # force thread switches inside the build — made the old race 20/20
        try:
            threads = [threading.Thread(target=idx.ensure_built, args=(store,)) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(old)

        assert store.reads == 1
        assert idx.query("wort7", top_k=1)[0]["id"] == "id7"

    def test_build_running_during_invalidate_is_discarded(self):
        """A build that read the collection before a delete/wipe must not publish the deleted chunks."""
        idx = BM25Index()
        store = _FakeStore([{"id": "gone", "content": "geloeschte Quelle", "metadata": {}}],
                           on_read=idx.invalidate)  # the collection changes while the build runs
        idx.ensure_built(store)

        assert idx.is_built is False
        assert idx.query("geloeschte") == []
