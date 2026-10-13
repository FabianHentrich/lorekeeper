from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.manager import HybridSearchConfig, RerankingConfig, RetrievalConfig, RulesSlotConfig
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import Retriever, _rrf_merge


@pytest.fixture
def config():
    """Provides a baseline RetrievalConfig matching Chroma parameters with reranking mocked out."""
    return RetrievalConfig(top_k=5, score_threshold=0.5, reranking=RerankingConfig(enabled=False))


@pytest.fixture
def embed_service():
    """Provides a sterile mock of the embedding service returning arbitrary constants."""
    svc = AsyncMock()
    svc.embed_text.return_value = [0.1, 0.2, 0.3]
    return svc


@pytest.fixture
def vectorstore():
    """Provides a Mock vectorstore configured to return synthetic result sets representing typical Chroma hits."""
    vs = MagicMock()
    vs.query.return_value = [
        {
            "id": "1",
            "content": "Arkenfeld is a city.",
            "metadata": {"source_file": "Orte/Arkenfeld.md", "document_type": "markdown", "heading_hierarchy": "Orte > Arkenfeld"},
            "score": 0.9,
        },
        {
            "id": "2",
            "content": "A low-relevance chunk.",
            "metadata": {"source_file": "misc.md", "document_type": "markdown", "heading_hierarchy": ""},
            "score": 0.3,  # below threshold
        },
    ]
    return vs


@pytest.fixture
def retriever(config, embed_service, vectorstore):
    """Provides a completely integrated test Retriever combining all mocked subsystems."""
    return Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)


class TestRetrieve:
    """Test suite targeting simple retrieval evaluations absent complex reranking loops."""

    @pytest.mark.asyncio
    async def test_returns_chunks_above_threshold(self, retriever):
        """Verify queries successfully return mapped dictionaries filtered against the threshold."""
        results = await retriever.retrieve("where is arkenfeld")
        assert len(results) == 1
        assert results[0].content == "Arkenfeld is a city."

    @pytest.mark.asyncio
    async def test_filters_out_low_scores(self, retriever):
        """Verify vectorstore hits dipping beneath `score_threshold` are stripped proactively."""
        results = await retriever.retrieve("something")
        scores = [r.score for r in results]
        assert all(s >= 0.5 for s in scores)

    @pytest.mark.asyncio
    async def test_chunk_fields_populated(self, retriever):
        """Verify schema projection extracts fundamental metadata cleanly from the unrolled DB representation."""
        results = await retriever.retrieve("query")
        chunk = results[0]
        assert chunk.source_file == "Orte/Arkenfeld.md"
        assert chunk.document_type == "markdown"
        assert chunk.heading == "Orte > Arkenfeld"
        assert chunk.score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_stage_timings_reported(self, retriever):
        """Per-stage durations land in the caller's dict; disabled stages are absent."""
        timings: dict[str, float] = {}
        await retriever.retrieve("where is arkenfeld", timings=timings)
        assert set(timings) == {"embed", "search", "expand"}
        assert all(v >= 0 for v in timings.values())

    @pytest.mark.asyncio
    async def test_image_type_excluded_via_filter(self, retriever, vectorstore):
        """Verify standard query constraints forcefully eject inline image references from standard text search."""
        await retriever.retrieve("query")
        kwargs = vectorstore.query.call_args[1]
        where = kwargs["where"]
        # must contain the image exclusion filter
        assert where is not None
        assert "document_type" in str(where)

    @pytest.mark.asyncio
    async def test_metadata_filters_combined(self, retriever, vectorstore):
        """Verify extra constraints stack gracefully via the `$and` implicit merger."""
        vectorstore.query.return_value = []
        await retriever.retrieve("query", metadata_filters={"category": "npc"})
        kwargs = vectorstore.query.call_args[1]
        where = kwargs["where"]
        assert "$and" in where

    @pytest.mark.asyncio
    async def test_custom_top_k(self, retriever, vectorstore):
        """Verify specific overrides map cleanly bypassing config default constraints."""
        vectorstore.query.return_value = []
        await retriever.retrieve("query", top_k=10)
        kwargs = vectorstore.query.call_args[1]
        assert kwargs["top_k"] == 10

    @pytest.mark.asyncio
    async def test_empty_vectorstore_returns_empty(self, retriever, vectorstore):
        """Verify missing results yield clean empty lists devoid of parsing crashes."""
        vectorstore.query.return_value = []
        results = await retriever.retrieve("nothing here")
        assert results == []


class TestRerank:
    """Test suite validating cross-encoder interactions enforcing relevance scoring over dense space assumptions."""

    @pytest.mark.asyncio
    async def test_reranking_reorders_chunks(self, config, embed_service, vectorstore):
        """Verify the cross-encoder correctly manipulates list indexing according to deep contextual evaluations."""
        config.reranking = RerankingConfig(enabled=True, top_k_rerank=2)
        vectorstore.query.return_value = [
            {"id": "1", "content": "chunk A", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.8},
            {"id": "2", "content": "chunk B", "metadata": {"source_file": "b.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.7},
            {"id": "3", "content": "chunk C", "metadata": {"source_file": "c.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.6},
        ]
        retriever = Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)

        mock_reranker = MagicMock()
        # reranker scores: C > A > B
        mock_reranker.predict.return_value = [0.5, 0.9, 0.3]
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("query")
        assert len(results) == 2
        assert results[0].content == "chunk B"  # highest reranker score (0.9 at index 1)
        assert results[1].content == "chunk A"

    @pytest.mark.asyncio
    async def test_max_per_source_backfills_when_pool_thin(self, config, embed_service, vectorstore):
        """Verify reranking caps gracefully relax mapping remaining items when alternative file pools run dry."""
        # Pool: 5 chunks from a.md, 1 from b.md. top_k_rerank=5, cap=2.
        # Pass 1: 2× a.md + 1× b.md = 3 selected, 3 a.md chunks parked.
        # Pass 2: backfill from overflow → 5 total. The cap must NOT reduce
        # the result count below top_k_rerank when alternatives are exhausted.
        config.reranking = RerankingConfig(
            enabled=True, top_k_rerank=5, max_per_source=2,
        )
        vectorstore.query.return_value = [
            {"id": "1", "content": "A1", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h1"}, "score": 0.95},
            {"id": "2", "content": "A2", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h2"}, "score": 0.9},
            {"id": "3", "content": "A3", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h3"}, "score": 0.85},
            {"id": "4", "content": "A4", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h4"}, "score": 0.8},
            {"id": "5", "content": "A5", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h5"}, "score": 0.75},
            {"id": "6", "content": "B1", "metadata": {"source_file": "b.md", "document_type": "markdown", "heading_hierarchy": "x"}, "score": 0.7},
        ]
        retriever = Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("query")
        # Must reach the requested count, not silently drop slots
        assert len(results) == 5
        sources = [r.source_file for r in results]
        assert sources.count("a.md") == 4   # 2 from cap + 2 backfilled
        assert sources.count("b.md") == 1
        # Re-sorted by score: highest first
        scores_descending = [r.score for r in results]
        assert scores_descending == sorted(scores_descending, reverse=True)

    @pytest.mark.asyncio
    async def test_max_per_source_caps_single_file_when_diversity_available(
        self, config, embed_service, vectorstore,
    ):
        """Verify distinct file enforcement rigidly rejects clustered hits provided diversity offsets exist in the pool."""
        # Pool: 3× a.md, 2× b.md, 1× c.md. top_k=5, cap=2.
        # With enough diversity, the cap stays binding (no backfill needed).
        config.reranking = RerankingConfig(
            enabled=True, top_k_rerank=5, max_per_source=2,
        )
        vectorstore.query.return_value = [
            {"id": "1", "content": "A1", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h1"}, "score": 0.95},
            {"id": "2", "content": "A2", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h2"}, "score": 0.9},
            {"id": "3", "content": "A3", "metadata": {"source_file": "a.md", "document_type": "markdown", "heading_hierarchy": "h3"}, "score": 0.85},
            {"id": "4", "content": "B1", "metadata": {"source_file": "b.md", "document_type": "markdown", "heading_hierarchy": "x"}, "score": 0.8},
            {"id": "5", "content": "B2", "metadata": {"source_file": "b.md", "document_type": "markdown", "heading_hierarchy": "y"}, "score": 0.75},
            {"id": "6", "content": "C1", "metadata": {"source_file": "c.md", "document_type": "markdown", "heading_hierarchy": "z"}, "score": 0.7},
        ]
        retriever = Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("query")
        sources = [r.source_file for r in results]
        # Cap binding: A3 must be skipped because b.md and c.md provide diversity.
        assert sources.count("a.md") == 2
        assert "A3" not in [r.content for r in results]
        assert "b.md" in sources
        assert "c.md" in sources

    @pytest.mark.asyncio
    async def test_max_per_source_counts_equally_named_files_per_source(
        self, config, embed_service, vectorstore,
    ):
        """Two sources with the same file name (e.g. llm_report.md) each get their own cap."""
        config.reranking = RerankingConfig(
            enabled=True, top_k_rerank=4, max_per_source=2,
        )

        def _chunk(i, sid):
            return {"id": str(i), "content": f"{sid}{i}", "score": 0.9,
                    "metadata": {"source_file": "llm_report.md", "source_id": sid,
                                 "document_type": "markdown", "heading_hierarchy": f"h{i}"}}

        vectorstore.query.return_value = [_chunk(1, "a"), _chunk(2, "a"), _chunk(3, "a"),
                                          _chunk(4, "b"), _chunk(5, "b"), _chunk(6, "c.md")]
        vectorstore.query.return_value[-1]["metadata"]["source_file"] = "c.md"
        retriever = Retriever(config=config, embedding_service=embed_service, vectorstore=vectorstore)
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("query")
        assert [r.content for r in results] == ["a1", "a2", "b4", "b5"]


def _slot_hit(id_, group, file, score=0.8):
    return {"id": id_, "content": f"text {id_}", "score": score,
            "metadata": {"source_file": file, "document_type": "markdown", "heading_hierarchy": id_, "group": group}}


class TestRulesSlot:
    """3.10: an entity file fills the candidates — one rulebook chunk gets in via a separate rules search."""

    MAIN = [_slot_hit(f"n{i}", "lore", "npc.md", 0.9 - i / 100) for i in range(4)]
    RULES = [_slot_hit("r1", "rules", "regeln.pdf"), _slot_hit("r2", "rules", "regeln.pdf")]

    def _retriever(self, embed_service, vectorstore, scores, **slot):
        cfg = RetrievalConfig(top_k=4, score_threshold=0.5,
                              reranking=RerankingConfig(top_k_rerank=3, max_per_source=0),
                              rules_slot=RulesSlotConfig(**{"enabled": True, "candidates": 2, "min_score": -4, **slot}))
        vectorstore.query.side_effect = lambda query_embedding, top_k, where: (
            self.RULES if "group" in str(where) else self.MAIN)
        ret = Retriever(config=cfg, embedding_service=embed_service, vectorstore=vectorstore)
        ret._reranker = MagicMock()
        ret._reranker.predict.side_effect = lambda pairs: scores[:len(pairs)]
        return ret

    @pytest.mark.asyncio
    async def test_best_rules_chunk_replaces_the_weakest(self, embed_service, vectorstore):
        # main n0..n3, then extra r1, r2
        ret = self._retriever(embed_service, vectorstore, [2.0, 1.0, 0.5, 0.1, -3.0, -1.0])
        results = await ret.retrieve("Was bewirkt Ritterlichkeit bei Escanor?")
        assert [r.heading for r in results] == ["n0", "n1", "r2"]  # r2 (-1.0) beats r1, replaces n2

    @pytest.mark.asyncio
    async def test_below_min_score_no_slot(self, embed_service, vectorstore):
        ret = self._retriever(embed_service, vectorstore, [2.0, 1.0, 0.5, 0.1, -6.0, -5.0])
        results = await ret.retrieve("Wer ist Arkenfeld?")
        assert [r.heading for r in results] == ["n0", "n1", "n2"]

    @pytest.mark.asyncio
    async def test_no_slot_when_a_rules_chunk_already_made_it(self, embed_service, vectorstore):
        self.MAIN = self.MAIN[:3] + [_slot_hit("rx", "rules", "regeln.pdf", 0.85)]
        ret = self._retriever(embed_service, vectorstore, [2.0, 1.0, 0.5, 3.0, -1.0, -1.0])
        results = await ret.retrieve("q")
        assert sum(r.metadata["group"] == "rules" for r in results) == 1
        assert results[0].heading == "rx"

    @pytest.mark.asyncio
    async def test_rules_search_keeps_the_request_filter(self, embed_service, vectorstore):
        """Player mode: the extra search is AND-ed with the caller's filter — it can't reach other sources."""
        ret = self._retriever(embed_service, vectorstore, [2.0, 1.0, 0.5, 0.1, -1.0, -1.0])
        await ret.retrieve("q", metadata_filters={"source_id": {"$in": ["regelbuch"]}})
        rules_where = [c.kwargs["where"] for c in vectorstore.query.call_args_list if "group" in str(c.kwargs["where"])]
        assert len(rules_where) == 1
        assert {"group": "rules"} in rules_where[0]["$and"]
        assert "{'source_id': {'$in': ['regelbuch']}}" in str(rules_where[0])

    @pytest.mark.asyncio
    async def test_no_slot_with_fewer_than_three_places(self, embed_service, vectorstore):
        """With top_k_rerank=1 a weak rules chunk would replace the only (strong) hit — no slot then."""
        ret = self._retriever(embed_service, vectorstore, [9.0, 8.0, 0.5, 0.1, -3.9, -3.9])
        results = await ret.retrieve("q", top_k_rerank=1)
        assert [r.heading for r in results] == ["n0"]
        assert vectorstore.query.call_count == 1   # the slot can't be used — no rules search either

    @pytest.mark.asyncio
    async def test_disabled_makes_no_extra_search(self, embed_service, vectorstore):
        ret = self._retriever(embed_service, vectorstore, [2.0, 1.0, 0.5, 0.1], enabled=False)
        results = await ret.retrieve("q")
        assert vectorstore.query.call_count == 1
        assert [r.heading for r in results] == ["n0", "n1", "n2"]


class TestRerankerBackend:
    """reranking.backend picks the GPU model; any DirectML failure falls back to torch with a warning."""

    @staticmethod
    def _retriever(backend):
        cfg = RetrievalConfig(reranking=RerankingConfig(backend=backend))
        return Retriever(config=cfg, embedding_service=AsyncMock(), vectorstore=MagicMock())

    def test_directml_backend_uses_gpu_model(self, monkeypatch):
        import src.retrieval.onnx_reranker as onnx_reranker
        gpu = MagicMock()
        monkeypatch.setattr(onnx_reranker, "DirectMLCrossEncoder", lambda model, max_length: gpu)
        torch_loader = MagicMock()
        monkeypatch.setattr("sentence_transformers.CrossEncoder", torch_loader)

        assert self._retriever("directml").get_reranker() is gpu
        torch_loader.assert_not_called()

    def test_directml_failure_falls_back_to_torch(self, monkeypatch, caplog):
        import src.retrieval.onnx_reranker as onnx_reranker

        def _unavailable(model, max_length):
            raise onnx_reranker.DirectMLUnavailable("DmlExecutionProvider missing")

        monkeypatch.setattr(onnx_reranker, "DirectMLCrossEncoder", _unavailable)
        cpu = MagicMock()
        monkeypatch.setattr("sentence_transformers.CrossEncoder", lambda model, max_length: cpu)

        with caplog.at_level("WARNING"):
            assert self._retriever("directml").get_reranker() is cpu
        assert "falling back to the CPU" in caplog.text

    @pytest.mark.asyncio
    async def test_gpu_failure_at_runtime_switches_to_the_cpu(self, monkeypatch, caplog):
        """A DirectML error during a query (driver reset, out of VRAM) must not fail every query until a
        restart: warn, load the CPU model, score again — and stay on the CPU."""
        import src.retrieval.onnx_reranker as onnx_reranker
        gpu = MagicMock()
        gpu.predict.side_effect = RuntimeError("DXGI_ERROR_DEVICE_REMOVED")
        monkeypatch.setattr(onnx_reranker, "DirectMLCrossEncoder", lambda model, max_length: gpu)
        cpu = MagicMock()
        cpu.predict.side_effect = lambda pairs: [float(len(pairs) - i) for i in range(len(pairs))]
        monkeypatch.setattr("sentence_transformers.CrossEncoder", lambda model, max_length: cpu)
        vs = MagicMock()
        vs.query.return_value = [
            {"id": str(i), "content": f"c{i}", "score": 0.9,
             "metadata": {"source_file": f"{i}.md", "document_type": "markdown", "heading_hierarchy": str(i)}}
            for i in range(3)
        ]
        ret = Retriever(config=RetrievalConfig(reranking=RerankingConfig(backend="directml")),
                        embedding_service=AsyncMock(), vectorstore=vs)

        with caplog.at_level("WARNING"):
            first = await ret.retrieve("q")
            second = await ret.retrieve("q")

        assert [r.heading for r in first] == ["0", "1", "2"] == [r.heading for r in second]
        assert "switching to the CPU" in caplog.text
        assert gpu.predict.call_count == 1 and cpu.predict.call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_gpu_failures_load_the_cpu_model_once(self, monkeypatch):
        """Two queries fail on the GPU at the same time: both still get an answer, the CPU model
        is loaded once, and the query that noticed second reuses it instead of re-raising."""
        import asyncio
        import threading

        import src.retrieval.onnx_reranker as onnx_reranker
        both_inside = threading.Barrier(2, timeout=10)

        def _gpu_predict(pairs):
            both_inside.wait()            # both queries are running on the GPU before it fails
            raise RuntimeError("DXGI_ERROR_DEVICE_REMOVED")

        gpu = MagicMock()
        gpu.predict.side_effect = _gpu_predict
        monkeypatch.setattr(onnx_reranker, "DirectMLCrossEncoder", lambda model, max_length: gpu)
        cpu_loads = []

        def _load_cpu(model, max_length):
            cpu_loads.append(model)
            cpu = MagicMock()
            cpu.predict.side_effect = lambda pairs: [float(len(pairs) - i) for i in range(len(pairs))]
            return cpu

        monkeypatch.setattr("sentence_transformers.CrossEncoder", _load_cpu)
        vs = MagicMock()
        vs.query.return_value = [
            {"id": str(i), "content": f"c{i}", "score": 0.9,
             "metadata": {"source_file": f"{i}.md", "document_type": "markdown", "heading_hierarchy": str(i)}}
            for i in range(3)
        ]
        ret = Retriever(config=RetrievalConfig(reranking=RerankingConfig(backend="directml")),
                        embedding_service=AsyncMock(), vectorstore=vs)
        ret.get_reranker()   # preloaded at startup, like src/main.py does

        first, second = await asyncio.gather(ret.retrieve("q1"), ret.retrieve("q2"))

        assert [r.heading for r in first] == ["0", "1", "2"] == [r.heading for r in second]
        assert len(cpu_loads) == 1

    @pytest.mark.asyncio
    async def test_cpu_reranker_errors_are_not_swallowed(self, embed_service, vectorstore):
        """Only the GPU gets a fallback — a failing CPU model is a real error."""
        ret = Retriever(config=RetrievalConfig(reranking=RerankingConfig(backend="torch")),
                        embedding_service=embed_service, vectorstore=vectorstore)
        ret._reranker = MagicMock()
        ret._reranker.predict.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            await ret.retrieve("where is arkenfeld")

    def test_torch_backend_never_tries_directml(self, monkeypatch):
        import src.retrieval.onnx_reranker as onnx_reranker
        gpu_loader = MagicMock()
        monkeypatch.setattr(onnx_reranker, "DirectMLCrossEncoder", gpu_loader)
        cpu = MagicMock()
        monkeypatch.setattr("sentence_transformers.CrossEncoder", lambda model, max_length: cpu)

        assert self._retriever("torch").get_reranker() is cpu
        gpu_loader.assert_not_called()


class TestRRFMerge:
    """Test suite validating the Reciprocal Rank Fusion primitive."""

    def test_empty_inputs(self):
        """Verify fusion of two empty lists yields an empty list."""
        assert _rrf_merge([], [], bm25_weight=0.3) == []

    def test_vector_only(self):
        """Verify a vector-only input preserves rank order and returns all items."""
        vector = [
            {"id": "a", "content": "A", "metadata": {}, "score": 0.9},
            {"id": "b", "content": "B", "metadata": {}, "score": 0.8},
        ]
        merged = _rrf_merge(vector, [], bm25_weight=0.3)
        assert [m["id"] for m in merged] == ["a", "b"]
        assert merged[0]["score"] > merged[1]["score"]

    def test_overlap_boosts_shared_item(self):
        """Verify items appearing in both lists rank above singletons of equal standalone rank."""
        vector = [
            {"id": "a", "content": "A", "metadata": {}, "score": 0.9},
            {"id": "b", "content": "B", "metadata": {}, "score": 0.8},
        ]
        bm25 = [
            {"id": "b", "content": "B", "metadata": {}, "score": 5.0},
            {"id": "c", "content": "C", "metadata": {}, "score": 3.0},
        ]
        merged = _rrf_merge(vector, bm25, bm25_weight=0.5)
        ids = [m["id"] for m in merged]
        # b appears in both lists → fused score > singleton a or c at same rank.
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_weight_shifts_balance(self):
        """Verify bm25_weight close to 1 lets the BM25 list dominate ordering."""
        vector = [{"id": "a", "content": "A", "metadata": {}, "score": 0.9}]
        bm25 = [{"id": "b", "content": "B", "metadata": {}, "score": 5.0}]
        merged_bm_heavy = _rrf_merge(vector, bm25, bm25_weight=0.9)
        assert merged_bm_heavy[0]["id"] == "b"
        merged_vec_heavy = _rrf_merge(vector, bm25, bm25_weight=0.1)
        assert merged_vec_heavy[0]["id"] == "a"


class TestHybridRetrieve:
    """Test suite verifying BM25 fusion wiring, thresholding and toggle semantics in retrieve()."""

    @pytest.fixture
    def hybrid_config(self):
        """Retrieval config with hybrid enabled by default and reranking off."""
        return RetrievalConfig(
            top_k=5,
            score_threshold=0.5,
            reranking=RerankingConfig(enabled=False),
            hybrid=HybridSearchConfig(enabled=True, bm25_weight=0.3, bm25_top_k=5),
        )

    def _make_bm25(self, docs):
        """Build an in-memory BM25 index from inline dicts, skipping the vectorstore."""
        store = MagicMock()
        store.get_all.return_value = {"ids": [d["id"] for d in docs], "documents": [d["content"] for d in docs],
                                      "metadatas": [d["metadata"] for d in docs]}
        idx = BM25Index()
        idx.ensure_built(store)
        return idx

    @pytest.mark.asyncio
    async def test_hybrid_merges_vector_and_bm25(self, hybrid_config, embed_service):
        """Verify BM25-only candidates reach the output when hybrid is active."""
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "semantic hit about the city",
             "metadata": {"source_file": "v.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
        ]
        bm25 = self._make_bm25([
            {"id": "v1", "content": "semantic hit about the city",
             "metadata": {"source_file": "v.md", "document_type": "markdown"}},
            {"id": "b1", "content": "Langschwert 15 Gold exact match",
             "metadata": {"source_file": "weapons.md", "document_type": "markdown"}},
        ])
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        results = await retriever.retrieve("Langschwert Gold")
        sources = [r.source_file for r in results]
        assert "v.md" in sources
        assert "weapons.md" in sources  # BM25-only hit survived fusion

    @pytest.mark.asyncio
    async def test_hybrid_disabled_skips_bm25(self, hybrid_config, embed_service):
        """Verify hybrid=False bypasses BM25 completely, never calling query()."""
        hybrid_config.hybrid.enabled = False
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "hit",
             "metadata": {"source_file": "v.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
        ]
        bm25 = MagicMock(spec=BM25Index)
        bm25.is_built = True
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        await retriever.retrieve("anything")
        bm25.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_flag_override_forces_bm25(self, hybrid_config, embed_service):
        """Verify per-request hybrid=True overrides a disabled config default."""
        hybrid_config.hybrid.enabled = False
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "hit",
             "metadata": {"source_file": "v.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
        ]
        bm25 = self._make_bm25([
            {"id": "b1", "content": "exact keyword match",
             "metadata": {"source_file": "kw.md", "document_type": "markdown"}},
        ])
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        results = await retriever.retrieve("exact keyword", hybrid=True)
        sources = [r.source_file for r in results]
        assert "kw.md" in sources

    @pytest.mark.asyncio
    async def test_hybrid_does_not_nullify_threshold(self, hybrid_config, embed_service):
        """Regression: score_threshold=0.5 must not wipe hybrid results despite tiny RRF scores."""
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "Arkenfeld details",
             "metadata": {"source_file": "Arkenfeld.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
            {"id": "v2", "content": "low score noise",
             "metadata": {"source_file": "noise.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.2},
        ]
        bm25 = self._make_bm25([
            {"id": "b1", "content": "Arkenfeld keyword",
             "metadata": {"source_file": "kw.md", "document_type": "markdown"}},
        ])
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        results = await retriever.retrieve("Arkenfeld")
        # High-score vector hit + BM25 hit must survive; low-score vector noise drops.
        sources = [r.source_file for r in results]
        assert "Arkenfeld.md" in sources
        assert "noise.md" not in sources
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_hybrid_lazy_builds_index(self, hybrid_config, embed_service):
        """Verify the BM25 index is built on first hybrid call if not already built."""
        vs = MagicMock()
        vs.query.return_value = []
        # Minimal fake corpus for build_from_vectorstore.
        vs.get_all.return_value = {
            "ids": ["b1"],
            "documents": ["keyword content"],
            "metadatas": [{"source_file": "kw.md", "document_type": "markdown"}],
        }

        bm25 = BM25Index()
        assert bm25.is_built is False
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)

        await retriever.retrieve("anything")
        assert bm25.is_built is True

    @pytest.mark.asyncio
    async def test_hybrid_results_reach_reranker(self, hybrid_config, embed_service):
        """Verify BM25-only candidates are passed into the cross-encoder reranker."""
        hybrid_config.reranking = RerankingConfig(enabled=True, top_k_rerank=3)
        vs = MagicMock()
        vs.query.return_value = [
            {"id": "v1", "content": "vector hit",
             "metadata": {"source_file": "v.md", "document_type": "markdown", "heading_hierarchy": ""}, "score": 0.9},
        ]
        bm25 = self._make_bm25([
            {"id": "b1", "content": "BM25-only exact keyword",
             "metadata": {"source_file": "kw.md", "document_type": "markdown"}},
        ])
        retriever = Retriever(config=hybrid_config, embedding_service=embed_service,
                              vectorstore=vs, bm25_index=bm25)
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.6, 0.8]  # rerank passes both
        retriever._reranker = mock_reranker

        results = await retriever.retrieve("exact keyword")
        # The reranker was called with both vector and BM25 candidates.
        pairs = mock_reranker.predict.call_args[0][0]
        contents = [p[1] for p in pairs]
        assert "vector hit" in contents
        assert "BM25-only exact keyword" in contents
        assert len(results) == 2


def _hit(heading, idx, score=0.9, source_file="regelbuch.pdf"):
    """Vectorstore-style hit for section expansion tests."""
    return {"id": f"{source_file}-{idx}", "content": f"text {idx}", "score": score,
            "metadata": {"source_file": source_file, "source_id": "rules", "document_type": "pdf",
                         "heading_hierarchy": heading, "chunk_index": idx}}


class TestSectionExpansion:
    """Test suite for replacing sibling fragments with their complete parent section."""

    _SECTION = [
        _hit("Waldläufer", 10), _hit("Waldläufer > Kurzprofil", 11), _hit("Waldläufer > Fesselschuss", 12),
        _hit("Waldläufer > Tiergefährten > Schwarzbär", 13), _hit("Waldläufer > Mehrfachschuss", 14),
        _hit("Krieger > Kurzprofil", 20),
    ]

    def _retriever(self, embed_service, hits, max_chunks=25):
        config = RetrievalConfig(top_k=10, score_threshold=0.5, reranking=RerankingConfig(enabled=False))
        config.section_expansion.max_chunks = max_chunks
        vs = MagicMock()
        vs.query.return_value = hits
        vs.get_chunks_for_file.return_value = list(reversed(self._SECTION))  # store order is arbitrary
        return Retriever(config=config, embedding_service=embed_service, vectorstore=vs), vs

    @pytest.mark.asyncio
    async def test_sibling_hits_expand_to_whole_section_in_document_order(self, embed_service, caplog):
        """Verify two sibling hits pull in every chunk under their parent, in chunk_index order, once —
        logged as one INFO summary (the per-section detail is DEBUG)."""
        hits = [_hit("Waldläufer > Mehrfachschuss", 14, 0.9), _hit("Waldläufer > Kurzprofil", 11, 0.8),
                _hit("Arkenfeld", 0, 0.7, source_file="Arkenfeld.md")]
        retriever, vs = self._retriever(embed_service, hits)

        with caplog.at_level("INFO", logger="src.retrieval.retriever"):
            results = await retriever.retrieve("fähigkeiten waldläufer")

        infos = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
        assert infos.count("Section expansion: 'Waldläufer' (regelbuch.pdf, 5 chunks)") == 1
        assert not any(m.startswith(("Expanded section", "Retrieved")) for m in infos)

        assert [r.metadata["chunk_index"] for r in results[:5]] == [10, 11, 12, 13, 14]
        assert results[5].source_file == "Arkenfeld.md"           # unrelated hit keeps its place after
        assert len(results) == 6                                  # no duplicates of the original hits
        assert all(r.score == 0.9 for r in results[:5])           # section inherits its best hit score
        vs.get_chunks_for_file.assert_called_once_with("rules", "regelbuch.pdf", {"document_type": {"$ne": "image"}})

    @pytest.mark.asyncio
    async def test_repeated_same_heading_does_not_expand(self, embed_service):
        """Verify several parts of one heading (no distinct siblings) don't trigger expansion."""
        hits = [_hit("NPC > Rolle > Waldläufer (Kern)", 1, source_file="npc.md"),
                _hit("NPC > Rolle > Waldläufer (Kern)", 2, source_file="npc.md")]
        retriever, vs = self._retriever(embed_service, hits)

        results = await retriever.retrieve("waldläufer")

        assert len(results) == 2
        vs.get_chunks_for_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_oversized_section_is_not_expanded(self, embed_service):
        """Verify a section above max_chunks is left as the original hits."""
        hits = [_hit("Waldläufer > Mehrfachschuss", 14), _hit("Waldläufer > Kurzprofil", 11)]
        retriever, _ = self._retriever(embed_service, hits, max_chunks=3)

        results = await retriever.retrieve("waldläufer")

        assert [r.metadata["chunk_index"] for r in results] == [14, 11]

    @pytest.mark.asyncio
    async def test_low_ranked_section_is_not_expanded(self, embed_service):
        """Verify two sibling hits that only appear below top_rank don't pull in their section."""
        hits = [_hit("Überschaden", 0, 0.9, source_file="other.pdf"),
                _hit("Überheilung", 1, 0.85, source_file="other.pdf"),
                _hit("Schadenstypen", 2, 0.8, source_file="other.pdf"),
                _hit("Waldläufer > Mehrfachschuss", 14, 0.7), _hit("Waldläufer > Kurzprofil", 11, 0.6)]
        retriever, vs = self._retriever(embed_service, hits)

        results = await retriever.retrieve("was ist überschaden")

        assert len(results) == 5
        vs.get_chunks_for_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_top_rank_zero_expands_regardless_of_rank(self, embed_service):
        """Verify top_rank=0 restores the unrestricted behaviour."""
        hits = [_hit("A", 0, 0.9, source_file="other.pdf"), _hit("B", 1, 0.85, source_file="other.pdf"),
                _hit("C", 2, 0.8, source_file="other.pdf"),
                _hit("Waldläufer > Mehrfachschuss", 14, 0.7), _hit("Waldläufer > Kurzprofil", 11, 0.6)]
        retriever, vs = self._retriever(embed_service, hits)
        retriever.config.section_expansion.top_rank = 0

        results = await retriever.retrieve("waldläufer")

        assert len(results) == 8
        vs.get_chunks_for_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_same_file_name_in_two_sources_is_not_deduplicated(self, embed_service):
        """Chunks are identified by (source_id, file, index) — equal names in two sources are different chunks."""
        other = {"id": "o", "content": "other source", "score": 0.5,
                 "metadata": {"source_file": "regelbuch.pdf", "source_id": "other", "document_type": "pdf",
                              "heading_hierarchy": "Anhang", "chunk_index": 10}}
        hits = [_hit("Waldläufer > Mehrfachschuss", 14, 0.9), _hit("Waldläufer > Kurzprofil", 11, 0.8), other]
        retriever, _ = self._retriever(embed_service, hits)

        results = await retriever.retrieve("waldläufer")

        assert ("other", 10) in {(r.metadata["source_id"], r.metadata["chunk_index"]) for r in results}

    _KRIEGER = [_hit("Krieger > Kurzprofil", 0, source_file="klassen.pdf"),
                _hit("Krieger > Schildwall", 1, source_file="klassen.pdf"),
                _hit("Krieger > Sturmangriff", 2, source_file="klassen.pdf")]
    # Waldläufer section (5 chunks, best hit rank 0) and Krieger section (3 chunks, best hit rank 2)
    _TWO_SECTIONS = [_hit("Waldläufer > Mehrfachschuss", 14, 0.9), _hit("Waldläufer > Kurzprofil", 11, 0.85),
                     _hit("Krieger > Schildwall", 1, 0.8, source_file="klassen.pdf"),
                     _hit("Krieger > Sturmangriff", 2, 0.75, source_file="klassen.pdf")]

    def _two_section_retriever(self, embed_service, budget):
        retriever, vs = self._retriever(embed_service, self._TWO_SECTIONS)
        retriever.config.section_expansion.max_total_chunks = budget
        vs.get_chunks_for_file.side_effect = lambda sid, f, where: list(
            self._KRIEGER if f == "klassen.pdf" else self._SECTION)
        return retriever

    @staticmethod
    def _idents(results):
        return [(r.source_file, r.metadata["chunk_index"]) for r in results]

    @pytest.mark.asyncio
    async def test_budget_that_fits_expands_every_section(self, embed_service):
        """4 hits + 3 new Waldläufer chunks + 1 new Krieger chunk = 8 ≤ budget."""
        retriever = self._two_section_retriever(embed_service, budget=8)

        results = await retriever.retrieve("klassen")

        assert len(results) == 8

    @pytest.mark.asyncio
    async def test_budget_drops_the_lower_ranked_section(self, embed_service):
        """Budget 7: the Waldläufer section (best hit first) fits, the Krieger section keeps its two hits."""
        retriever = self._two_section_retriever(embed_service, budget=7)

        results = await retriever.retrieve("klassen")

        assert self._idents(results) == [("regelbuch.pdf", i) for i in (10, 11, 12, 13, 14)] + [
            ("klassen.pdf", 1), ("klassen.pdf", 2)]

    @pytest.mark.asyncio
    async def test_budget_skips_a_section_that_does_not_fit_but_expands_a_smaller_one(self, embed_service):
        """Budget 5: the Waldläufer section (+3) doesn't fit, the Krieger section (+1) still does."""
        retriever = self._two_section_retriever(embed_service, budget=5)

        results = await retriever.retrieve("klassen")

        assert self._idents(results) == [("regelbuch.pdf", 14), ("regelbuch.pdf", 11),
                                         ("klassen.pdf", 0), ("klassen.pdf", 1), ("klassen.pdf", 2)]

    @pytest.mark.asyncio
    async def test_budget_counts_chunks_of_nested_sections_once(self, embed_service):
        """Sections 'X > Y' and 'X' overlap: after 'X > Y' (+1 chunk) the outer 'X' costs only the one
        chunk not loaded yet — 4 hits + 1 + 1 = 6 fits; counting the overlap twice would drop 'X'."""
        nested = [_hit("X", 0, source_file="n.pdf"), _hit("X > c", 1, source_file="n.pdf"),
                  _hit("X > d", 2, source_file="n.pdf"), _hit("X > Y > a", 3, source_file="n.pdf"),
                  _hit("X > Y > b", 4, source_file="n.pdf"), _hit("X > Y > e", 5, source_file="n.pdf")]
        hits = [nested[3], nested[4], nested[1], nested[2]]
        retriever, vs = self._retriever(embed_service, hits)
        retriever.config.section_expansion.max_total_chunks = 6
        vs.get_chunks_for_file.return_value = nested

        results = await retriever.retrieve("x")

        assert sorted(r.metadata["chunk_index"] for r in results) == [0, 1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_budget_never_drops_reranked_hits(self, embed_service):
        """A budget below the rerank result only blocks expansion — the hits themselves stay."""
        retriever = self._two_section_retriever(embed_service, budget=2)

        results = await retriever.retrieve("klassen")

        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_disabled_expansion_keeps_hits(self, embed_service):
        """Verify section_expansion.enabled=False leaves retrieval untouched."""
        hits = [_hit("Waldläufer > Mehrfachschuss", 14), _hit("Waldläufer > Kurzprofil", 11)]
        retriever, vs = self._retriever(embed_service, hits)
        retriever.config.section_expansion.enabled = False

        results = await retriever.retrieve("waldläufer")

        assert len(results) == 2
        vs.get_chunks_for_file.assert_not_called()


class TestRerankerDevice:
    """Where the reranker runs — shown in the chat sidebar; cpu_fallback is the silent slow path."""

    @pytest.mark.parametrize("enabled, backend, loaded, on_gpu, expected", [
        (False, "directml", True, True, "off"),
        (True, "directml", False, False, "not_loaded"),
        (True, "directml", True, True, "gpu"),
        (True, "directml", True, False, "cpu_fallback"),
        (True, "torch", True, False, "cpu"),
    ])
    def test_device(self, embed_service, enabled, backend, loaded, on_gpu, expected):
        from src.config.manager import RerankingConfig, RetrievalConfig
        config = RetrievalConfig(reranking=RerankingConfig(enabled=enabled, backend=backend))
        retriever = Retriever(config=config, embedding_service=embed_service, vectorstore=MagicMock())
        retriever._reranker = object() if loaded else None
        retriever._reranker_on_gpu = on_gpu
        assert retriever.reranker_device == expected
