from unittest.mock import MagicMock

import pytest

from src.config.manager import VectorStoreConfig
from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vectorstore import VectorStoreService


@pytest.fixture
def vs_config():
    """Provides a baseline VectorStoreConfig targeting transient embedded Chroma usage."""
    return VectorStoreConfig(
        mode="embedded",
        persist_directory="./test_chroma",
        collection_name="test",
        distance_metric="cosine",
    )


@pytest.fixture
def embed_service():
    """Provides a functionally mocked EmbeddingService emitting reliable dimensional arrays."""
    svc = MagicMock(spec=EmbeddingService)
    svc.embed_texts_sync.return_value = [[0.1, 0.2, 0.3]]
    return svc


@pytest.fixture
def mock_collection():
    """Provides a Mock representing the active ChromaDB collection table bindings."""
    col = MagicMock()
    col.count.return_value = 0
    return col


@pytest.fixture
def service(vs_config, embed_service, mock_collection):
    """Provides a completely integrated test VectorStoreService with a mocked underlying database layer."""
    svc = VectorStoreService(vs_config, embed_service)
    svc._collection = mock_collection
    mock_client = MagicMock()
    mock_client.heartbeat.return_value = True
    svc._client = mock_client
    return svc


@pytest.fixture
def sample_chunk():
    """Provides a standardized fully hydrated Chunk payload for database insertion simulations."""
    return Chunk(
        content="Arkenfeld is a city.",
        source_file="Orte/Arkenfeld.md",
        source_path="/vault/Orte/Arkenfeld.md",
        document_type="markdown",
        heading_hierarchy=["Orte", "Arkenfeld"],
        chunk_index=0,
        total_chunks=1,
        metadata={"aliases": ["Arken"], "content_hash": "abc123"},
    )


class TestBuildEmbedText:
    """Test suite validating identity framing appended to dense text blocks prior to embedding calculations."""

    def test_includes_filename_stem(self, service, sample_chunk):
        """Verify the document filename acts as a distinct semantic beacon mapping contextual identities."""
        text = service._build_embed_text(sample_chunk)
        assert "Arkenfeld" in text

    def test_includes_aliases(self, service, sample_chunk):
        """Verify metadata aliases inject seamlessly directly into identity header blocks."""
        text = service._build_embed_text(sample_chunk)
        assert "Arken" in text

    def test_includes_content(self, service, sample_chunk):
        """Verify pure contextual contents are preserved below the identity blocks unconditionally."""
        text = service._build_embed_text(sample_chunk)
        assert "Arkenfeld is a city." in text

    def test_alias_list(self, service):
        """Verify multiple unstructured aliases unroll neatly forming dense keyword clusters."""
        chunk = Chunk(
            content="text",
            source_file="file.md",
            source_path="/file.md",
            document_type="markdown",
            metadata={"aliases": ["Alias1", "Alias2"]},
        )
        text = service._build_embed_text(chunk)
        assert "Alias1" in text
        assert "Alias2" in text

    def test_no_duplicate_stem_in_aliases(self, service):
        """Verify the primary document identity deduplicates proactively avoiding dimensional weighting imbalances."""
        chunk = Chunk(
            content="text",
            source_file="Arkenfeld.md",
            source_path="/Arkenfeld.md",
            document_type="markdown",
            metadata={"aliases": ["Arkenfeld", "Arken"]},
        )
        text = service._build_embed_text(chunk)
        # stem should not appear twice in identity line
        identity_line = text.split("\n\n")[0]
        assert identity_line.count("Arkenfeld") == 1

    _WEAPONS = ("**7.1.1 Waffen** > **Fernkampfwaffen**\n\n"
                "|**Waffe**|**Kat.**|**normal**|**Besonderheiten**|\n"
                "|---|---|---|---|\n"
                "|**Kurzbogen**|leicht|GEW + 1|**15 m**|\n"
                "|**l**<br>**Langbogen**|normal|GEW + 2|**60 m**|\n"
                "|**Armbrust**|schwer|GEW + 5|**45 m**; nur ein Geschoß, Nach-|\n"
                "||||laden als**Bonusaktion**|")

    def _embed_text(self, service, content):
        return service._build_embed_text(Chunk(content=content, source_file="pnp_regelbuch.pdf",
                                               source_path="/r.pdf", document_type="pdf", metadata={}))

    def test_table_gets_its_row_names_after_the_identity(self, service):
        """One vector averages all rows — the names make a single row findable ("Armbrust" ranked 47th)."""
        parts = self._embed_text(service, self._WEAPONS).split("\n\n")

        assert parts[0] == "pnp_regelbuch"
        assert parts[1] == "Einträge: Kurzbogen, l Langbogen, Armbrust"
        assert self._WEAPONS in "\n\n".join(parts[2:])     # the table itself stays unchanged

    def test_header_separator_and_continuation_rows_are_not_names(self, service):
        names = self._embed_text(service, self._WEAPONS).split("\n\n")[1]

        assert "Waffe" not in names and "---" not in names and "laden" not in names

    def test_two_tables_each_skip_their_header(self, service):
        content = "|A|B|\n|---|---|\n|x|1|\n\nText\n\n|C|D|\n|:--|--:|\n|y|2|"
        assert self._embed_text(service, content).split("\n\n")[1] == "Einträge: x, y"

    def test_text_without_table_has_no_names_line(self, service, sample_chunk):
        assert "Einträge:" not in service._build_embed_text(sample_chunk)


class TestChunkId:
    """Test suite verifying hash stability for reproducible partial sequence mapping."""

    def test_deterministic(self, service, sample_chunk):
        """Verify consistent id calculations yield mathematically equal identity hashes."""
        id1 = service._chunk_id(sample_chunk)
        id2 = service._chunk_id(sample_chunk)
        assert id1 == id2

    def test_different_for_different_source(self, service, sample_chunk):
        """Verify hash permutations definitively differ across distinct pathing bounds."""
        other = Chunk(
            content="other",
            source_file="other.md",
            source_path="/other.md",
            document_type="markdown",
            chunk_index=0,
        )
        assert service._chunk_id(sample_chunk) != service._chunk_id(other)

    def test_different_for_same_path_in_different_sources(self, service):
        """Verify two sources sharing a relative path (e.g. Index.md) get distinct ids."""
        a = Chunk(content="a", source_file="Index.md", source_path="/a/Index.md",
                  document_type="markdown", metadata={"source_id": "vault"})
        b = Chunk(content="b", source_file="Index.md", source_path="/b/Index.md",
                  document_type="markdown", metadata={"source_id": "rules"})
        assert service._chunk_id(a) != service._chunk_id(b)


class TestGetCollection:
    """Test suite for collection creation and distance-metric handling."""

    """Against a real in-memory ChromaDB: the API's quirks (l2 trap, ignored config on reopen) are the point."""

    def _service(self, vs_config, embed_service, name, ef=1000):
        import chromadb
        cfg = vs_config.model_copy(update={"collection_name": name, "hnsw_ef_search": ef})
        svc = VectorStoreService(cfg, embed_service)
        svc._client = chromadb.EphemeralClient()
        return svc

    @staticmethod
    def _name():
        import uuid
        return f"vs-{uuid.uuid4().hex[:12]}"

    def test_new_collection_is_cosine_with_the_configured_ef_search(self, vs_config, embed_service):
        """Chroma 1.5 silently made an l2 collection when space came via metadata next to a configuration."""
        hnsw = self._service(vs_config, embed_service, self._name())._get_collection().configuration_json["hnsw"]
        assert (hnsw["space"], hnsw["ef_search"]) == ("cosine", 1000)

    def test_existing_collection_gets_the_configured_ef_search(self, vs_config, embed_service, caplog):
        """get_or_create ignores the configuration of an existing collection — the service applies it."""
        name = self._name()
        self._service(vs_config, embed_service, name, ef=100)._get_collection()

        with caplog.at_level("INFO"):
            col = self._service(vs_config, embed_service, name, ef=1000)._get_collection()

        assert col.configuration_json["hnsw"]["ef_search"] == 1000
        assert "ef_search 100 → 1000" in caplog.text

    def test_warns_on_legacy_l2_collection(self, vs_config, embed_service, caplog):
        """Verify a collection created without an explicit space (l2 default) triggers a wipe hint."""
        name = self._name()
        svc = self._service(vs_config, embed_service, name)
        svc._get_client().create_collection(name)          # legacy: no space given
        with caplog.at_level("WARNING"):
            svc._get_collection()
        assert "uses distance 'l2'" in caplog.text

    def test_no_warning_when_space_matches(self, vs_config, embed_service, caplog):
        """Verify no warning for a collection already using the configured metric."""
        with caplog.at_level("WARNING"):
            self._service(vs_config, embed_service, self._name())._get_collection()
        assert "uses distance" not in caplog.text


class TestGetChunksForFile:
    """Test suite for loading all chunks of one file (section expansion)."""

    def test_scoped_to_source_and_file(self, service, mock_collection):
        """Verify the query is scoped to (source_id, source_file) and returns id/content/metadata dicts."""
        mock_collection.get.return_value = {
            "ids": ["a", "b"],
            "documents": ["text a", "text b"],
            "metadatas": [{"chunk_index": 0}, {"chunk_index": 1}],
        }
        items = service.get_chunks_for_file("rules", "regelbuch.pdf")

        where = mock_collection.get.call_args[1]["where"]
        assert where == {"$and": [{"source_id": "rules"}, {"source_file": "regelbuch.pdf"}]}
        assert items == [
            {"id": "a", "content": "text a", "metadata": {"chunk_index": 0}},
            {"id": "b", "content": "text b", "metadata": {"chunk_index": 1}},
        ]


class TestUpsertChunks:
    """Test suite validating ingestion chunk transposition mapping directly towards Chroma representations."""

    def test_empty_list_is_noop(self, service, mock_collection):
        """Verify zero length insertions exit gracefully rather than throwing exceptions or pinging the database."""
        service.upsert_chunks([])
        mock_collection.upsert.assert_not_called()

    def test_upsert_called_with_correct_fields(self, service, mock_collection, sample_chunk, embed_service):
        """Verify all extracted chunk details properly map to the parallel array columns in Chroma (ids/documents/metadatas/embeddings)."""
        service.upsert_chunks([sample_chunk])
        mock_collection.upsert.assert_called_once()
        kwargs = mock_collection.upsert.call_args[1]
        assert len(kwargs["ids"]) == 1
        assert kwargs["documents"][0] == "Arkenfeld is a city."
        assert kwargs["metadatas"][0]["source_file"] == "Orte/Arkenfeld.md"

    def test_metadata_list_values_flattened(self, service, mock_collection, embed_service):
        """Verify multidimensional arrays and sets (e.g. tags, wikilinks) are safely flattened into delimited strings for compatibility."""
        chunk = Chunk(
            content="text",
            source_file="file.md",
            source_path="/file.md",
            document_type="markdown",
            metadata={"wikilinks": ["Link1", "Link2"]},
        )
        service.upsert_chunks([chunk])
        meta = mock_collection.upsert.call_args[1]["metadatas"][0]
        assert isinstance(meta["wikilinks"], str)
        assert "Link1" in meta["wikilinks"]


class TestQuery:
    """Test suite verifying query mapping logic interacting with Vector endpoints."""

    def test_returns_scored_items(self, service, mock_collection):
        """Verify Euclidean conversion translating Cosine distances accurately mapping into ascending positive relevance scores."""
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["some content"]],
            "metadatas": [[{"source_file": "f.md", "document_type": "markdown"}]],
            "distances": [[0.2]],
        }
        results = service.query([0.1, 0.2], top_k=1)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.8)
        assert results[0]["content"] == "some content"

    def test_empty_result(self, service, mock_collection):
        """Verify barren returns gracefully resolve mapping into empty generic lists."""
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        results = service.query([0.0], top_k=5)
        assert results == []

    def test_where_filter_passed_through(self, service, mock_collection):
        """Verify advanced filtering predicates accurately pipe natively to the backend layer."""
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        service.query([0.0], top_k=3, where={"document_type": "markdown"})
        kwargs = mock_collection.query.call_args[1]
        assert kwargs["where"] == {"document_type": "markdown"}


class TestGetContentHashesForSource:
    """Test suite targeting incremental re-ingestion caching utilities ensuring synchronization safety."""

    def test_returns_mapping(self, service, mock_collection):
        """Verify bulk metadata pulls cleanly condense into simple file hash dictionaries."""
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_file": "a.md", "content_hash": "hash1"},
                {"source_file": "b.md", "content_hash": "hash2"},
            ]
        }
        result = service.get_content_hashes_for_source("src1")
        assert result == {"a.md": "hash1", "b.md": "hash2"}
        # Verify the where clause scopes by source_id
        assert mock_collection.get.call_args[1]["where"] == {"source_id": "src1"}

    def test_skips_entries_without_hash(self, service, mock_collection):
        """Verify fragmented or malformed hash metadata strips cleanly avoiding corruption cascades."""
        mock_collection.get.return_value = {"metadatas": [{"source_file": "a.md"}]}
        result = service.get_content_hashes_for_source("src1")
        assert result == {}

    def test_database_error_is_raised(self, service, mock_collection):
        """An empty result would make ingestion treat every file as new and leave stale chunks — raise."""
        mock_collection.get.side_effect = Exception("chroma down")
        with pytest.raises(Exception, match="chroma down"):
            service.get_content_hashes_for_source("src1")


class TestDeleteBySourceFile:
    """Test suite for file-scoped chunk deletion."""

    def test_returns_deleted_chunk_count(self, service, mock_collection):
        """Verify the method deletes scoped by source_id + source_file and reports the chunk count."""
        mock_collection.get.return_value = {"ids": ["a", "b", "c"]}
        assert service.delete_by_source_file("src1", "a.md") == 3
        where = {"$and": [{"source_id": "src1"}, {"source_file": "a.md"}]}
        mock_collection.delete.assert_called_once_with(where=where)

    def test_skips_delete_when_nothing_stored(self, service, mock_collection):
        """Verify no delete call is issued for a file without chunks."""
        mock_collection.get.return_value = {"ids": []}
        assert service.delete_by_source_file("src1", "a.md") == 0
        mock_collection.delete.assert_not_called()


class TestHealthCheck:
    """Test suite targeting system telemetry polling capabilities."""

    def test_returns_true_when_reachable(self, service):
        """Verify standard telemetry pulses validate correctly against the host API."""
        assert service.health_check() is True

    def test_returns_false_on_exception(self, service):
        """Verify connection exceptions flag the system gracefully as unreachable."""
        service._client.heartbeat.side_effect = Exception("timeout")
        assert service.health_check() is False


class TestListSourceIds:
    """The ingest compares these with sources.yaml to delete chunks of removed sources."""

    def test_distinct_ids_without_empty_ones(self, service, mock_collection):
        mock_collection.get.return_value = {"metadatas": [
            {"source_id": "vault"}, {"source_id": "regelbuch"}, {"source_id": "vault"}, {"source_id": ""}, {}, None,
        ]}
        assert service.list_source_ids() == {"vault", "regelbuch"}

    def test_database_error_is_raised(self, service, mock_collection):
        """An error must not look like an empty index — the caller would then delete nothing, but must know."""
        mock_collection.get.side_effect = RuntimeError("locked")
        with pytest.raises(RuntimeError):
            service.list_source_ids()


class TestListFiles:
    """Chunk inspector: indexed files of one source."""

    def test_sorted_unique_files_of_the_source(self, service, mock_collection):
        mock_collection.get.return_value = {"metadatas": [
            {"source_file": "b.md", "content_hash": "h2"}, {"source_file": "a.md", "content_hash": "h1"},
            {"source_file": "b.md", "content_hash": "h2"}, {},
        ]}
        assert service.list_files("vault") == ["a.md", "b.md"]
        assert mock_collection.get.call_args[1]["where"] == {"source_id": "vault"}

    def test_database_error_is_raised(self, service, mock_collection):
        """An error must not look like an empty source."""
        mock_collection.get.side_effect = RuntimeError("locked")
        with pytest.raises(RuntimeError):
            service.list_files("vault")


class TestWipeCollection:
    def test_missing_collection_is_fine(self, service):
        from chromadb.errors import NotFoundError
        service._client.delete_collection.side_effect = NotFoundError("Collection does not exist")
        service.wipe_collection()   # no error: nothing to wipe

    def test_other_errors_reach_the_caller(self, service):
        service._client.delete_collection.side_effect = RuntimeError("database is locked")
        with pytest.raises(RuntimeError, match="locked"):
            service.wipe_collection()




class TestCompareWithExactSearch:
    """The search-exactness check (roadmap 5.5) against a real in-memory ChromaDB with random vectors."""

    @staticmethod
    def _store(vs_config, embed_service, ef):
        import uuid

        import chromadb
        import numpy as np
        cfg = vs_config.model_copy(update={"collection_name": f"ex-{uuid.uuid4().hex[:12]}", "hnsw_ef_search": ef})
        svc = VectorStoreService(cfg, embed_service)
        svc._client = chromadb.EphemeralClient()
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(3000, 24)).astype("float32")
        col = svc._get_collection()
        for start in range(0, 3000, 1000):
            col.add(ids=[f"c{i}" for i in range(start, start + 1000)],
                    embeddings=vectors[start:start + 1000].tolist(),
                    metadatas=[{"document_type": "image" if i % 50 == 0 else "markdown"}
                               for i in range(start, start + 1000)])
        queries = rng.normal(size=(40, 24)).astype("float32").tolist()
        return svc, queries

    def test_a_narrow_search_misses_chunks_of_the_exact_top_k(self, vs_config, embed_service):
        svc, queries = self._store(vs_config, embed_service, ef=15)
        missed = svc.compare_with_exact_search(queries, 15, {"document_type": {"$ne": "image"}})
        assert len(missed) == 40 and sum(missed) > 0

    def test_a_wide_search_is_exact(self, vs_config, embed_service):
        svc, queries = self._store(vs_config, embed_service, ef=3000)
        assert svc.compare_with_exact_search(queries, 15, {"document_type": {"$ne": "image"}}) == [0] * 40

    def test_tied_chunks_at_the_cut_off_are_not_deviations(self, vs_config, embed_service):
        """Every vector stored three times, top_k 14: a triple straddles the cut-off for most queries,
        and which copies HNSW returns is arbitrary. Comparing ID sets reported ~half the queries."""
        import uuid

        import chromadb
        import numpy as np
        cfg = vs_config.model_copy(update={"collection_name": f"tie-{uuid.uuid4().hex[:12]}", "hnsw_ef_search": 3000})
        svc = VectorStoreService(cfg, embed_service)
        svc._client = chromadb.EphemeralClient()
        rng = np.random.default_rng(11)
        base = rng.normal(size=(300, 24)).astype("float32")
        svc._get_collection().add(ids=[f"c{i}-{copy}" for i in range(300) for copy in range(3)],
                                  embeddings=[base[i].tolist() for i in range(300) for _ in range(3)],
                                  metadatas=[{"document_type": "markdown"}] * 900)
        queries = rng.normal(size=(40, 24)).astype("float32").tolist()

        assert svc.compare_with_exact_search(queries, 14, None) == [0] * 40


class TestGetTableEntries:
    """Entry slot (roadmap 3.10): find tables by a row's first cell — real in-memory Chroma for $contains."""

    WEAPONS = ("7.1.1 Waffen > Nahkampfwaffen\n\n|**Waffe**|**Kat.**|**normal**|\n|---|---|---|\n"
               "|**Dolch**|leicht|STÄ + 2|\n|**Kurzschwert**|normal|STÄ + 2|")
    MAGIC = ("7.2.1 Magische Waffen\n\n|**Waffe**|**Wirkung**|\n|---|---|\n"
             "|**Dolch der Schatten**|Schattenschaden|")
    NOT_FIRST = "Loot\n\n|**Fundort**|**Beute**|\n|---|---|\n|**Gruft**|**Dolch**|"

    @pytest.fixture
    def store(self, vs_config, embed_service):
        import uuid

        import chromadb
        svc = VectorStoreService(vs_config, embed_service)
        svc._collection = chromadb.EphemeralClient().create_collection(
            f"entries_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
        svc._collection.add(
            ids=["weapons", "magic", "loot"],
            documents=[self.WEAPONS, self.MAGIC, self.NOT_FIRST],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            metadatas=[{"source_id": "rules"}, {"source_id": "rules"}, {"source_id": "gm"}],
        )
        return svc

    def test_finds_the_table_whose_row_starts_with_the_name(self, store):
        hits = store.get_table_entries(["Dolch"], [1.0, 0.0, 0.0])
        # not "Dolch der Schatten" (other cell), not the loot table ("Dolch" is not the row's first cell)
        assert [h["id"] for h in hits] == ["weapons"]
        assert hits[0]["score"] == pytest.approx(1.0)  # real cosine to the query

    def test_request_filter_is_applied(self, store):
        """Player mode: the slot may only return what the request's filter allows."""
        assert store.get_table_entries(["Dolch"], [1.0, 0.0, 0.0], where={"source_id": "gm"}) == []
        assert store.get_table_entries(["Gruft"], [1.0, 0.0, 0.0], where={"source_id": "gm"})[0]["id"] == "loot"
        assert store.get_table_entries(["Gruft"], [1.0, 0.0, 0.0], where={"source_id": "rules"}) == []

    def test_more_matched_names_rank_first_and_limit_holds(self, store):
        hits = store.get_table_entries(["Dolch der Schatten", "Dolch", "Kurzschwert"], [0.0, 1.0, 0.0], limit=1)
        assert [h["id"] for h in hits] == ["weapons"]  # two names beat a better cosine with one

    def test_no_names_no_lookup(self, service, mock_collection):
        assert service.get_table_entries([], [1.0]) == []
        mock_collection.get.assert_not_called()
