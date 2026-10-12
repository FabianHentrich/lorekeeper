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

    def _fresh(self, vs_config, embed_service, collection_metadata):
        svc = VectorStoreService(vs_config, embed_service)
        client = MagicMock()
        client.get_or_create_collection.return_value.metadata = collection_metadata
        svc._client = client
        return svc, client

    def test_cosine_space_passed_explicitly(self, vs_config, embed_service):
        """Verify cosine is requested explicitly instead of relying on Chroma's l2 default."""
        svc, client = self._fresh(vs_config, embed_service, {"hnsw:space": "cosine"})
        svc._get_collection()
        kwargs = client.get_or_create_collection.call_args[1]
        assert kwargs["metadata"] == {"hnsw:space": "cosine"}

    def test_warns_on_legacy_l2_collection(self, vs_config, embed_service, caplog):
        """Verify a collection created without hnsw:space (l2 default) triggers a wipe hint."""
        svc, _ = self._fresh(vs_config, embed_service, None)
        with caplog.at_level("WARNING"):
            svc._get_collection()
        assert "uses distance 'l2'" in caplog.text

    def test_no_warning_when_space_matches(self, vs_config, embed_service, caplog):
        """Verify no warning for a collection already using the configured metric."""
        svc, _ = self._fresh(vs_config, embed_service, {"hnsw:space": "cosine"})
        with caplog.at_level("WARNING"):
            svc._get_collection()
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

    def test_returns_empty_on_exception(self, service, mock_collection):
        """Verify total database failure during index synchronization cleanly assumes 'everything is new/modified' caching failure states."""
        mock_collection.get.side_effect = Exception("chroma down")
        result = service.get_content_hashes_for_source("src1")
        assert result == {}


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


class TestListFiles:
    """Chunk inspector: indexed files of one source."""

    def test_sorted_unique_files_of_the_source(self, service, mock_collection):
        mock_collection.get.return_value = {"metadatas": [
            {"source_file": "b.md"}, {"source_file": "a.md"}, {"source_file": "b.md"}, {},
        ]}
        assert service.list_files("vault") == ["a.md", "b.md"]
        assert mock_collection.get.call_args[1]["where"] == {"source_id": "vault"}

    def test_database_error_is_raised(self, service, mock_collection):
        """Unlike get_content_hashes_for_source, an error must not look like an empty source."""
        mock_collection.get.side_effect = RuntimeError("locked")
        with pytest.raises(RuntimeError):
            service.list_files("vault")
