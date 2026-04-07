from unittest.mock import MagicMock, patch

import pytest

from src.config.manager import EmbeddingsConfig, VectorStoreConfig
from src.ingestion.chunking import Chunk
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vectorstore import VectorStoreService


@pytest.fixture
def vs_config():
    return VectorStoreConfig(
        mode="embedded",
        persist_directory="./test_chroma",
        collection_name="test",
        distance_metric="cosine",
    )


@pytest.fixture
def embed_service():
    svc = MagicMock(spec=EmbeddingService)
    svc.embed_texts_sync.return_value = [[0.1, 0.2, 0.3]]
    return svc


@pytest.fixture
def mock_collection():
    col = MagicMock()
    col.count.return_value = 0
    return col


@pytest.fixture
def service(vs_config, embed_service, mock_collection):
    svc = VectorStoreService(vs_config, embed_service)
    svc._collection = mock_collection
    mock_client = MagicMock()
    mock_client.heartbeat.return_value = True
    svc._client = mock_client
    return svc


@pytest.fixture
def sample_chunk():
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
    def test_includes_filename_stem(self, service, sample_chunk):
        text = service._build_embed_text(sample_chunk)
        assert "Arkenfeld" in text

    def test_includes_aliases(self, service, sample_chunk):
        text = service._build_embed_text(sample_chunk)
        assert "Arken" in text

    def test_includes_content(self, service, sample_chunk):
        text = service._build_embed_text(sample_chunk)
        assert "Arkenfeld is a city." in text

    def test_alias_list(self, service):
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
    def test_deterministic(self, service, sample_chunk):
        id1 = service._chunk_id(sample_chunk)
        id2 = service._chunk_id(sample_chunk)
        assert id1 == id2

    def test_different_for_different_source(self, service, sample_chunk):
        other = Chunk(
            content="other",
            source_file="other.md",
            source_path="/other.md",
            document_type="markdown",
            chunk_index=0,
        )
        assert service._chunk_id(sample_chunk) != service._chunk_id(other)


class TestUpsertChunks:
    def test_empty_list_is_noop(self, service, mock_collection):
        service.upsert_chunks([])
        mock_collection.upsert.assert_not_called()

    def test_upsert_called_with_correct_fields(self, service, mock_collection, sample_chunk, embed_service):
        service.upsert_chunks([sample_chunk])
        mock_collection.upsert.assert_called_once()
        kwargs = mock_collection.upsert.call_args[1]
        assert len(kwargs["ids"]) == 1
        assert kwargs["documents"][0] == "Arkenfeld is a city."
        assert kwargs["metadatas"][0]["source_file"] == "Orte/Arkenfeld.md"

    def test_metadata_list_values_flattened(self, service, mock_collection, embed_service):
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
    def test_returns_scored_items(self, service, mock_collection):
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
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        results = service.query([0.0], top_k=5)
        assert results == []

    def test_where_filter_passed_through(self, service, mock_collection):
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        service.query([0.0], top_k=3, where={"document_type": "markdown"})
        kwargs = mock_collection.query.call_args[1]
        assert kwargs["where"] == {"document_type": "markdown"}


class TestGetAllContentHashes:
    def test_returns_mapping(self, service, mock_collection):
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_file": "a.md", "content_hash": "hash1"},
                {"source_file": "b.md", "content_hash": "hash2"},
            ]
        }
        result = service.get_all_content_hashes()
        assert result == {"a.md": "hash1", "b.md": "hash2"}

    def test_skips_entries_without_hash(self, service, mock_collection):
        mock_collection.get.return_value = {
            "metadatas": [{"source_file": "a.md"}]
        }
        result = service.get_all_content_hashes()
        assert result == {}

    def test_returns_empty_on_exception(self, service, mock_collection):
        mock_collection.get.side_effect = Exception("chroma down")
        result = service.get_all_content_hashes()
        assert result == {}


class TestHealthCheck:
    def test_returns_true_when_reachable(self, service):
        assert service.health_check() is True

    def test_returns_false_on_exception(self, service):
        service._client.heartbeat.side_effect = Exception("timeout")
        assert service.health_check() is False
