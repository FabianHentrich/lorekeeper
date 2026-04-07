import pytest

from src.config.manager import ChunkingConfig
from src.ingestion.chunking import chunk_documents, _estimate_tokens
from src.ingestion.parsers.base import ParsedDocument


def _make_doc(content: str, source="test.md", heading=None):
    return ParsedDocument(
        content=content,
        source_file=source,
        source_path=f"/path/{source}",
        document_type="markdown",
        heading_hierarchy=heading or [],
    )


class TestChunking:
    def test_heading_aware_small_sections(self):
        # Use long enough content to avoid merging
        content_a = " ".join(["Wort"] * 80)
        content_b = " ".join(["Text"] * 80)
        docs = [
            _make_doc(content_a, heading=["H1", "A"]),
            _make_doc(content_b, heading=["H1", "B"]),
        ]
        config = ChunkingConfig(strategy="heading_aware", max_chunk_size=512, min_chunk_size=5)
        chunks = chunk_documents(docs, config)
        assert len(chunks) == 2
        assert "Wort" in chunks[0].content
        assert "Text" in chunks[1].content

    def test_heading_aware_splits_large_section(self):
        long_text = " ".join(["Wort"] * 1000)
        docs = [_make_doc(long_text)]
        config = ChunkingConfig(strategy="heading_aware", max_chunk_size=100, chunk_overlap=10, min_chunk_size=5)
        chunks = chunk_documents(docs, config)
        assert len(chunks) > 1

    def test_merges_small_chunks(self):
        docs = [
            _make_doc("Tiny.", heading=["A"]),
            _make_doc("Also tiny.", heading=["B"]),
        ]
        config = ChunkingConfig(strategy="heading_aware", max_chunk_size=512, min_chunk_size=100)
        chunks = chunk_documents(docs, config)
        assert len(chunks) == 1  # Merged because both < min_chunk_size

    def test_recursive_strategy(self):
        text = "\n\n".join([" ".join(["Wort"] * 50) for _ in range(5)])
        docs = [_make_doc(text)]
        config = ChunkingConfig(strategy="recursive", max_chunk_size=50, chunk_overlap=5, min_chunk_size=1)
        chunks = chunk_documents(docs, config)
        assert len(chunks) >= 2

    def test_fixed_size_strategy(self):
        text = " ".join(["word"] * 100)
        docs = [_make_doc(text)]
        config = ChunkingConfig(strategy="fixed_size", max_chunk_size=30, chunk_overlap=5, min_chunk_size=1)
        chunks = chunk_documents(docs, config)
        assert len(chunks) >= 3

    def test_chunk_indices_set(self):
        content_a = " ".join(["Alpha"] * 80)
        content_b = " ".join(["Beta"] * 80)
        docs = [
            _make_doc(content_a, heading=["A"]),
            _make_doc(content_b, heading=["B"]),
        ]
        config = ChunkingConfig(strategy="heading_aware", max_chunk_size=512, min_chunk_size=5)
        chunks = chunk_documents(docs, config)
        assert len(chunks) == 2
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[0].total_chunks == 2

    def test_empty_documents(self):
        docs = [_make_doc("")]
        config = ChunkingConfig(strategy="heading_aware")
        chunks = chunk_documents(docs, config)
        assert len(chunks) == 0

    def test_unknown_strategy_raises(self):
        docs = [_make_doc("Content")]
        config = ChunkingConfig(strategy="unknown")
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_documents(docs, config)


class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("Ein kurzer Satz.") >= 1

    def test_empty(self):
        assert _estimate_tokens("") >= 1
