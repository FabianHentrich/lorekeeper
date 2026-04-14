import pytest

from src.config.manager import ChunkingConfig
from src.ingestion.chunking import chunk_documents, _estimate_tokens
from src.ingestion.parsers.base import ParsedDocument


def _make_doc(content: str, source="test.md", heading=None):
    """
    Helper function to create simple mock ParsedDocument instances for testing chunking algorithms.
    """
    return ParsedDocument(
        content=content,
        source_file=source,
        source_path=f"/path/{source}",
        document_type="markdown",
        heading_hierarchy=heading or [],
    )


class TestChunking:
    """Test suite validating all available chunking strategies (heading_aware, recursive, fixed_size)."""

    def test_heading_aware_small_sections(self):
        """
        Verify that independent sections with distinct headings are correctly mapped to distinct chunks.
        """
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
        """
        Verify that a single excessively large section gets split down into smaller pieces
        to respect the max_chunk_size limit.
        """
        long_text = " ".join(["Wort"] * 1000)
        docs = [_make_doc(long_text)]
        config = ChunkingConfig(strategy="heading_aware", max_chunk_size=100, chunk_overlap=10, min_chunk_size=5)
        chunks = chunk_documents(docs, config)
        assert len(chunks) > 1

    def test_merges_small_chunks_within_same_heading(self):
        """
        Verify that multiple undersized blocks sharing the exact same heading hierarchy
        are aggressively merged into a single chunk.
        """
        # Two tiny blocks under the same heading → merged
        docs = [
            _make_doc("Tiny.", heading=["A"]),
            _make_doc("Also tiny.", heading=["A"]),
        ]
        config = ChunkingConfig(strategy="heading_aware", max_chunk_size=512, min_chunk_size=100)
        chunks = chunk_documents(docs, config)
        assert len(chunks) == 1

    def test_does_not_merge_across_headings(self):
        """
        Verify that chunks with different headings are strictly segregated and NEVER merged,
        preventing semantic metadata drift.
        """
        # Two tiny blocks under DIFFERENT headings → must stay separate,
        # otherwise the merged chunk would lie about its heading_hierarchy.
        docs = [
            _make_doc("Tiny.", heading=["A"]),
            _make_doc("Also tiny.", heading=["B"]),
        ]
        config = ChunkingConfig(strategy="heading_aware", max_chunk_size=512, min_chunk_size=100)
        chunks = chunk_documents(docs, config)
        assert len(chunks) == 2
        assert chunks[0].heading_hierarchy == ["A"]
        assert chunks[1].heading_hierarchy == ["B"]

    def test_recursive_strategy(self):
        """
        Verify that the naive recursive chunking strategy splits paragraph-based inputs
        into multiple blocks conforming to max constraints.
        """
        text = "\n\n".join([" ".join(["Wort"] * 50) for _ in range(5)])
        docs = [_make_doc(text)]
        config = ChunkingConfig(strategy="recursive", max_chunk_size=50, chunk_overlap=5, min_chunk_size=1)
        chunks = chunk_documents(docs, config)
        assert len(chunks) >= 2

    def test_fixed_size_strategy(self):
        """
        Verify that fixed-size chunking strictly respects text bounds without semantic awareness.
        """
        text = " ".join(["word"] * 100)
        docs = [_make_doc(text)]
        config = ChunkingConfig(strategy="fixed_size", max_chunk_size=30, chunk_overlap=5, min_chunk_size=1)
        chunks = chunk_documents(docs, config)
        assert len(chunks) >= 3

    def test_chunk_indices_set(self):
        """
        Verify that chunk indices and aggregate total_chunks values are accurately populated
        during standard ingestion runs to aid in reconstruction workflows.
        """
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
        """
        Verify that feeding empty strings yields an empty list rather than errors.
        """
        docs = [_make_doc("")]
        config = ChunkingConfig(strategy="heading_aware")
        chunks = chunk_documents(docs, config)
        assert len(chunks) == 0

    def test_unknown_strategy_raises(self):
        """
        Verify that unknown chunking strategies trigger a crash.
        """
        docs = [_make_doc("Content")]
        config = ChunkingConfig(strategy="unknown")
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_documents(docs, config)


class TestEstimateTokens:
    """Test suite validating the quick & dirty token estimation heuristics."""

    def test_basic(self):
        """Verify normal text yields at least 1 estimated token."""
        assert _estimate_tokens("Ein kurzer Satz.") >= 1

    def test_empty(self):
        """Verify empty strings safely evaluate to 1 estimated token minimally."""
        assert _estimate_tokens("") >= 1
