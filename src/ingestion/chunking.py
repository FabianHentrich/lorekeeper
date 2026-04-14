from dataclasses import dataclass, field

from src.config.manager import ChunkingConfig
from src.ingestion.parsers.base import ParsedDocument


@dataclass
class Chunk:
    """Represents a discrete sequence of text ready for embedding and insertion into the vector store.

    Contains the extracted raw text along with inherited metadata (like file source and heading hierarchy)
    that allows the LLM and the UI to trace the information back to its origin.
    """
    content: str
    source_file: str
    source_path: str
    document_type: str
    heading_hierarchy: list[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: dict = field(default_factory=dict)


def chunk_documents(documents: list[ParsedDocument], config: ChunkingConfig) -> list[Chunk]:
    """Route a list of parsed documents through the configured chunking strategy.

    Available strategies: 'heading_aware', 'recursive', or 'fixed_size'.
    """
    strategy = config.strategy
    if strategy == "heading_aware":
        return _heading_aware_chunking(documents, config)
    elif strategy == "recursive":
        return _recursive_chunking(documents, config)
    elif strategy == "fixed_size":
        return _fixed_size_chunking(documents, config)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def _heading_aware_chunking(documents: list[ParsedDocument], config: ChunkingConfig) -> list[Chunk]:
    """Each ParsedDocument section becomes a chunk; tables are kept atomic,
    oversized prose sections get split recursively.

    This is the preferred strategy because it naturally bounds context to structural sections
    defined by the author (like Markdown headings).
    """
    chunks = []
    for doc in documents:
        text = doc.content.strip()
        if not text:
            continue

        for block_text, is_table in _split_into_blocks(text):
            if is_table:
                # Tables are atomic — split only at row boundaries if very large
                if _estimate_tokens(block_text) <= config.max_chunk_size * 3:
                    chunks.append(_doc_to_chunk(doc, block_text))
                else:
                    for sub in _split_table_by_rows(block_text, config.max_chunk_size):
                        chunks.append(_doc_to_chunk(doc, sub))
            else:
                if _estimate_tokens(block_text) <= config.max_chunk_size:
                    chunks.append(_doc_to_chunk(doc, block_text))
                else:
                    for sub in _recursive_split(block_text, config.max_chunk_size, config.chunk_overlap):
                        chunks.append(_doc_to_chunk(doc, sub))

    chunks = _merge_small_chunks(chunks, config.min_chunk_size)
    _set_indices(chunks)
    return chunks


def _recursive_chunking(documents: list[ParsedDocument], config: ChunkingConfig) -> list[Chunk]:
    """Split documents using a sequence of separators (e.g. paragraphs, then sentences, then words),
    attempting to keep chunks under the maximum size limit with overlap.
    """
    chunks = []
    for doc in documents:
        text = doc.content.strip()
        if not text:
            continue
        sub_texts = _recursive_split(text, config.max_chunk_size, config.chunk_overlap)
        for sub in sub_texts:
            chunks.append(_doc_to_chunk(doc, sub))

    chunks = _merge_small_chunks(chunks, config.min_chunk_size)
    _set_indices(chunks)
    return chunks


def _fixed_size_chunking(documents: list[ParsedDocument], config: ChunkingConfig) -> list[Chunk]:
    """A naive fallback strategy that splits text strictly by a fixed word count and overlaps."""
    chunks = []
    for doc in documents:
        text = doc.content.strip()
        if not text:
            continue
        words = text.split()
        size = config.max_chunk_size
        overlap = config.chunk_overlap
        i = 0
        while i < len(words):
            chunk_words = words[i:i + size]
            chunks.append(_doc_to_chunk(doc, " ".join(chunk_words)))
            i += size - overlap

    chunks = _merge_small_chunks(chunks, config.min_chunk_size)
    _set_indices(chunks)
    return chunks


def _split_into_blocks(text: str) -> list[tuple[str, bool]]:
    """Split text into (content, is_table) blocks.

    A table block is a contiguous sequence of lines starting with '|'.
    Prose blocks are everything else.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, bool]] = []
    current: list[str] = []
    in_table = False

    for line in lines:
        line_is_table = bool(line.strip()) and line.strip().startswith("|")

        if line_is_table != in_table and current:
            block = "\n".join(current).strip()
            if block:
                blocks.append((block, in_table))
            current = []
            in_table = line_is_table
        elif not current:
            in_table = line_is_table

        current.append(line)

    if current:
        block = "\n".join(current).strip()
        if block:
            blocks.append((block, in_table))

    return blocks


def _split_table_by_rows(text: str, max_size: int) -> list[str]:
    """Split a very large table at row boundaries, repeating the header in each part."""
    lines = [l for l in text.split("\n") if l.strip()]

    # Identify header: first line + optional separator line (contains ---)
    header_lines: list[str] = []
    data_lines: list[str] = []
    for i, line in enumerate(lines):
        if i == 0 or (i == 1 and "---" in line):
            header_lines.append(line)
        else:
            data_lines.append(line)

    header = "\n".join(header_lines)
    parts: list[str] = []
    current_rows = list(header_lines)

    for row in data_lines:
        candidate = "\n".join(current_rows + [row])
        if _estimate_tokens(candidate) > max_size and len(current_rows) > len(header_lines):
            parts.append("\n".join(current_rows))
            current_rows = list(header_lines) + [row]
        else:
            current_rows.append(row)

    if current_rows:
        parts.append("\n".join(current_rows))

    return [p for p in parts if p.strip()]


def _recursive_split(text: str, max_size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]
    return _split_with_separators(text, max_size, overlap, separators)


def _split_with_separators(text: str, max_size: int, overlap: int, separators: list[str]) -> list[str]:
    if _estimate_tokens(text) <= max_size:
        return [text]

    if not separators:
        # Hard split by words as last resort
        words = text.split()
        parts = []
        i = 0
        while i < len(words):
            parts.append(" ".join(words[i:i + max_size]))
            i += max_size - overlap
        return parts

    sep = separators[0]
    segments = text.split(sep)

    parts = []
    current = ""
    for segment in segments:
        candidate = (current + sep + segment).strip() if current else segment.strip()
        if _estimate_tokens(candidate) > max_size and current:
            parts.append(current.strip())
            overlap_text = _get_overlap_text(current, overlap)
            current = (overlap_text + sep + segment).strip() if overlap_text else segment.strip()
        else:
            current = candidate

    if current.strip():
        parts.append(current.strip())

    result = []
    for part in parts:
        if _estimate_tokens(part) > max_size:
            result.extend(_split_with_separators(part, max_size, overlap, separators[1:]))
        else:
            result.append(part)

    return result


def _get_overlap_text(text: str, overlap_tokens: int) -> str:
    words = text.split()
    if len(words) <= overlap_tokens:
        return text
    return " ".join(words[-overlap_tokens:])


def _estimate_tokens(text: str) -> int:
    """Estimate tokens via character count. German averages ~3.5 chars per token."""
    return max(1, int(len(text) / 3.5))


def _doc_to_chunk(doc: ParsedDocument, content: str) -> Chunk:
    """Convert a ParsedDocument slice into a Chunk object, injecting heading paths.

    Prepend heading path so document title/section is part of the embedded text.
    Without this, a chunk from 'Arkenfeld.md > Steckbrief' contains no mention
    of 'Arkenfeld' and an embedding search for 'was ist Arkenfeld?' finds nothing.
    """
    # Prepend heading path so document title/section is part of the embedded text.
    if doc.heading_hierarchy:
        heading_prefix = " > ".join(doc.heading_hierarchy)
        content = f"{heading_prefix}\n\n{content}"

    return Chunk(
        content=content,
        source_file=doc.source_file,
        source_path=doc.source_path,
        document_type=doc.document_type,
        heading_hierarchy=list(doc.heading_hierarchy),
        metadata=dict(doc.metadata),
    )


def _merge_small_chunks(chunks: list[Chunk], min_size: int) -> list[Chunk]:
    """Merge tiny chunks into the previous one — but only within the same
    heading boundary.

    Merging across headings would make the chunk's heading_hierarchy lie about half
    its content (e.g. a tiny "Bruchgraben" table absorbing the next section "Verborgenes Erbe"
    and still being labelled "Bruchgraben"), which corrupts both retrieval display and the
    per-source diversity logic in the reranker.
    """
    if not chunks:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        same_section = (
            prev.source_file == chunk.source_file
            and prev.heading_hierarchy == chunk.heading_hierarchy
        )
        if _estimate_tokens(prev.content) < min_size and same_section:
            prev.content += "\n\n" + chunk.content
        else:
            merged.append(chunk)

    return merged


def _set_indices(chunks: list[Chunk]):
    """Assign sequential indices and total counts to chunks grouped by their source file."""
    groups: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.source_file, []).append(chunk)

    for group in groups.values():
        total = len(group)
        for i, chunk in enumerate(group):
            chunk.chunk_index = i
            chunk.total_chunks = total
