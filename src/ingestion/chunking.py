"""Chunking: parsed documents → chunks for embedding.

Every chunk starts with its heading path (so a section without the entity's name still matches
questions about it). Sizes are estimated tokens (``len(text) / 3.5``); ``max_chunk_size`` keeps the
embed text under the 512-token limit of e5 and the reranker.
"""
import re
from dataclasses import dataclass, field

from src.config.manager import ChunkingConfig
from src.ingestion.parsers.base import ParsedDocument


@dataclass
class Chunk:
    """A text slice ready for embedding, carrying source/heading metadata for traceability in the UI."""
    content: str
    source_file: str
    source_path: str
    document_type: str
    heading_hierarchy: list[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: dict = field(default_factory=dict)


def chunk_documents(documents: list[ParsedDocument], config: ChunkingConfig) -> list[Chunk]:
    """Split parsed documents into chunks with the configured strategy.

    Strategies: 'heading_aware' (default — one section per chunk, tables atomic, long prose
    split recursively), 'recursive' (paragraph → line → sentence → word) and 'fixed_size' (word
    windows). Afterwards tiny chunks are merged within their heading and indices are set per file.
    """
    split = _STRATEGIES.get(config.strategy)
    if split is None:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")
    chunks = [
        _doc_to_chunk(doc, piece)
        for doc in documents
        if (text := doc.content.strip())
        for piece in split(text, config)
        if _HAS_CONTENT.search(piece)  # a piece of only "---" or "|  |" would be the heading path alone
    ]
    chunks = _merge_small_chunks(chunks, config.min_chunk_size)
    _set_indices(chunks)
    return chunks


def _heading_aware_split(text: str, config: ChunkingConfig) -> list[str]:
    """One piece per prose/table block of a section. Tables stay atomic (split only at rows when
    very large); oversized prose is split recursively."""
    pieces: list[str] = []
    for block_text, is_table in _split_into_blocks(text):
        if is_table:
            # 1.5x keeps table + heading/identity prefix under the 512-token limit of e5 and the
            # reranker; beyond it the tail is silently truncated.
            if _estimate_tokens(block_text) <= config.max_chunk_size * 1.5:
                pieces.append(block_text)
            else:
                pieces.extend(_split_table_by_rows(block_text, config.max_chunk_size))
        else:
            pieces.extend(_split_prose(block_text, config.max_chunk_size, config.chunk_overlap))
    return _attach_small_pieces(pieces, config.min_chunk_size)


def _attach_small_pieces(pieces: list[str], min_size: int) -> list[str]:
    """Join a piece under ``min_size`` tokens to the next piece of the section — mostly a label that introduces
    a table ("- L2" from a folded callout, "**Optionen:**"), which the prose/table split cut off: alone it was
    a chunk of the heading path plus two characters (14.10.: 33 in the vault) and its table lost the label.
    A small last piece joins the previous one."""
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        if carry:
            piece = f"{carry}\n\n{piece}"
        if _estimate_tokens(piece) < min_size:
            carry = piece
        else:
            merged.append(piece)
            carry = ""
    if carry:
        if merged:
            merged[-1] += f"\n\n{carry}"
        else:
            merged.append(carry)
    return merged


def _recursive_split(text: str, config: ChunkingConfig) -> list[str]:
    """The whole text split paragraph → line → sentence → word until each piece fits."""
    return _split_prose(text, config.max_chunk_size, config.chunk_overlap)


def _fixed_size_split(text: str, config: ChunkingConfig) -> list[str]:
    """Naive fallback: fixed word windows with overlap."""
    return _word_split(text, config.max_chunk_size, config.chunk_overlap)


_STRATEGIES = {
    "heading_aware": _heading_aware_split,
    "recursive": _recursive_split,
    "fixed_size": _fixed_size_split,
}

_SEPARATORS = ["\n\n", "\n", ". ", " "]
_HAS_CONTENT = re.compile(r"[^\W_]")  # a letter or digit


def _word_split(text: str, size: int, overlap: int) -> list[str]:
    """Windows of ``size`` words, each starting ``size - overlap`` words after the previous one.

    The config rejects overlap >= size; the step is still at least one word, so an invalid
    config (e.g. an old settings.local.yaml) can't make the loop stand still.
    """
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


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
        line_is_table = line.strip().startswith("|")

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
    lines = [ln for ln in text.split("\n") if ln.strip()]

    # Identify header: first line + optional separator line (contains ---)
    header_lines: list[str] = []
    data_lines: list[str] = []
    for i, line in enumerate(lines):
        if i == 0 or (i == 1 and "---" in line):
            header_lines.append(line)
        else:
            data_lines.append(line)

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


def _split_prose(text: str, max_size: int, overlap: int, separators: list[str] = _SEPARATORS) -> list[str]:
    """Recursively split by the first separator that keeps every piece under max_size;
    words are the last resort."""
    if _estimate_tokens(text) <= max_size:
        return [text]

    if not separators:
        return _word_split(text, max_size, overlap)

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
            result.extend(_split_prose(part, max_size, overlap, separators[1:]))
        else:
            result.append(part)

    return result


def _get_overlap_text(text: str, overlap_tokens: int) -> str:
    """Return the last `overlap_tokens` words of text, used to bridge successive chunks."""
    words = text.split()
    if len(words) <= overlap_tokens:
        return text
    return " ".join(words[-overlap_tokens:])


CHARS_PER_TOKEN = 3.5  # German average — also bounds the chat history (conversation manager)


def _estimate_tokens(text: str) -> int:
    """Estimate tokens via character count (``CHARS_PER_TOKEN``)."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _doc_to_chunk(doc: ParsedDocument, content: str) -> Chunk:
    """Wrap a text slice in a Chunk, prepending the heading path.

    Without the prefix, a chunk from 'Arkenfeld.md > Steckbrief' contains no
    mention of 'Arkenfeld' and searches like 'was ist Arkenfeld?' miss it.
    """
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
