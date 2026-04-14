from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    """Represents a structurally intact slice of a document.

    Before chunking restricts lengths, ParsedDocuments represent larger semantic
    blocks (like a full markdown section or a full PDF page). The metadata and
    heading hierarchies captured here will be inherited by all resulting chunks.
    """
    content: str
    source_file: str
    source_path: str
    document_type: str
    heading_hierarchy: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract base class for all file format parsers in the ingestion pipeline.

    Implementations must define how to test a file for compatibility and how to
    extract semantic blocks (ParsedDocuments) from that file type.
    """
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Determine if this parser is capable of reading the given file.
        Usually checks the file extension.
        """
        ...

    @abstractmethod
    def parse(self, file_path: Path, base_path: Path | None = None) -> list[ParsedDocument]:
        """Read the file from disk and convert its contents into a list of parsed sections.

        The structural boundaries represented in the returned ParsedDocuments
        dictate where the chunking algorithms consider semantic breaks to be.
        """
        ...
