from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    content: str
    source_file: str
    source_path: str
    document_type: str
    heading_hierarchy: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseParser(ABC):
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool: ...

    @abstractmethod
    def parse(self, file_path: Path, base_path: Path | None = None) -> list[ParsedDocument]: ...
