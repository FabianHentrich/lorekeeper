import re
from pathlib import Path

import pymupdf4llm

from .base import BaseParser, ParsedDocument
from .markdown import MarkdownParser

_NUMBERED_HEADING = re.compile(r'^(#{1,6})(\s+)(\*{0,2}\d+[\d.]*\s*.+)', re.MULTILINE)


def _normalize_pdf_headings(md_text: str) -> str:
    """Promote numbered section headings (e.g. '## **5.4.5 Mönch**') to H1.

    pymupdf4llm renders all headings at the same ## level, so numbered section
    titles and their subsections become siblings. Promoting numbered headings
    makes subsections correctly nest beneath their parent class/section.
    """
    def promote(m: re.Match) -> str:
        return f"#{m.group(2)}{m.group(3)}"

    return _NUMBERED_HEADING.sub(promote, md_text)


class PDFParser(BaseParser):
    def __init__(self):
        self._md_parser = MarkdownParser()

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path, base_path: Path | None = None) -> list[ParsedDocument]:
        md_text = _normalize_pdf_headings(pymupdf4llm.to_markdown(str(file_path)))

        source_file = file_path.name
        source_path = str(file_path.resolve())

        sections = self._md_parser._split_by_headings(md_text)

        documents = []
        for heading_hierarchy, section_content in sections:
            section_content = section_content.strip()
            if not section_content:
                continue

            documents.append(ParsedDocument(
                content=section_content,
                source_file=source_file,
                source_path=source_path,
                document_type="pdf",
                heading_hierarchy=heading_hierarchy,
                metadata={},
            ))

        return documents if documents else [ParsedDocument(
            content="(empty PDF)",
            source_file=source_file,
            source_path=source_path,
            document_type="pdf",
            heading_hierarchy=[],
            metadata={},
        )]
