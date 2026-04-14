import logging
import re
import tempfile
from pathlib import Path

import pymupdf4llm

from src.config.manager import PdfConfig
from .base import BaseParser, ParsedDocument
from .markdown import MarkdownParser

logger = logging.getLogger(__name__)

_NUMBERED_HEADING = re.compile(r'^(#{1,6})(\s+)(\*{0,2}\d+[\d.]*\s*.+)', re.MULTILINE)
_IMAGE_REF = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')


def _normalize_pdf_headings(md_text: str) -> str:
    """Promote numbered section headings (e.g. '## **5.4.5 Mnch**') to H1.

    pymupdf4llm renders all headings at the same ## level, so numbered section
    titles and their subsections become siblings. Promoting numbered headings
    makes subsections correctly nest beneath their parent class/section.
    """
    def promote(m: re.Match) -> str:
        return f"#{m.group(2)}{m.group(3)}"

    return _NUMBERED_HEADING.sub(promote, md_text)


def _apply_toc_headings(md_text: str, toc_items: list[list]) -> str:
    """Replace flat headings with correctly levelled ones using PDF TOC data.

    pymupdf4llm's toc_items are [level, title, page_number] entries from the
    PDF's table of contents. When available, they give us the true heading
    hierarchy that pymupdf4llm's font-size heuristics often miss.
    """
    if not toc_items:
        return md_text

    # Build lookup: normalised title -> target level
    toc_levels: dict[str, int] = {}
    for entry in toc_items:
        level, title = entry[0], entry[1]
        normalised = re.sub(r'\s+', ' ', title.strip().lower())
        toc_levels[normalised] = level

    def _fix_heading(m: re.Match) -> str:
        hashes, rest = m.group(1), m.group(2)
        # Strip bold markers for lookup
        clean = re.sub(r'\*{1,2}', '', rest).strip()
        normalised = re.sub(r'\s+', ' ', clean.lower())
        target = toc_levels.get(normalised)
        if target is not None:
            return f"{'#' * target} {rest}"
        return m.group(0)

    return re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE).sub(_fix_heading, md_text)


def _extract_image_docs(md_text: str, image_dir: Path, source_file: str,
                        source_path: str) -> tuple[str, list[ParsedDocument]]:
    """Find image references in markdown, create ParsedDocuments for existing
    image files, and replace references with descriptive placeholder text.

    Returns the cleaned markdown text and a list of image stubs representing
    the extracted visuals.
    """
    image_docs: list[ParsedDocument] = []

    def _replace_image(m: re.Match) -> str:
        alt_text = m.group(1)
        img_path = Path(m.group(2))

        # Resolve relative paths against image_dir
        if not img_path.is_absolute():
            img_path = image_dir / img_path

        if not img_path.exists():
            return ""

        name = img_path.stem.replace("_", " ").replace("-", " ")
        label = alt_text or name

        image_docs.append(ParsedDocument(
            content=f"Bild: {label} (aus PDF: {source_file})",
            source_file=source_file,
            source_path=str(img_path.resolve()),
            document_type="image",
            heading_hierarchy=[source_file, label],
            metadata={"image_format": img_path.suffix.lower(), "extracted_from_pdf": True},
        ))

        return f"[Bild: {label}]"

    cleaned = _IMAGE_REF.sub(_replace_image, md_text)
    return cleaned, image_docs


class PDFParser(BaseParser):
    """Parser that converts PDF documents into structured text blocks.

    Uses pymupdf4llm to extract text and structure (TOCs, tables) into a raw
    Markdown format. It attempts OCR on embedded images if configured, and
    reconstructs broken heading hierarchies before delegating the final slicing
    to the MarkdownParser logic.
    """
    def __init__(self, pdf_config: PdfConfig | None = None):
        self._md_parser = MarkdownParser()
        self._config = pdf_config or PdfConfig()

    def can_parse(self, file_path: Path) -> bool:
        """Check if the provided file is a PDF."""
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path, base_path: Path | None = None) -> list[ParsedDocument]:
        """Extract text from the PDF, normalize headings, and return structural blocks.

        Creates a temporary directory for intermediary image extraction if enabled.
        Catches OCR failures and automatically retries with pure text extraction fallback.
        """
        source_file = file_path.name
        source_path = str(file_path.resolve())
        cfg = self._config

        with tempfile.TemporaryDirectory(prefix="lorekeeper_pdf_") as tmp_dir:
            image_dir = Path(tmp_dir)

            kwargs: dict = {
                "show_progress": False,
                "ignore_code": True,
                "page_chunks": True,
            }

            # OCR
            if cfg.ocr_enabled:
                kwargs["use_ocr"] = True
                kwargs["ocr_language"] = cfg.ocr_language
                kwargs["ocr_dpi"] = cfg.ocr_dpi
            else:
                kwargs["use_ocr"] = False

            # Image extraction
            if cfg.extract_images:
                kwargs["write_images"] = True
                kwargs["image_path"] = str(image_dir)
                kwargs["image_format"] = cfg.image_format

            try:
                page_chunks = pymupdf4llm.to_markdown(str(file_path), **kwargs)
            except Exception:
                logger.exception("pymupdf4llm failed for %s, retrying without OCR", file_path.name)
                kwargs["use_ocr"] = False
                kwargs.pop("ocr_language", None)
                kwargs.pop("ocr_dpi", None)
                page_chunks = pymupdf4llm.to_markdown(str(file_path), **kwargs)

            # Combine page texts, applying heading corrections per page
            all_image_docs: list[ParsedDocument] = []
            page_texts: list[str] = []

            for page in page_chunks:
                text = page["text"]
                toc_items = page.get("toc_items", [])

                # 1. Use TOC data for heading hierarchy (most reliable)
                text = _apply_toc_headings(text, toc_items)
                # 2. Fallback regex for numbered headings not in TOC
                text = _normalize_pdf_headings(text)

                # Extract image references
                if cfg.extract_images:
                    text, img_docs = _extract_image_docs(
                        text, image_dir, source_file, source_path
                    )
                    all_image_docs.extend(img_docs)

                page_texts.append(text)

            combined_md = "\n\n".join(page_texts)

            # Split into sections by heading hierarchy (reuse MarkdownParser logic)
            sections = self._md_parser._split_by_headings(combined_md)

            documents: list[ParsedDocument] = []
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

            # Add extracted image documents
            documents.extend(all_image_docs)

            if not documents:
                documents.append(ParsedDocument(
                    content="(empty PDF)",
                    source_file=source_file,
                    source_path=source_path,
                    document_type="pdf",
                    heading_hierarchy=[],
                    metadata={},
                ))

            return documents
