from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.manager import PdfConfig
from src.ingestion.parsers.pdf import (
    PDFParser,
    _apply_toc_headings,
    _extract_image_docs,
    _normalize_pdf_headings,
)


@pytest.fixture
def pdf_parser():
    """Provides a PDFParser instance configured with default settings (OCR enabled)."""
    return PDFParser()


@pytest.fixture
def pdf_parser_no_ocr():
    """Provides a PDFParser instance explicitly disabling OCR and image extraction."""
    return PDFParser(pdf_config=PdfConfig(ocr_enabled=False, extract_images=False))


class TestNormalizePdfHeadings:
    """Test suite validating heuristics for restructuring broken PDF heading hierarchies."""

    def test_promote_numbered_heading(self):
        """Verify that strictly numbered headers (e.g. 5.4) natively elevate to H1."""
        text = "## **5.4 Mnch**\nSome content."
        result = _normalize_pdf_headings(text)
        assert result.startswith("# **5.4 Mnch**")

    def test_leave_non_numbered_heading(self):
        """Verify that semantic string headers aren't falsely promoted by numbering rules."""
        text = "## Einleitung\nSome content."
        result = _normalize_pdf_headings(text)
        assert result == text

    def test_promote_multiple(self):
        """Verify that multiple numerical headers sequentially contained within one block are all promoted."""
        text = "## **1. Rassen**\nText.\n## **1.1 Elfen**\nMehr."
        result = _normalize_pdf_headings(text)
        assert result.count("# **1. Rassen**") == 1
        assert result.count("# **1.1 Elfen**") == 1


class TestApplyTocHeadings:
    """Test suite validating PDF Table of Contents injection into flattened markdown."""

    def test_correct_heading_levels(self):
        """Verify that explicit TOC depth definitions rewrite markdown syntax headers accordingly."""
        md = "## Kapitel Eins\nText.\n## Unterabschnitt\nMehr."
        toc = [[1, "Kapitel Eins", 1], [2, "Unterabschnitt", 1]]
        result = _apply_toc_headings(md, toc)
        assert "# Kapitel Eins" in result
        assert "## Unterabschnitt" in result

    def test_no_toc_returns_unchanged(self):
        """Verify that an absence of a TOC payload leaves the raw markdown untouched."""
        md = "## Heading\nContent."
        result = _apply_toc_headings(md, [])
        assert result == md

    def test_bold_heading_matched(self):
        """Verify that wildcard bold markdown syntax surrounding headers still match strictly against TOC."""
        md = "## **Gtter**\nText."
        toc = [[1, "Gtter", 1]]
        result = _apply_toc_headings(md, toc)
        assert result.startswith("# **Gtter**")

    def test_unmatched_heading_unchanged(self):
        """Verify that markdown headers lacking a TOC counterpart remain structurally unmodified."""
        md = "## Unbekannt\nText."
        toc = [[1, "Kapitel", 1]]
        result = _apply_toc_headings(md, toc)
        assert "## Unbekannt" in result


class TestExtractImageDocs:
    """Test suite validating image metadata detachment during PDF ingestion."""

    def test_extract_existing_image(self, tmp_path):
        """Verify embedded markdown images are intercepted and emitted as secondary Mock Image ParsedDocuments."""
        img = tmp_path / "image1.png"
        img.write_bytes(b"fake png")

        md = f"Text davor.\n![Ein Bild]({img})\nText danach."
        cleaned, docs = _extract_image_docs(md, tmp_path, "test.pdf", "/path/test.pdf")

        assert len(docs) == 1
        assert docs[0].document_type == "image"
        assert "Ein Bild" in docs[0].content
        assert docs[0].metadata["extracted_from_pdf"] is True
        assert "[Bild: Ein Bild]" in cleaned
        assert "![" not in cleaned

    def test_missing_image_removed(self, tmp_path):
        """Verify malformed image references quietly dissolve out of text instead of breaking."""
        md = "![alt](nonexistent.png)"
        cleaned, docs = _extract_image_docs(md, tmp_path, "test.pdf", "/path/test.pdf")
        assert len(docs) == 0
        assert cleaned == ""

    def test_relative_path_resolved(self, tmp_path):
        """Verify that relative file references successfully bind against the designated image directory."""
        img = tmp_path / "page1-0.png"
        img.write_bytes(b"fake")

        md = "![](page1-0.png)"
        cleaned, docs = _extract_image_docs(md, tmp_path, "test.pdf", "/path/test.pdf")
        assert len(docs) == 1
        assert "page1 0" in docs[0].content


class TestPDFParser:
    """Test suite validating the high-level orchestration loop of the PyMuPDF4LLM wrapper."""

    def test_can_parse(self, pdf_parser):
        """Verify proper extension matching for PDFs."""
        assert pdf_parser.can_parse(Path("rules.pdf"))
        assert pdf_parser.can_parse(Path("RULES.PDF"))
        assert not pdf_parser.can_parse(Path("notes.md"))

    def test_default_config(self, pdf_parser):
        """Verify default setup initializes Tesseract/OCR routines implicitly."""
        cfg = pdf_parser._config
        assert cfg.ocr_enabled is True
        assert cfg.ocr_language == "deu"
        assert cfg.extract_images is True

    def test_custom_config(self, pdf_parser_no_ocr):
        """Verify custom configurations accurately suppress image parsing payloads."""
        cfg = pdf_parser_no_ocr._config
        assert cfg.ocr_enabled is False
        assert cfg.extract_images is False

    @patch("src.ingestion.parsers.pdf.pymupdf4llm.to_markdown")
    def test_ocr_params_passed(self, mock_to_md, pdf_parser):
        """Verify the translation driver applies deep kwargs scaling rules when OCR is enabled."""
        mock_to_md.return_value = [{"text": "# Title\nContent.", "toc_items": []}]

        pdf_parser.parse(Path("test.pdf"))

        call_kwargs = mock_to_md.call_args[1]
        assert call_kwargs["use_ocr"] is True
        assert call_kwargs["ocr_language"] == "deu"
        assert call_kwargs["ocr_dpi"] == 300
        assert call_kwargs["page_chunks"] is True
        assert call_kwargs["ignore_code"] is True

    @patch("src.ingestion.parsers.pdf.pymupdf4llm.to_markdown")
    def test_ocr_disabled_params(self, mock_to_md, pdf_parser_no_ocr):
        """Verify non-OCR configurations omit heavy layout scanning metrics."""
        mock_to_md.return_value = [{"text": "Content.", "toc_items": []}]

        pdf_parser_no_ocr.parse(Path("test.pdf"))

        call_kwargs = mock_to_md.call_args[1]
        assert call_kwargs["use_ocr"] is False
        assert "ocr_language" not in call_kwargs
        assert "write_images" not in call_kwargs

    @patch("src.ingestion.parsers.pdf.pymupdf4llm.to_markdown")
    def test_sections_split_by_headings(self, mock_to_md, pdf_parser_no_ocr):
        """Verify that the parser utilizes logical text boundaries resulting in discrete sub-documents."""
        mock_to_md.return_value = [
            {"text": "# Kapitel 1\nErster Absatz.\n## Abschnitt A\nDetails.", "toc_items": []},
        ]

        docs = pdf_parser_no_ocr.parse(Path("test.pdf"))
        text_docs = [d for d in docs if d.document_type == "pdf"]

        assert len(text_docs) >= 2
        headings = [d.heading_hierarchy for d in text_docs]
        assert ["Kapitel 1", "Abschnitt A"] in headings

    @patch("src.ingestion.parsers.pdf.pymupdf4llm.to_markdown")
    def test_toc_items_fix_heading_levels(self, mock_to_md, pdf_parser_no_ocr):
        """Verify end-to-end integration of TOC hierarchy normalizations directly replacing flat structures."""
        mock_to_md.return_value = [
            {
                "text": "## Rassen\nIntro.\n## Elfen\nSpitze Ohren.",
                "toc_items": [[1, "Rassen", 1], [2, "Elfen", 1]],
            },
        ]

        docs = pdf_parser_no_ocr.parse(Path("test.pdf"))
        text_docs = [d for d in docs if d.document_type == "pdf"]

        headings = [d.heading_hierarchy for d in text_docs]
        assert ["Rassen", "Elfen"] in headings

    @patch("src.ingestion.parsers.pdf.pymupdf4llm.to_markdown")
    def test_empty_pdf(self, mock_to_md, pdf_parser_no_ocr):
        """Verify empty PDF extraction returns a placeholder rather than failing."""
        mock_to_md.return_value = [{"text": "", "toc_items": []}]

        docs = pdf_parser_no_ocr.parse(Path("empty.pdf"))
        assert len(docs) == 1
        assert docs[0].content == "(empty PDF)"

    @patch("src.ingestion.parsers.pdf.pymupdf4llm.to_markdown")
    def test_fallback_on_ocr_error(self, mock_to_md, pdf_parser):
        """
        If OCR fails (e.g. Tesseract executable is missing), the parser must
        gracefully retry immediately with OCR deactivated rather than crashing entirely.
        """
        mock_to_md.side_effect = [
            RuntimeError("OCR engine not found"),
            [{"text": "# Fallback\nContent.", "toc_items": []}],
        ]

        docs = pdf_parser.parse(Path("test.pdf"))
        assert len(docs) >= 1
        assert any("Content" in d.content for d in docs)
        # Second call should have use_ocr=False
        assert mock_to_md.call_args[1]["use_ocr"] is False

    @patch("src.ingestion.parsers.pdf.pymupdf4llm.to_markdown")
    def test_multi_page_combined(self, mock_to_md, pdf_parser_no_ocr):
        """Verify multi-page PDF inputs reliably stitch discrete page texts together."""
        mock_to_md.return_value = [
            {"text": "# Seite 1\nErster Text.", "toc_items": []},
            {"text": "# Seite 2\nZweiter Text.", "toc_items": []},
        ]

        docs = pdf_parser_no_ocr.parse(Path("test.pdf"))
        text_docs = [d for d in docs if d.document_type == "pdf"]
        all_content = " ".join(d.content for d in text_docs)

        assert "Erster Text" in all_content
        assert "Zweiter Text" in all_content
