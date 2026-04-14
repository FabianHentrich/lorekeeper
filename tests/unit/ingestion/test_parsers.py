from pathlib import Path

import pytest

from src.ingestion.parsers.markdown import MarkdownParser
from src.ingestion.parsers.image import ImageMetaParser


@pytest.fixture
def md_parser():
    """Provides a fresh instance of MarkdownParser for testing."""
    return MarkdownParser()


@pytest.fixture
def img_parser():
    """Provides a fresh instance of ImageMetaParser for testing."""
    return ImageMetaParser()


class TestMarkdownParser:
    """Test suite validating standard Markdown rules and Obsidian-specific extensions."""

    def test_can_parse(self, md_parser):
        """Verify that the parser correctly binds exclusively to .md files."""
        assert md_parser.can_parse(Path("test.md"))
        assert not md_parser.can_parse(Path("test.txt"))
        assert not md_parser.can_parse(Path("test.pdf"))

    def test_basic_parsing(self, md_parser, tmp_path):
        """Verify that pure text and basic headers resolve into ParsedDocument instances."""
        f = tmp_path / "test.md"
        f.write_text("# Title\nSome content here.", encoding="utf-8")

        docs = md_parser.parse(f)
        assert len(docs) >= 1
        assert docs[0].document_type == "markdown"
        assert "Some content here" in docs[0].content

    def test_frontmatter_aliases(self, md_parser, tmp_path):
        """Verify that YAML frontmatter blocks correctly expose alias metadata tags."""
        f = tmp_path / "npc.md"
        f.write_text("---\naliases: [Malek, Der Monarch]\n---\n# NPC\nContent.", encoding="utf-8")

        docs = md_parser.parse(f)
        assert docs[0].metadata["aliases"] == ["Malek", "Der Monarch"]

    def test_wikilinks_extracted(self, md_parser, tmp_path):
        """Verify that double bracket syntax correctly isolates targets into metadata arrays."""
        f = tmp_path / "test.md"
        f.write_text("# Test\nEr kennt den [[Schwarzer Monolith]] und den [[Schattenorden]].", encoding="utf-8")

        docs = md_parser.parse(f)
        wikilinks = docs[0].metadata["wikilinks"]
        assert "Schwarzer Monolith" in wikilinks
        assert "Schattenorden" in wikilinks

    def test_wikilink_alias(self, md_parser, tmp_path):
        """Verify that aliased wikilinks keep only the alias literal in unstructured text."""
        f = tmp_path / "test.md"
        f.write_text("# Test\nAus den [[Nordlande|Nordlanden]].", encoding="utf-8")

        docs = md_parser.parse(f)
        assert "Nordlanden" in docs[0].content
        assert "[[" not in docs[0].content

    def test_image_embeds_removed(self, md_parser, tmp_path):
        """Verify that embedded exclamation-prefixed links are expunged from the plain token sequence."""
        f = tmp_path / "test.md"
        f.write_text("# Test\nText vor Bild.\n![[portrait.png]]\nText nach Bild.", encoding="utf-8")

        docs = md_parser.parse(f)
        full_text = " ".join(d.content for d in docs)
        assert "portrait.png" not in full_text
        assert "Text vor Bild" in full_text

    def test_callouts_cleaned(self, md_parser, tmp_path):
        """Verify that Obsidian blockquote structures are stripped leaving plain text payloads."""
        f = tmp_path / "test.md"
        f.write_text("# Test\n> [!abstract] Zusammenfassung\n> Wichtiger Inhalt hier.", encoding="utf-8")

        docs = md_parser.parse(f)
        full_text = " ".join(d.content for d in docs)
        assert "Wichtiger Inhalt hier" in full_text
        assert "[!abstract]" not in full_text

    def test_tags_extracted(self, md_parser, tmp_path):
        """Verify that trailing `#` prefixed terms map into the `obsidian_tags` metadata store."""
        f = tmp_path / "test.md"
        f.write_text("# Test\nEin Charakter. #NPC #DunkleMagie", encoding="utf-8")

        docs = md_parser.parse(f)
        tags = docs[0].metadata["obsidian_tags"]
        assert "NPC" in tags
        assert "DunkleMagie" in tags

    def test_heading_hierarchy(self, md_parser, tmp_path):
        """Verify that nested headers dynamically enrich the hierarchical metadata trails."""
        f = tmp_path / "test.md"
        f.write_text("# Charakter\n## Hintergrund\nText hier.\n## Ausrüstung\nSchwert.", encoding="utf-8")

        docs = md_parser.parse(f)
        hierarchies = [d.heading_hierarchy for d in docs]
        assert ["Charakter", "Hintergrund"] in hierarchies
        assert ["Charakter", "Ausrüstung"] in hierarchies

    def test_empty_file(self, md_parser, tmp_path):
        """Verify that absolutely blank files still generate an empty document stub."""
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")

        docs = md_parser.parse(f)
        assert len(docs) == 1  # Fallback document


class TestImageMetaParser:
    """Test suite validating metadata extraction constraints targeting binary image assets."""

    def test_can_parse(self, img_parser):
        """Verify that typical web/image extensions successfully map to this handler."""
        assert img_parser.can_parse(Path("test.png"))
        assert img_parser.can_parse(Path("test.jpg"))
        assert img_parser.can_parse(Path("test.webp"))
        assert not img_parser.can_parse(Path("test.md"))

    def test_parse_image(self, img_parser, tmp_path):
        """Verify that filesystem paths seamlessly yield a structural textual stub representing the binary."""
        img = tmp_path / "NPCs" / "aldric_portrait.png"
        img.parent.mkdir()
        img.write_bytes(b"fake image data")

        docs = img_parser.parse(img)
        assert len(docs) == 1
        assert docs[0].document_type == "image"
        assert "aldric portrait" in docs[0].content
        assert "NPCs" in docs[0].content
