from pathlib import Path

import pytest

from src.ingestion.parsers.image import ImageMetaParser
from src.ingestion.parsers.markdown import MarkdownParser


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

    def test_image_embeds_kept_per_section(self, md_parser, tmp_path):
        """Roadmap 2: the chat shows the images the note embeds — per section, also '![[x.png|300]]'."""
        f = tmp_path / "Malek.md"
        f.write_text("# Malek\n## Hintergrund\nEin Soldat.\n![[Karte.png|300]]\n## Ziele\nMacht.",
                     encoding="utf-8")

        docs = {d.heading_hierarchy[-1]: d for d in md_parser.parse(f)}
        assert docs["Hintergrund"].metadata["image_embeds"] == ["Karte.png"]
        assert "300" not in docs["Hintergrund"].content          # the size part is no text either
        assert "image_embeds" not in docs["Ziele"].metadata

    def test_image_of_a_section_without_text_moves_to_the_next_one(self, md_parser, tmp_path):
        """The portrait right under the title has no text of its own — it belongs to what follows."""
        f = tmp_path / "Malek.md"
        f.write_text("# Malek\n![[Malek_1.png]]\n## Hintergrund\nEin Soldat.\n## Galerie\n![[Malek_2.png]]",
                     encoding="utf-8")

        docs = md_parser.parse(f)
        assert [d.heading_hierarchy[-1] for d in docs] == ["Hintergrund"]
        assert docs[0].metadata["image_embeds"] == ["Malek_1.png", "Malek_2.png"]  # trailing one too

    def test_lead_image_under_the_title_goes_to_every_section(self, md_parser, tmp_path):
        """The portrait/map above the first sub-heading stands for the whole note; a section's own
        images come first. Embeds under a second H1 are that topic's, not the note's."""
        f = tmp_path / "Malek.md"
        f.write_text("![[Banner.png]]\nVorwort.\n# Malek\n![[Malek_1.png]]\nDer Monarch.\n"
                     "## Rolle\nAnführer.\n![[Malek_2.png]]\n## Ziele\nMacht.\n# Anhang\n![[Skizze.png]]\nNotizen.",
                     encoding="utf-8")

        docs = {(d.heading_hierarchy or ["—"])[-1]: d.metadata.get("image_embeds") for d in md_parser.parse(f)}
        lead = ["Banner.png", "Malek_1.png"]
        assert docs["—"] == lead and docs["Malek"] == ["Malek_1.png", "Banner.png"]
        assert docs["Rolle"] == ["Malek_2.png", *lead]
        assert docs["Ziele"] == lead
        assert docs["Anhang"] == ["Skizze.png", *lead]

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

    def test_hash_lines_in_code_block_are_not_headings(self, md_parser, tmp_path):
        """Verify '# comment' lines inside fenced code blocks don't break the heading path."""
        f = tmp_path / "test.md"
        f.write_text(
            "# Würfel\n## Makros\n```python\n# kein Heading\nroll(2)\n```\nNach dem Code.",
            encoding="utf-8",
        )

        docs = md_parser.parse(f)
        assert [d.heading_hierarchy for d in docs] == [["Würfel", "Makros"]]
        assert "# kein Heading" in docs[0].content
        assert "Nach dem Code." in docs[0].content

    def test_empty_file(self, md_parser, tmp_path):
        """Verify that absolutely blank files still generate an empty document stub."""
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")

        docs = md_parser.parse(f)
        assert len(docs) == 1  # Fallback document


_VAULT_ENEMY = """# Grabeswolf

## Kampfwerte

LP 40, Biss 2W6 #wolf

## Alte Vault-Werte

> [!note]- Bisherige Werte aus dem Vault (nicht mehr maßgeblich)
> ## Attribute
>
> LP 25, Biss 1W6 [[Altes Rudel]] #veraltet
> ### Fähigkeiten
> Heulen

## Taktik

Jagt im Rudel.
"""


class TestSkipSections:
    """``ingestion.skip_sections``: whole sections left out of the index (outdated stat blocks in the vault)."""

    def _parse(self, tmp_path, text, skip):
        f = tmp_path / "Grabeswolf.md"
        f.write_text(text, encoding="utf-8")
        return MarkdownParser(skip_sections=skip).parse(f)

    def test_skipped_section_and_its_callout_headings_are_gone(self, tmp_path):
        """Headings inside the callout ('> ## Attribute') must not end the skipped section — after
        callout cleaning they would look like ordinary headings and bring the old values back."""
        docs = self._parse(tmp_path, _VAULT_ENEMY, ["Alte Vault-Werte"])

        text = "\n".join(d.content for d in docs)
        assert "LP 25" not in text and "Heulen" not in text
        assert [d.heading_hierarchy for d in docs] == [["Grabeswolf", "Kampfwerte"], ["Grabeswolf", "Taktik"]]

    def test_tags_and_wikilinks_of_the_skipped_section_are_dropped(self, tmp_path):
        docs = self._parse(tmp_path, _VAULT_ENEMY, ["Alte Vault-Werte"])

        assert docs[0].metadata["obsidian_tags"] == ["wolf"]
        assert docs[0].metadata["wikilinks"] == []

    def test_title_match_ignores_case_and_bold(self, tmp_path):
        text = _VAULT_ENEMY.replace("## Alte Vault-Werte", "## **alte vault-werte**")
        docs = self._parse(tmp_path, text, ["Alte Vault-Werte"])

        assert "LP 25" not in "\n".join(d.content for d in docs)

    def test_skipped_section_ends_at_a_higher_heading(self, tmp_path):
        """A deeper skipped section ends at the next heading of its level or above."""
        text = "# A\n\n## Werte\n\n### Alte Vault-Werte\n\nalt\n\n### Neu\n\nneu\n\n# B\n\nb\n"
        docs = self._parse(tmp_path, text, ["Alte Vault-Werte"])

        assert [d.content for d in docs] == ["neu", "b"]

    def test_without_skip_list_everything_stays(self, tmp_path):
        docs = self._parse(tmp_path, _VAULT_ENEMY, [])

        assert "LP 25" in "\n".join(d.content for d in docs)


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
