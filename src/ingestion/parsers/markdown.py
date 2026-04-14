import re
from pathlib import Path

import frontmatter

from .base import BaseParser, ParsedDocument

# Obsidian syntax patterns
WIKILINK_PATTERN = re.compile(r"!?\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
IMAGE_EMBED_PATTERN = re.compile(r"!\[\[([^\]]+\.(png|jpg|jpeg|webp|gif))\]\]", re.IGNORECASE)
CALLOUT_PATTERN = re.compile(r"^>\s*\[!(\w+)\]\s*(.*)$", re.MULTILINE)
CALLOUT_CONTINUATION = re.compile(r"^>\s?(.*)$", re.MULTILINE)
TAG_PATTERN = re.compile(r"(?:^|\s)#([A-Za-zÀ-ÿ\w]+)(?=\s|$)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownParser(BaseParser):
    """Parser that converts Markdown documents into semantic ParsedDocument blocks.

    Includes specific logic to handle and strip Obsidian-specific syntax like
    Wikilinks (e.g. [[Link|Alias]]), Image embeds (![[image.png]]), and Callouts
    (> [!info]), ensuring that only clean text goes into the embedding vector store.
    It builds a structural hierarchy by tracking Markdown heading levels (H1-H6).
    """
    def can_parse(self, file_path: Path) -> bool:
        """True for `.md` files."""
        return file_path.suffix.lower() == ".md"

    def parse(self, file_path: Path, base_path: Path | None = None) -> list[ParsedDocument]:
        """Read the markdown file and slice it into blocks based on headings.

        Extracts YAML frontmatter (like 'aliases') and inline tags into metadata objects
        which are then assigned to every extracted slice. The text content is cleaned
        of Obsidian-specific markdown artifacts before being split.
        """
        text = file_path.read_text(encoding="utf-8")
        post = frontmatter.loads(text)

        fm_metadata = dict(post.metadata) if post.metadata else {}
        aliases = fm_metadata.pop("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        content = post.content
        tags = self._extract_tags(content)
        wikilinks = self._extract_wikilinks(content)
        content = self._clean_obsidian_syntax(content)

        sections = self._split_by_headings(content)

        source_file = file_path.name
        source_path = str(file_path.resolve())

        documents = []
        for heading_hierarchy, section_content in sections:
            section_content = section_content.strip()
            if not section_content:
                continue

            metadata = {
                **fm_metadata,
                "aliases": aliases,
                "obsidian_tags": tags,
                "wikilinks": wikilinks,
            }

            documents.append(ParsedDocument(
                content=section_content,
                source_file=source_file,
                source_path=source_path,
                document_type="markdown",
                heading_hierarchy=heading_hierarchy,
                metadata=metadata,
            ))

        if not documents:
            documents.append(ParsedDocument(
                content=content.strip() or "(empty)",
                source_file=source_file,
                source_path=source_path,
                document_type="markdown",
                heading_hierarchy=[],
                metadata={"aliases": aliases, "obsidian_tags": tags, "wikilinks": wikilinks},
            ))

        return documents

    def _extract_tags(self, text: str) -> list[str]:
        """Find inline #tags matching Obsidian tag specifications."""
        return list(set(TAG_PATTERN.findall(text)))

    def _extract_wikilinks(self, text: str) -> list[str]:
        """Extract explicit outbound [[wikilinks]] from the raw text for metadata storage."""
        links = []
        for match in WIKILINK_PATTERN.finditer(text):
            if not IMAGE_EMBED_PATTERN.match(match.group(0)):
                links.append(match.group(1).strip())
        return list(set(links))

    def _clean_obsidian_syntax(self, text: str) -> str:
        """Strip or convert Obsidian-specific UI formatting into readable plain text.

        This prevents things like `> [!abstract]` from appearing verbatim within context window text.
        """
        # Remove image embeds
        text = IMAGE_EMBED_PATTERN.sub("", text)

        # Replace wikilinks with display text
        def replace_wikilink(m):
            """Render a wikilink as its alias if present, otherwise its target name."""
            alias = m.group(2)
            return alias if alias else m.group(1)

        text = WIKILINK_PATTERN.sub(replace_wikilink, text)

        # Clean callouts: remove > [!type] marker, keep content
        text = CALLOUT_PATTERN.sub(r"\2", text)

        # Remove leading > from callout continuation lines
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if line.startswith("> "):
                cleaned.append(line[2:])
            elif line == ">":
                cleaned.append("")
            else:
                cleaned.append(line)
        text = "\n".join(cleaned)

        # Remove inline tags (keep word, remove #)
        text = re.sub(r"(?:^|\s)#([A-Za-zÀ-ÿ\w]+)", r" \1", text)

        return text.strip()

    def _split_by_headings(self, text: str) -> list[tuple[list[str], str]]:
        """Slice the cleaned text into hierarchical blocks based on header levels.

        As it iterates through lines, it tracks the current H1 -> H2 -> H3 depth.
        When a new heading is encountered, it flushes the previous block and updates the path.
        """
        sections = []
        current_hierarchy = []
        current_content_lines = []
        current_levels = []

        for line in text.split("\n"):
            heading_match = HEADING_PATTERN.match(line)
            if heading_match:
                # Save previous section
                if current_content_lines:
                    sections.append((list(current_hierarchy), "\n".join(current_content_lines)))
                    current_content_lines = []

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Adjust hierarchy based on level
                while current_levels and current_levels[-1] >= level:
                    current_levels.pop()
                    if current_hierarchy:
                        current_hierarchy.pop()

                current_levels.append(level)
                current_hierarchy.append(title)
            else:
                current_content_lines.append(line)

        # Last section
        if current_content_lines:
            sections.append((list(current_hierarchy), "\n".join(current_content_lines)))

        return sections
