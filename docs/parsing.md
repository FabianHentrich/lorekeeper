# Parsing Pipeline

LoreKeeper supports three document types: Markdown (Obsidian), PDF, and images.
Each has a dedicated parser that produces `ParsedDocument` entries, which then
flow into the [chunking pipeline](../ARCHITECTURE.md) and vector store.

All parsers implement `BaseParser` with `can_parse()` and `parse()`.
The `IngestionOrchestrator` selects the right parser based on file extension.

```
File on disk
  │
  ├── .md  → MarkdownParser   ─┐
  ├── .pdf → PDFParser         ├──→ list[ParsedDocument] → Chunker → Embeddings → ChromaDB
  └── .png/.jpg/… → ImageMetaParser ─┘
```

---

## Markdown (Obsidian)

**File:** `src/ingestion/parsers/markdown.py`
**Extensions:** `.md`

The Markdown parser is designed for Obsidian vaults and handles all
Obsidian-specific syntax that standard Markdown parsers would ignore or
break on.

### What it does

1. **YAML frontmatter** — extracted via `python-frontmatter`. The `aliases`
   field is stored separately (used for identity-layer embedding); all other
   frontmatter keys are passed through as metadata.

2. **Wikilinks** — `[[Target]]` and `[[Target|Display Text]]` are detected.
   Targets are stored in `metadata.wikilinks` for potential graph expansion
   during retrieval. In the content text, wikilinks are replaced with their
   display text (alias if present, otherwise the target name).

3. **Image embeds** — `![[image.png]]` references are removed from the text
   entirely. Images are indexed separately by the `ImageMetaParser`.

4. **Callouts** — `> [!abstract] Title` markers are stripped, but the callout
   content is kept as plain text. Continuation lines (`> ...`) are unindented.

5. **Tags** — `#TagName` at the end of lines is extracted into
   `metadata.obsidian_tags`. The `#` prefix is removed from the content text.

6. **Heading-based splitting** — the document is split into sections by
   Markdown headings (`#` through `######`). Each section becomes a separate
   `ParsedDocument` with a `heading_hierarchy` list reflecting its nesting
   (e.g. `["Charakter", "Hintergrund"]` for an H2 under an H1).

### Output per section

| Field | Value |
|-------|-------|
| `content` | Cleaned section text (no wikilink brackets, no image embeds, no callout markers) |
| `document_type` | `"markdown"` |
| `heading_hierarchy` | `["H1 title", "H2 title", ...]` |
| `metadata.aliases` | From frontmatter `aliases` field |
| `metadata.obsidian_tags` | Extracted `#Tag` values |
| `metadata.wikilinks` | Extracted `[[link]]` targets |

### Obsidian syntax reference

| Syntax | Handling |
|--------|----------|
| `[[Wikilink]]` | Store target in `wikilinks`, replace with plain text |
| `[[Link\|Alias]]` | Store `Link` in `wikilinks`, display `Alias` in text |
| `![[image.png]]` | Remove from text (indexed separately by ImageMetaParser) |
| `> [!abstract] Title` | Remove callout marker, keep content as plain text |
| `#Tag` | Extract to `obsidian_tags`, remove `#` from text |
| `aliases:` in YAML frontmatter | Store as additional search terms in metadata |

---

## PDF

**File:** `src/ingestion/parsers/pdf.py`
**Extensions:** `.pdf`
**Config:** `ingestion.pdf` in `settings.yaml` (see [configuration.md](configuration.md))

The PDF parser uses `pymupdf4llm` to convert PDF pages to Markdown, then
applies OCR, heading correction, and image extraction before splitting
by headings (reusing the Markdown parser's splitting logic).

### Pipeline

```
PDF file
  │
  ▼
pymupdf4llm.to_markdown(page_chunks=True)
  │  ├── OCR (RapidOCR) on text-less regions
  │  ├── Table detection → Markdown tables
  │  └── Image extraction → temp directory
  │
  ▼  per page:
1. TOC-based heading correction
   (PDF table-of-contents → correct heading levels)
  │
  ▼
2. Numbered-heading regex fallback
   (e.g. "## **5.4 Mönch**" → "# **5.4 Mönch**")
  │
  ▼
3. Image reference extraction
   (![alt](path) → [Bild: alt] + ParsedDocument)
  │
  ▼
Combined Markdown → split by headings → list[ParsedDocument]
```

### OCR

When `ocr_enabled: true` (default), pymupdf4llm uses RapidOCR
(`rapidocr_onnxruntime`) to recognize text in image-only regions of the PDF.
This handles mixed PDFs where some pages are scanned and others have
extractable text.

- **Language:** `ocr_language` (default `"deu"` for German, ISO 639-3 codes)
- **Resolution:** `ocr_dpi` (default `300`)
- **Fallback:** If OCR fails (engine error, corrupt page), the parser
  automatically retries the entire document without OCR and logs a warning.

### Heading hierarchy

pymupdf4llm often renders all headings at the same level (`##`), losing the
document's real structure. The parser corrects this in two passes:

1. **TOC data** (most reliable) — if the PDF has a table of contents,
   `toc_items` provides `[level, title, page]` entries. Headings in the
   Markdown are matched by title and re-levelled accordingly.

2. **Numbered-heading regex** (fallback) — headings like `## **5.4.5 Mönch**`
   are promoted to `#`, since numbered section titles indicate top-level
   structure that pymupdf4llm missed.

### Image extraction

When `extract_images: true` (default), embedded images are written to a
temporary directory during parsing. For each extracted image:

- A `ParsedDocument` with `document_type="image"` is created
  (content: `"Bild: {name} (aus PDF: {filename})"`)
- The `![alt](path)` reference in the Markdown is replaced with
  `[Bild: {name}]` as a text placeholder
- Metadata includes `extracted_from_pdf: true` and the image format
- Temporary image files are cleaned up automatically after parsing

### Other settings

- `ignore_code=True` — prevents monospace text (common in rulebooks,
  stat blocks) from being misidentified as code blocks
- `page_chunks=True` — processes pages individually for per-page TOC data
  and image extraction, then combines the results

### Output per section

| Field | Value |
|-------|-------|
| `content` | Cleaned section text with image placeholders |
| `document_type` | `"pdf"` (text sections) or `"image"` (extracted images) |
| `heading_hierarchy` | Corrected via TOC / numbered-heading regex |
| `metadata` | `{}` for text sections; `{image_format, extracted_from_pdf}` for images |

---

## Images

**File:** `src/ingestion/parsers/image.py`
**Extensions:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`

The image parser creates minimal metadata entries so that images are
findable by filename and folder path. It does not analyze image content
(no OCR, no vision model).

### What it does

1. Extracts the filename stem, replacing `_` and `-` with spaces
   (e.g. `karte_arkenfeld.png` → `"karte arkenfeld"`)
2. Builds a `heading_hierarchy` from the folder path segments plus the
   cleaned filename
3. Creates a single `ParsedDocument` with a descriptive content string:
   `"Bild: karte arkenfeld (Pfad: Orte)"`

### Output

| Field | Value |
|-------|-------|
| `content` | `"Bild: {name} (Pfad: {folder})"` |
| `document_type` | `"image"` |
| `heading_hierarchy` | `["folder1", "folder2", "cleaned filename"]` |
| `metadata.image_format` | `.png`, `.jpg`, etc. |

### Limitations

- Content is derived only from filename and path — the image itself is not
  analyzed. This means retrieval only matches on obvious filename terms.
- For richer image indexing (e.g. vision-model descriptions), see the
  roadmap in `plan roadmap.md`.

---

## How parsing flows into chunking

After parsing, `ParsedDocument` entries are passed to `chunk_documents()`
(see [configuration.md — chunking](configuration.md#chunking)):

1. **heading_aware** (default) — each `ParsedDocument` section becomes a
   chunk. Tables (lines starting with `|`) are kept atomic. Oversized prose
   sections are split recursively. Small chunks are merged only within the
   same heading boundary.

2. The heading hierarchy and filename stem are prepended to each chunk's
   content before embedding (identity layer), so a stat table with no
   self-reference still matches queries about its subject.

3. `document_type` is preserved through chunking and stored in ChromaDB
   metadata — the UI uses it to render image sources via `st.image()`.
