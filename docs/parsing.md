# Parsing

Wie LoreKeeper Text, Metadaten und Struktur aus den Dateiformaten holt, bevor Chunking und
Embedding beginnen: Obsidian-Markdown, PDF mit OCR und Bild-Metadaten. Einstellungen:
[configuration.md](configuration.md) → `ingestion`, `chunking`.

## Inhaltsverzeichnis
1. [Überblick](#überblick)
2. [Markdown (Obsidian)](#markdown-obsidian)
3. [PDF](#pdf)
4. [Bilder](#bilder)
5. [Vom Parsen zum Chunking](#vom-parsen-zum-chunking)

---

## Überblick

Drei Dokumenttypen, je ein Parser, der `ParsedDocument`-Einträge erzeugt. Alle implementieren
`BaseParser` mit `can_parse()` und `parse()`; der `IngestionOrchestrator` wählt nach Dateiendung.

```
Datei
  │
  ├── .md  → MarkdownParser   ─┐
  ├── .pdf → PDFParser         ├──→ list[ParsedDocument] → Chunker → Embeddings → ChromaDB
  └── .png/.jpg/… → ImageMetaParser ─┘
```

Welche Endungen überhaupt gelesen werden, legt `ingestion.supported_formats` fest (Standard:
`.md`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`). `.tex` wird bewusst nicht gelesen.

---

## Markdown (Obsidian)

**Datei:** `src/ingestion/parsers/markdown.py` · **Endung:** `.md`

Für Obsidian-Vaults gebaut; behandelt die Obsidian-Syntax, an der normale Markdown-Parser
scheitern.

### Was er macht

0. **Ausgelassene Abschnitte** — Abschnitte, deren Überschrift in `ingestion.skip_sections` steht
   (Groß-/Kleinschreibung und `**` egal), fallen vor allem anderen weg: von der Überschrift bis zur
   nächsten Überschrift gleicher oder höherer Ebene. Das passiert auf dem Rohtext, vor dem Bereinigen
   der Callouts — Überschriften **im** Callout (`> ## Attribute`) beenden den Abschnitt nicht; nach
   dem Bereinigen sähen sie wie normale Überschriften aus und die alten Werte kämen darunter zurück.
   Tags und Wikilinks des Abschnitts zählen nicht. Genutzt für „Alte Vault-Werte“: 59 Gegner- und
   Dämonen-Dateien bewahren die Werte von vor dem 30.09. in einem eingeklappten Callout auf
   („nicht mehr maßgeblich“); indiziert konkurrierten sie mit den aktuellen Werten (13.10.).
1. **YAML-Frontmatter** — über `python-frontmatter`. `aliases` wird gesondert gespeichert (für die
   Identitäts-Schicht im Embedding), die übrigen Schlüssel gehen als Metadaten durch.
2. **Wikilinks** — `[[Ziel]]` und `[[Ziel|Anzeigetext]]` werden erkannt. Die Ziele landen in
   `metadata.wikilinks` (für eine spätere Graph-Expansion, Roadmap 3.5); im Text steht der
   Anzeigetext bzw. der Zielname.
3. **Bild-Einbettungen** — `![[bild.png]]` und `![[bild.png|300]]` verlassen den Text, der Dateiname
   bleibt aber als `metadata.image_embeds` am Abschnitt, in dem das Bild steht (Roadmap 2, 14.10.).
   **Titelbilder** — Einbettungen vor der ersten Unterüberschrift (Text vor dem Titel und der erste
   H1-Abschnitt: Porträt, Karte) stehen für die ganze Notiz und hängen an **jedem** Abschnitt, nach dessen
   eigenen Bildern. Grund: live am 14.10. bekam keine Frage zu Malek oder Arkenfeld ein Bild, weil die
   Antwort aus „Erscheinung“ oder „Steckbrief“ kam, das Porträt aber nur am Einleitungsabschnitt hing.
   Ein Abschnitt ohne eigenen Text gibt seine Bilder an den nächsten Abschnitt weiter, Bilder am
   Dateiende gehen an den letzten. Der Orchestrator löst die Namen
   **innerhalb derselben Quelle** zu Pfaden auf (wie Obsidian: Dateiname, Groß-/Kleinschreibung egal,
   ohne `.trash`/`.obsidian`) und speichert sie als `images` — so kommen nie Bilder einer anderen,
   etwa nicht freigegebenen Quelle in eine Antwort. Die Bilddateien selbst indiziert zusätzlich der
   `ImageMetaParser`. Der Inhalts-Hash deckt nur die Notiz ab: ein neu abgelegtes Bild erscheint erst
   nach einer Änderung der Notiz oder einem vollen Re-Index der Quelle.
4. **Callouts** — `> [!abstract] Titel` verliert die Markierung, der Inhalt bleibt als normaler
   Text; Folgezeilen (`> …`) werden ausgerückt.
5. **Tags** — `#Tag` am Zeilenende wird nach `metadata.obsidian_tags` übernommen, das `#` aus dem
   Text entfernt.
6. **Aufteilung nach Überschriften** — `#` bis `######`. Jeder Abschnitt wird ein eigenes
   `ParsedDocument` mit `heading_hierarchy` (z.B. `["Charakter", "Hintergrund"]` für eine H2 unter
   einer H1).

### Ausgabe je Abschnitt

| Feld | Wert |
|------|------|
| `content` | Bereinigter Abschnittstext (ohne Wikilink-Klammern, Bild-Einbettungen, Callout-Marker) |
| `document_type` | `"markdown"` |
| `heading_hierarchy` | `["H1-Titel", "H2-Titel", …]` |
| `metadata.aliases` | Aus dem Frontmatter |
| `metadata.obsidian_tags` | Gefundene `#Tags` |
| `metadata.wikilinks` | Gefundene `[[Link]]`-Ziele |
| `metadata.image_embeds` | Eingebettete Bilder des Abschnitts (Dateinamen; im Index als `images` = Pfade) |

### Obsidian-Syntax

| Syntax | Behandlung |
|--------|------------|
| `[[Wikilink]]` | Ziel in `wikilinks`, im Text als Klartext |
| `[[Link\|Alias]]` | `Link` in `wikilinks`, im Text `Alias` |
| `![[bild.png]]` / `![[bild.png\|300]]` | Aus dem Text entfernt, Name nach `image_embeds` (Bild separat indiziert) |
| `> [!abstract] Titel` | Marker weg, Inhalt als Text |
| `#Tag` | Nach `obsidian_tags`, `#` aus dem Text |
| `aliases:` im Frontmatter | Zusätzliche Suchbegriffe |

Nicht indiziert (über `exclude_patterns`): `.obsidian/*`, `.trash/*`, `*alt.md`, `*(1).md`.

---

## PDF

**Datei:** `src/ingestion/parsers/pdf.py` · **Endung:** `.pdf` · **Einstellungen:** `ingestion.pdf`

`pymupdf4llm` wandelt die Seiten in Markdown; danach folgen OCR, Korrektur der Überschriften und
Bildextraktion, und schließlich die Aufteilung nach Überschriften (dieselbe Logik wie bei Markdown).

### Ablauf

```
PDF
  │
  ▼
pymupdf4llm.to_markdown(page_chunks=True)
  │  ├── OCR (RapidOCR) für Bereiche ohne Text
  │  ├── Tabellen → Markdown-Tabellen
  │  └── Bilder → temporärer Ordner
  │
  ▼  je Seite:
1. Überschriften über das Inhaltsverzeichnis korrigieren
  │
  ▼
2. Fallback: nummerierte Überschriften per Regex
   (z.B. "## **5.4 Mönch**" → "# **5.4 Mönch**")
  │
  ▼
3. Bildverweise auflösen
   (![alt](pfad) → [Bild: alt] + eigenes ParsedDocument)
  │
  ▼
Gesamtes Markdown → nach Überschriften teilen → list[ParsedDocument]
```

### OCR

Mit `ocr_enabled: true` (Standard) erkennt RapidOCR (`rapidocr_onnxruntime`) Text in Bereichen, die
nur aus Bild bestehen — für gemischte PDFs mit gescannten und normalen Seiten. Sprache über
`ocr_language` (`deu`), Auflösung über `ocr_dpi` (300). Scheitert OCR (Engine-Fehler, kaputte
Seite), wiederholt der Parser das ganze Dokument ohne OCR und schreibt eine Warnung ins Log.

### Überschriften-Hierarchie

`pymupdf4llm` setzt oft alle Überschriften auf dieselbe Ebene (`##`) und verliert die Struktur.
Korrektur in zwei Schritten:

1. **Inhaltsverzeichnis** (am zuverlässigsten) — hat das PDF eins, liefert `toc_items`
   `[Ebene, Titel, Seite]`; Überschriften werden per Titel zugeordnet und neu eingestuft.
2. **Regex für nummerierte Überschriften** (Fallback) — `## **5.4.5 Mönch**` wird hochgestuft, weil
   nummerierte Titel auf Struktur hindeuten, die `pymupdf4llm` übersehen hat.

Die Nummern in den Überschriften nutzt die Evaluation für die Abschnittsprüfung
(`expected_section`).

### Bildextraktion

Mit `extract_images: true` (Standard) landen eingebettete Bilder in einem temporären Ordner. Für
jedes Bild:

- entsteht ein `ParsedDocument` mit `document_type="image"`
  (Inhalt: `"Bild: {name} (aus PDF: {datei})"`),
- wird `![alt](pfad)` im Markdown durch `[Bild: {name}]` ersetzt,
- stehen `extracted_from_pdf: true` und das Bildformat in den Metadaten,
- werden die temporären Dateien nach dem Parsen gelöscht.

### Weitere Einstellungen

- `ignore_code=True` — Festbreitentext (häufig in Regelwerken und Werteblöcken) wird nicht als
  Code-Block erkannt.
- `page_chunks=True` — Seiten einzeln verarbeiten (Inhaltsverzeichnis und Bilder je Seite), dann
  zusammenführen.

### Ausgabe je Abschnitt

| Feld | Wert |
|------|------|
| `content` | Bereinigter Text mit Bild-Platzhaltern |
| `document_type` | `"pdf"` (Text) oder `"image"` (extrahierte Bilder) |
| `heading_hierarchy` | Über Inhaltsverzeichnis / Regex korrigiert |
| `metadata` | `{}` bei Text; `{image_format, extracted_from_pdf}` bei Bildern |

---

## Bilder

**Datei:** `src/ingestion/parsers/image.py` · **Endungen:** `.png`, `.jpg`, `.jpeg`, `.webp`
(der Parser kann auch `.gif`, das steht aber nicht in `supported_formats`)

Erzeugt nur minimale Einträge, damit Bilder über Dateiname und Ordner auffindbar sind. Der
Bildinhalt wird nicht analysiert (kein OCR, kein Vision-Modell).

1. Dateiname ohne Endung, `_` und `-` werden zu Leerzeichen
   (`karte_arkenfeld.png` → `"karte arkenfeld"`).
2. `heading_hierarchy` aus den Ordnern plus bereinigtem Namen.
3. Ein `ParsedDocument` mit `"Bild: karte arkenfeld (Pfad: Orte)"`.

| Feld | Wert |
|------|------|
| `content` | `"Bild: {name} (Pfad: {ordner})"` |
| `document_type` | `"image"` |
| `heading_hierarchy` | `["ordner1", "ordner2", "bereinigter name"]` |
| `metadata.image_format` | `.png`, `.jpg`, … |

**Grenzen:** Der Inhalt stammt nur aus Name und Pfad — gefunden wird nur über offensichtliche
Begriffe im Dateinamen. Bilder sind zudem aus der Suche ausgeschlossen
([embedding-strategy.md](embedding-strategy.md#warum-bilder-aus-der-suche-ausschließen)).
Bildbeschreibungen per Vision-Modell: Roadmap 2.

---

## Vom Parsen zum Chunking

Die `ParsedDocument`-Einträge gehen an `chunk_documents()`
([configuration.md](configuration.md#chunking)):

1. **heading_aware** (Standard) — jeder Abschnitt wird ein Chunk; Tabellen (Zeilen mit `|`) bleiben
   ganz, zu lange Abschnitte werden rekursiv geteilt, kleine nur innerhalb derselben Überschrift
   zusammengelegt.
2. Vor dem Embedding werden Dateiname, Aliase und Überschriftenpfad vorangestellt
   (Identitäts-Schicht) — so passt auch eine Werte-Tabelle ohne Namensnennung zu Fragen nach ihrem
   Gegenstand.
3. `document_type` bleibt erhalten und steht in den ChromaDB-Metadaten — die UI rendert damit
   Bild-Quellen per `st.image()`.
