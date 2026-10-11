# LoreKeeper — Roadmap & Bearbeitungsliste

Stand: 2026-04-14

---

## 1. PDF-Parsing verbessern (OCR + Struktur) — ✅ Erledigt

### Was wurde umgesetzt

Der `PDFParser` nutzt jetzt die vollen Fähigkeiten von pymupdf4llm statt nur die Defaults:

- **OCR**: `rapidocr_onnxruntime` als Backend, pymupdf4llm erkennt es automatisch. Deutsche Sprache (`deu`) als Default, 300 DPI. Automatischer Fallback ohne OCR wenn das Backend einen Fehler wirft.
- **Bildextraktion**: Eingebettete Bilder werden in ein temporäres Verzeichnis extrahiert und als eigene `ParsedDocument`-Einträge (`document_type="image"`) in die Pipeline eingespeist. Bild-Referenzen im Markdown werden durch beschreibende Platzhalter ersetzt.
- **Heading-Hierarchie**: Zweistufige Korrektur — zuerst TOC-Daten des PDFs (zuverlässigste Quelle), dann Numbered-Heading-Regex als Fallback. `page_chunks=True` liefert strukturierte Daten pro Seite.
- **Layout**: `ignore_code=True` verhindert, dass monospace-Text in Regelwerken fälschlich als Code-Block erkannt wird.

### Konfiguration

Neuer Block in `ingestion.pdf` (alle Werte haben Defaults):

```yaml
ingestion:
  pdf:
    ocr_enabled: true        # OCR für Text-lose Regionen
    ocr_language: "deu"      # Tesseract-Sprachcode
    ocr_dpi: 300             # DPI für OCR-Rendering
    extract_images: true     # Eingebettete Bilder extrahieren
    image_format: "png"      # Bildformat bei Extraktion
```

### Geänderte Dateien

| Datei | Änderung |
|-------|----------|
| `requirements.txt` | `rapidocr_onnxruntime>=1.4.0` hinzugefügt |
| `src/config/manager.py` | `PdfConfig` Pydantic-Model |
| `src/ingestion/parsers/pdf.py` | OCR, Bilder, TOC-Headings, Fallback |
| `src/ingestion/orchestrator.py` | Reicht `PdfConfig` an Parser |
| `tests/unit/ingestion/test_pdf_parser.py` | 20 neue Tests |

### Offene Verbesserungen (optional, spätere Iteration)

- **Docling als Alternative**: Für Fälle wo pymupdf4llm's Layout-Erkennung nicht reicht (komplexe mehrspaltige Layouts, verschachtelte Tabellen). Erfordert PyTorch als Dependency.
- **Vision-LLM für Bild-Beschreibungen**: Extrahierte PDF-Bilder aktuell nur mit Dateiname indiziert. Gemini Vision könnte semantische Beschreibungen generieren (siehe Punkt 2).

---

## 2. Bilder im Chat anzeigen (Image Retrieval)

### Problem

- Bilder werden aktuell nur als **Metadaten** indiziert (`Bild: name (Pfad: ordner)`) — der Inhalt des Bildes wird nicht erfasst
- Im Chat werden Bilder als Quelle angezeigt (`st.image()`), aber **nur wenn sie als Source-Referenz vom Retriever kommen** — der Retriever kann sie aber kaum finden, weil der indizierte Text fast leer ist
- **Ziel**: User fragt z.B. "Wie sieht die Karte von Arkenfeld aus?" → LoreKeeper zeigt das Bild inline im Chat

### Teilprobleme

#### 2a. Bild-Inhalte sinnvoll indizieren

Aktuell ist der indizierte Content für ein Bild nur `"Bild: karte arkenfeld (Pfad: Orte)"` — das matcht kaum semantisch.

**Optionen:**

| Ansatz | Vorteil | Nachteil |
|--------|---------|----------|
| **Vision-LLM Beschreibung** (Gemini Vision / LLaVA) | Reichhaltige semantische Beschreibung, matcht auf natürliche Fragen | API-Kosten (Gemini) oder lokales Modell (LLaVA ~5 GB VRAM) |
| **CLIP-Embeddings** | Direkte Bild→Vector-Suche ohne Umweg über Text | Separater Embedding-Space, nicht kompatibel mit E5-Text-Embeddings, eigene Collection nötig |
| **Manuelle Captions** (Obsidian `alt`-Text / Frontmatter) | Kein Modell nötig, User hat volle Kontrolle | Mühsam bei vielen Bildern, oft nicht vorhanden |
| **Dateiname + Ordner + Obsidian-Context** | Kein Extra-Modell, nutzt vorhandene Infos | Sehr begrenzt, matcht nur auf offensichtliche Begriffe |

**Empfehlung**: Vision-LLM-Beschreibung bei Ingestion:
- Gemini Flash ist günstig (~0.01 ct/Bild) und liefert gute Beschreibungen
- Beschreibung wird als `content` des Bild-Chunks gespeichert
- Fallback: Dateiname + Ordner wenn kein Vision-Modell konfiguriert

#### 2b. Bilder in der Chat-Antwort anzeigen

Aktuell zeigt die UI Bilder nur im Source-Expander. Für echtes "Bild-im-Chat" braucht es:

1. **Backend**: Bild-Quellen als spezielle Referenz im Response markieren (schon vorhanden: `document_type: "image"`)
2. **Prompt-Anpassung**: System-Prompt muss dem LLM sagen, dass es Bild-Referenzen einbetten kann, z.B. `[Bild: Karte_Arkenfeld.png]`
3. **UI-Rendering**: Chat-Nachricht parsen, `[Bild: ...]` durch `st.image()` ersetzen
4. **Bild-Serving**: Bilder über Backend-Endpoint ausliefern (`GET /files/{path}`) statt lokaler Pfade — damit es auch im Docker/Remote-Setup funktioniert

### Implementierungsplan

**Phase 1 — Bessere Indizierung:**
1. Optionalen Vision-Beschreibungs-Schritt in `ImageMetaParser` einbauen (Gemini Vision oder lokales Modell)
2. Config-Key: `ingestion.image_description: none | gemini | local`
3. Beschreibung als Content speichern, Originalpfad als Metadatum beibehalten
4. Re-Ingest der Bilder

**Phase 2 — Inline-Anzeige:**
1. `GET /files/{path}` Endpoint für Bild-Serving (Base64 oder FileResponse)
2. System-Prompt erweitern: LLM soll `![Bildname](image:dateiname.png)` in Antworten nutzen wenn ein Bild-Chunk relevant ist
3. UI: Markdown-Post-Processing — `image:` URLs durch `st.image()` oder echte Backend-URLs ersetzen
4. Fallback: Bilder weiterhin im Source-Expander zeigen wenn Inline-Rendering fehlschlägt

### Aufwand

Phase 1: Mittel (Vision-API-Call bei Ingestion + Config)
Phase 2: Mittel-Hoch (Endpoint, Prompt-Engineering, UI-Rendering, Edge Cases)

---

## 3. Weitere Feature-Vorschläge

### 3.1 Hybrid Search (BM25 + Vector) — ✅ Erledigt

Umgesetzt mit `rank_bm25` als In-Memory-Index, lazy aus der ChromaDB-Collection gebaut und nach Ingestion invalidiert. Vektor- und BM25-Suche laufen parallel, Fusion via Reciprocal Rank Fusion (`hybrid.bm25_weight` steuert den Anteil), danach wie gehabt Cross-Encoder-Rerank.

**Wichtig**: `score_threshold` greift auf die Cosine-Scores **vor** der Fusion — RRF-Scores liegen auf einer anderen Skala (max ≈ 1/61) und würden vom Cosine-Threshold komplett gefiltert. Dieser Bug wurde im Review gefunden und gefixt.

**Neu**:
- `retrieval.hybrid` Config-Block (`enabled`, `bm25_weight`, `bm25_top_k`) — Default `enabled: false`
- Per-Request Override via `hybrid_search` auf `QueryRequest`
- Streamlit-Sidebar-Toggle für Chat-Queries
- Evaluation-Seite: Radio-Selector (Config / Vektor only / Hybrid) + A/B-Button für Golden-Set-Vergleich
- Eval-CLI-Flags `--hybrid` / `--no-hybrid`
- Golden Set um 6 Keyword-fokussierte Pairs (`k001`–`k006`) für differenzielle A/B-Messung erweitert

### 3.2 Multi-Session / Session-History

**Problem**: Sessions sind flüchtig (RAM) und gehen bei Neustart verloren. Es gibt keine Möglichkeit, vergangene Gespräche wieder zu laden.

**Lösung**:
- Sessions in SQLite oder JSON persistieren
- UI: Session-Liste in der Sidebar ("Letzte Gespräche"), klickbar zum Wiederladen
- Export als Markdown

**Aufwand**: Mittel.

### 3.3 Chunk-Inspector / Debug-Ansicht

**Problem**: Wenn der Retriever falsche Chunks liefert, ist unklar warum. Man muss in ChromaDB schauen.

**Lösung**: Admin-Seite die pro Dokument die gespeicherten Chunks anzeigt — Content, Embedding-Dimension, Metadaten, Score bei Test-Query. Hilft beim Tuning.

**Aufwand**: Gering-Mittel. `GET /admin/chunks?source=...` Endpoint + UI-Seite.

### 3.4 Antwort-Feedback (Thumbs Up/Down)

**Problem**: Keine Möglichkeit, Antwortqualität systematisch zu tracken.

**Lösung**: Thumbs-up/down unter jeder Antwort, gespeichert mit Frage + Antwort + Quellen. Basis für späteres Prompt-Tuning und Evaluation.

**Aufwand**: Gering. Button + JSON-Log oder SQLite.

### 3.5 Wikilink-Auflösung im Retrieval

**Problem**: Obsidian-Wikilinks (`[[Arkenfeld]]`) werden als Metadaten gespeichert, aber nicht für Retrieval genutzt. Wenn ein Chunk auf `[[Malek]]` verweist, könnte der Retriever automatisch den Malek-Chunk mitliefern.

**Lösung**: Nach dem initialen Retrieval: Wikilinks der Top-Chunks extrahieren, referenzierte Chunks nachladen (Graph-Expansion). Gibt dem LLM mehr Kontext bei vernetzten Themen.

**Aufwand**: Mittel. Wikilink→Chunk-Mapping beim Ingest bauen, Retriever erweitern.


### 3.6 Chat-Verlauf im Antwort-Prompt

**Problem**: Das LLM sieht bei der Antwort nur die kondensierte Frage, nicht den bisherigen Verlauf. Folgefragen, die sich auf die *vorherige Antwort* beziehen ("fass das kürzer zusammen", "was hast du eben über seine Schwester gesagt?"), funktionieren deshalb nicht — der Condense-Schritt löst nur Referenzen für das Retrieval auf.

**Lösung**: Die letzten 1–2 Frage/Antwort-Paare (begrenzt durch `conversation.max_context_tokens`) zusätzlich in den QA-Prompt geben. `qa`-Template in `prompts.yaml` um einen optionalen `history`-Block erweitern, `render_qa()` reicht ihn durch.

**Aufwand**: Gering-Mittel. Achtung auf Kontextlänge bei kleinen Modellen ("Lost in the Middle").

### 3.7 Authentifizierung für die API

**Problem**: Die API hat keinerlei Auth. Lokal unkritisch, aber im Docker-Setup bindet sie auf `0.0.0.0` — dann kann jeder im Netz `/sources/scan` (listet beliebige Pfade auf dem Host), `/admin/wipe` (löscht die Collection), `PUT /config` und `POST /provider/gemini/key` aufrufen.

**Lösung**: Einfacher API-Key aus `.env` (`LOREKEEPER_API_KEY`), geprüft per FastAPI-Dependency auf allen schreibenden/Admin-Routen; die Streamlit-UI schickt ihn als Header mit. Ohne gesetzten Key bleibt das Verhalten wie bisher (lokaler Betrieb). Zusätzlich `/sources/scan` auf erlaubte Basisverzeichnisse beschränken.

**Aufwand**: Gering-Mittel.

### 3.8 Laufenden Ingest-Job wieder aufnehmen

**Problem**: Startet man eine Indizierung, während schon eine läuft, antwortet das Backend mit 409 („Eine Indexierung läuft bereits") — die UI zeigt nur diese Warnung. Den Fortschritt des laufenden Jobs sieht man nicht mehr, z.B. nachdem ein Widget-Klick den Poll-Loop abgebrochen hat.

**Lösung**: `_start_ingest_job` gibt im 409-Body die `job_id` des laufenden Jobs mit; `_start_ingest` in `ui/pages/1_Sources.py` pollt dann diesen Job statt nur zu warnen. Alternativ `GET /ingest/active`.

**Aufwand**: Gering.

### 3.9 Nicht-blockierender Ingest-Fortschritt in der UI

**Problem**: `_poll_ingest_job` pollt synchron in einer `while`-Schleife. Solange ein Job läuft (Minuten), rendert der Rest der Sources-Seite nicht (bei Reindex pro Source fehlen alle Elemente unterhalb des Expanders), und jede Interaktion bricht die Fortschrittsanzeige ab — der Job läuft im Backend weiter.

**Lösung**: `job_id` in `st.session_state` halten und den Fortschritt in einem `st.fragment(run_every=...)` aktualisieren. Kombiniert mit 3.8 überlebt die Anzeige auch Seitenwechsel.

**Aufwand**: Gering-Mittel.

### 3.10 Regelwerk-Slot im Retrieval (Vault ↔ Regelwerk)

**Problem**: Fragen, die eine Vault-Entität *und* einen Regelbegriff nennen („Was bewirkt Ritterlichkeit bei Escanor?“, „Was kann Thalions Phönixfeder?“), liefern nur Chunks aus der NPC-Datei. Die Regeldefinition im PDF erreicht die Top-8 nicht — die NPC-Datei belegt alle Plätze bis zum `max_per_source`-Cap, der Rest geht an andere Vault-Dateien. Messbar im Golden Set: x002–x008 sind Abschnitt-Misses (`expected_section`), obwohl der Datei-Hit stimmt.

**Lösung**: Nach dem Rerank mindestens einen Platz für die Gruppe `rules` reservieren, wenn ein Regel-Chunk unter den Kandidaten ist und einen Mindest-Rerank-Score erreicht (analog zum bestehenden Backfill im Soft-Cap). Optional nur auslösen, wenn die Frage einen Begriff enthält, der als Abschnittsüberschrift im Regelwerk existiert (Eigenheit, Zauber, Artefakt). Danach Golden Set mit Abschnittsprüfung laufen lassen; Ziel: x002–x008 treffen, ohne die übrigen 122/137 zu verschlechtern.

**Aufwand**: Mittel.

**Restliche Abschnitt-Misses (separat ansehen)**: Wortähnlichkeit zieht falsche Dateien („Armbrust“ → Bargrun der Armdrücker, „Verbände“ → Balancing-Notiz) und Nachbarabschnitte gewinnen (Erlernen → 5.3 statt 5.1, Vorteil/Nachteil → 1.1 statt 1.4).

## 4.0 Zugriff auf Lorekeeper aus dem Netzwerk und dem Internet (falls das machbar ist)

**Problem**: mein Laptop ist nicht stark genung um ein LLM laufen zu lassen. deshalb will ich meinen gaming PC laufen lassen und die UI am laptop abrufen. und dann mindetsens aus dem selben netzwerk aber potenziell auch aus dem internet wenn wir nicht zu hause sind. alternative dazu wäre nur den gemini key bezahlvariante zu nehmen. da kann man aber meines wissens nach keine Budget einstellen sodass da das finanzielle risiko steht wenn was schief geht.

**Lösung**: 

**Aufwand**: 

---

## Priorisierungsvorschlag

| # | Feature | Impact | Aufwand | Empfehlung |
|---|---------|--------|---------|------------|
| 1 | ~~PDF-Parsing (OCR + Struktur)~~ | ~~Hoch~~ | ~~Mittel~~ | ✅ Erledigt |
| 2 | Bilder im Chat (Phase 1: Beschreibung) | Mittel — Karten/Handouts suchbar machen | Mittel | Danach |
| 3 | ~~Hybrid Search (BM25)~~ | ~~Hoch~~ | ~~Mittel~~ | ✅ Erledigt |
| 4 | Bilder im Chat (Phase 2: Inline) | Nice-to-have | Mittel-Hoch | Später |
| 5 | Antwort-Feedback | Gering-Mittel | Gering | Quick Win |
| 6 | Session-History | Mittel | Mittel | Wenn Zeit |
| 7 | Chunk-Inspector | Gering (Dev-Tool) | Gering-Mittel | Wenn nötig |
| 8 | Wikilink-Expansion | Mittel | Mittel | Experimentell |
| 9 | Chat-Verlauf im Antwort-Prompt | Mittel — bessere Folgefragen | Gering-Mittel | Nach Bugfix-Phase |
| 10 | API-Authentifizierung | Mittel — nur relevant bei Netzwerk-/Docker-Betrieb | Gering-Mittel | Vor Remote-Deployment |
| 11 | Laufenden Ingest-Job wieder aufnehmen | Gering-Mittel | Gering | Nach Bugfix-Phase |
| 12 | Nicht-blockierender Ingest-Fortschritt | Mittel — Sources-Seite bleibt bedienbar | Gering-Mittel | Nach Bugfix-Phase, zusammen mit 11 |
| 13 | Regelwerk-Slot im Retrieval | Mittel — Vault↔Regel-Fragen finden die Regeldefinition | Mittel | Nach Bugfix-Phase |
