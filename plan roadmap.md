# LoreKeeper — Roadmap & Bearbeitungsliste

Stand: 2026-10-13

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
- Der Retriever schließt Bild-Chunks derzeit **komplett aus** (`src/retrieval/retriever.py`: Filter `document_type != image`), weil ihr fast leerer Text sonst nur Rauschen liefert — Bilder erscheinen deshalb nie als Quelle, obwohl die UI `st.image()` könnte
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
- Lokal über Ollama mit einem Vision-Modell (z.B. Qwen2.5-VL / Gemma 3) — passt zur Entscheidung „Gemini nur Fallback“ (Kostenrisiko, siehe 4.0); das Vision-Modell verdrängt beim Indizieren das Chat-Modell aus dem VRAM, also nur als Ingest-Schritt
- Alternative Gemini Flash (günstig, ~0,01 ct/Bild), aber kostenpflichtig
- Beschreibung wird als `content` des Bild-Chunks gespeichert
- Fallback: Dateiname + Ordner wenn kein Vision-Modell konfiguriert
- **Voraussetzung**: den Bild-Ausschluss im Retriever (`document_type != image`) aufheben, sobald Bilder echte Beschreibungen haben

#### 2b. Bilder in der Chat-Antwort anzeigen

Aktuell zeigt die UI Bilder nur im Source-Expander. Für echtes "Bild-im-Chat" braucht es:

1. **Backend**: Bild-Quellen als spezielle Referenz im Response markieren (schon vorhanden: `document_type: "image"`)
2. **Prompt-Anpassung**: System-Prompt muss dem LLM sagen, dass es Bild-Referenzen einbetten kann, z.B. `[Bild: Karte_Arkenfeld.png]`
3. **UI-Rendering**: Chat-Nachricht parsen, `[Bild: ...]` durch `st.image()` ersetzen
4. **Bild-Serving**: Bilder über Backend-Endpoint ausliefern (`GET /files/{path}`) statt lokaler Pfade — nötig, sobald die UI von einem anderen Gerät im Heimnetz geöffnet wird (4.1); Pfade nur innerhalb der Source-Ordner zulassen

### Implementierungsplan

**Phase 1 — Bessere Indizierung:**
1. Optionalen Vision-Beschreibungs-Schritt in `ImageMetaParser` einbauen (Gemini Vision oder lokales Modell)
2. Config-Key: `ingestion.image_description: none | gemini | local`
3. Beschreibung als Content speichern, Originalpfad als Metadatum beibehalten
4. Bild-Ausschluss im Retriever aufheben, Golden-Set-Fragen für Bilder (i001–i010) prüfen
5. Re-Ingest der Bilder (Last: Vision-Modell auf der GPU, nicht parallel zur Simulation)
6. **Vorher klären (Review 13.10.)**: Aus PDFs extrahierte Bilder zeigen heute auf ein temporäres Verzeichnis, das nach dem Parsen gelöscht wird — für Bildbeschreibungen und die Anzeige muss die Extraktion an einen dauerhaften Ort schreiben. Heute werden Bild-Chunks vom Retriever komplett ausgeschlossen, also nie gefunden.

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

**⏸ Zurückgestellt (12.10.)**: Fraglich, ob gespeicherte Sessions bei einem Wissenstool überhaupt Nutzen haben — erst entscheiden, dann ggf. umsetzen.

### 3.3 Chunk-Inspector / Debug-Ansicht — ✅ Erledigt

**Problem**: Wenn der Retriever falsche Chunks liefert, ist unklar warum. Man muss in ChromaDB schauen.

**Lösung**: Admin-Seite die pro Dokument die gespeicherten Chunks anzeigt — Content, Embedding-Dimension, Metadaten, Score bei Test-Query. Hilft beim Tuning.

**Aufwand**: Gering-Mittel. `GET /admin/chunks?source=...` Endpoint + UI-Seite.

**Teilweise abgedeckt**: Evaluation → „Frage testen“ zeigt bereits, welche Chunks (mit Score, Überschrift, `source_id`) eine Frage liefert. Offen bleibt nur die Sicht „alle Chunks einer Datei“.

**Umgesetzt (12.10.)**: Evaluation → Tab „Chunks“: Quelle und Datei wählen → alle gespeicherten Chunks in Dokumentreihenfolge (Überschriftenpfad, Länge, Textanfang), Statistik min/Ø/max mit Warnung für Chunks unter 100 Zeichen, voller Text + Metadaten eines gewählten Chunks. Endpoints `GET /eval/chunks/files` und `GET /eval/chunks` (nur lesend, nutzen die vorhandenen VectorStore-Abfragen). Scores pro Frage bleiben in „Frage testen“.

### 3.4 Antwort-Feedback (Thumbs Up/Down) — ✅ Erledigt

**Problem**: Keine Möglichkeit, Antwortqualität systematisch zu tracken.

**Lösung**: Thumbs-up/down unter jeder Antwort, gespeichert mit Frage + Antwort + Quellen. Basis für späteres Prompt-Tuning und Evaluation.

**Umgesetzt (12.10.)**: 👍/👎 (`st.feedback`) unter jeder Antwort, für Spieler **und** Spielleiter (Entscheidung). `POST /feedback` hängt eine Zeile an `logs/feedback.jsonl` (gitignored, keine neue Abhängigkeit): Zeit, Rolle, Frage, Antwort, Quelldateien, Modell, Suchanfrage, Session, Timings. Ändern/Zurücknehmen = neue Zeile mit derselben `message_id`, die letzte gilt. Tests: `tests/ui/test_feedback.py`, `test_feedback_*` in `tests/integration/test_api_routes.py`. Auswertung siehe `docs/operations.md` → Logs.

✅ **Erledigt**: Tab „Feedback“ auf der Evaluation-Seite zeigt 👎 (optional 👍) und übernimmt ausgewählte Antworten als Golden-Set-Fragen `f###`.

**Aufwand**: Gering (umgesetzt).

**Zusatz**: Bewertungen mit Daumen um kurzen kommentar möglichkeit erweitern - OFFEN

### 3.5 Wikilink-Auflösung im Retrieval

**Problem**: Obsidian-Wikilinks (`[[Arkenfeld]]`) werden als Metadaten gespeichert, aber nicht für Retrieval genutzt. Wenn ein Chunk auf `[[Malek]]` verweist, könnte der Retriever automatisch den Malek-Chunk mitliefern.

**Lösung**: Nach dem initialen Retrieval: Wikilinks der Top-Chunks extrahieren, referenzierte Chunks nachladen (Graph-Expansion). Gibt dem LLM mehr Kontext bei vernetzten Themen.

**Aufwand**: Mittel. Wikilink→Chunk-Mapping beim Ingest bauen, Retriever erweitern.


### 3.6 Chat-Verlauf im Antwort-Prompt — ✅ Erledigt

**Problem**: Das LLM sieht bei der Antwort nur die kondensierte Frage, nicht den bisherigen Verlauf. Folgefragen, die sich auf die *vorherige Antwort* beziehen ("fass das kürzer zusammen", "was hast du eben über seine Schwester gesagt?"), funktionieren deshalb nicht — der Condense-Schritt löst nur Referenzen für das Retrieval auf.

**Lösung**: Die letzten 1–2 Frage/Antwort-Paare (begrenzt durch `conversation.max_context_tokens`) zusätzlich in den QA-Prompt geben. `qa`-Template in `prompts.yaml` um einen optionalen `history`-Block erweitern, `render_qa()` reicht ihn durch.

**Umgesetzt (12.10.)**: Nur das **letzte** Frage/Antwort-Paar (`conversation.answer_history_pairs: 1`, 0 = aus), frühere Antwort auf 1500 Zeichen gekürzt (`answer_history_max_chars`). Steht vor den Quellen und ist als „keine Quelle“ markiert: Fakten nur aus den Quellen. Begründung: mehr Verlauf verwässert die Quellen und gibt früheren (evtl. halluzinierten) Antworten Gewicht; das Sliding Window (8 Paare) bleibt nur für den Condense-Schritt. Sessions sind rollengebunden → Spieler sehen nur ihren eigenen Verlauf. Prompt-Varianten ohne `history`-Block ignorieren ihn.

**Offen**: Wirkung mit echten LLM-Antworten prüfen (Folgefragen wie „fass das kürzer zusammen“, „was hast du eben über X gesagt?“) — braucht freie CPU/GPU.

**Aufwand**: Gering-Mittel (umgesetzt).

### 3.7 Zugriffsschutz für die UI — ✅ Erledigt

**Problem**: Die API selbst ist nicht exponiert — uvicorn bindet auf `127.0.0.1`, und alle API-Aufrufe macht der Streamlit-Prozess serverseitig. Exponiert ist die **Streamlit-UI**: sie lauscht standardmäßig auf allen Interfaces und hat keinen Login. Wer im selben Netz `http://<pc>:8501` öffnet, kann über die Admin-Seiten indizieren, die Collection löschen (`/admin/wipe`), Config und Gemini-Key ändern. Zusätzlich lässt das Feld „API URL“ in der Sidebar den Streamlit-Server beliebige URLs abfragen.

**Lösung**:
- Passwort-Gate (Secret aus `.env`) vor den Admin-Seiten (Sources, Settings, Prompts, Evaluation); der Chat bleibt frei oder hängt optional am selben Gate
- „API URL“-Feld durch Config-Wert ersetzen (nicht im Browser editierbar)
- Backend bleibt auf `127.0.0.1` — ein API-Key wird erst nötig, wenn die API selbst je exponiert wird

**Aufwand**: Gering. Teil von 4.1.

**Umgesetzt** (`ui/admin_gate.py`): `LOREKEEPER_ADMIN_PASSWORD` in `.env` schützt Sources, Evaluation, Prompts, Settings sowie Provider-Wechsel und Gemini-Key in der Chat-Sidebar; der Chat bleibt offen (Entscheidung: nur Admin-Seiten schützen). Login gilt pro Browser-Session. Ohne Passwort alles offen wie bisher. „API URL“-Feld entfernt, Backend-URL aus `LOREKEEPER_API_URL`. Tests: `tests/ui/test_admin_gate.py`.

### 3.8 Laufenden Ingest-Job wieder aufnehmen — ✅ Erledigt

**Problem**: Startet man eine Indizierung, während schon eine läuft, antwortet das Backend mit 409 („Eine Indexierung läuft bereits") — die UI zeigt nur diese Warnung. Den Fortschritt des laufenden Jobs sieht man nicht mehr, z.B. nachdem ein Widget-Klick den Poll-Loop abgebrochen hat.

**Lösung**: `_start_ingest_job` gibt im 409-Body die `job_id` des laufenden Jobs mit; `_start_ingest` in `ui/pages/1_Sources.py` pollt dann diesen Job statt nur zu warnen. Alternativ `GET /ingest/active`.

**Aufwand**: Gering.

### 3.9 Nicht-blockierender Ingest-Fortschritt in der UI — ✅ Erledigt

**Problem**: `_poll_ingest_job` pollt synchron in einer `while`-Schleife. Solange ein Job läuft (Minuten), rendert der Rest der Sources-Seite nicht (bei Reindex pro Source fehlen alle Elemente unterhalb des Expanders), und jede Interaktion bricht die Fortschrittsanzeige ab — der Job läuft im Backend weiter.

**Lösung**: `job_id` in `st.session_state` halten und den Fortschritt in einem `st.fragment(run_every=...)` aktualisieren. Kombiniert mit 3.8 überlebt die Anzeige auch Seitenwechsel.

**Umgesetzt (3.8 + 3.9)**: `GET /ingest/active` liefert den laufenden Job (oder `null`); die Sources-Seite hält die Job-ID in `st.session_state` und zeigt den Fortschritt in einem `st.fragment(run_every=1.5)` unter den Ingest-Buttons. Die Seite bleibt vollständig gerendert und bedienbar, Klicks und Seitenwechsel verlieren den Job nicht, Ingest-/Reindex-Buttons sind währenddessen gesperrt. Bei 409 (Job aus einer anderen Browser-Session) hängt sich die Seite über `/ingest/active` an. Tests: `tests/ui/test_sources_ingest.py`, `test_ingest_active_reports_the_running_job`. **Offen**: einmal im laufenden Browser durchklicken (Job starten, auf der Seite klicken, Seite wechseln und zurück) — am besten beim nächsten Re-Index (6.0).

**Aufwand**: Gering-Mittel (umgesetzt).

### 3.10 Regelwerk-Slot im Retrieval (Vault ↔ Regelwerk) — ✅ Erledigt

**Problem**: Fragen, die eine Vault-Entität *und* einen Regelbegriff nennen („Was bewirkt Ritterlichkeit bei Escanor?“, „Was kann Thalions Phönixfeder?“), liefern nur Chunks aus der NPC-Datei. Die Regeldefinition im PDF erreicht die Top-8 nicht — die NPC-Datei belegt alle Plätze bis zum `max_per_source`-Cap, der Rest geht an andere Vault-Dateien. Messbar im Golden Set: x002–x008 sind Abschnitt-Misses (`expected_section`), obwohl der Datei-Hit stimmt.

**Lösung**: Nach dem Rerank mindestens einen Platz für die Gruppe `rules` reservieren, wenn ein Regel-Chunk unter den Kandidaten ist und einen Mindest-Rerank-Score erreicht (analog zum bestehenden Backfill im Soft-Cap). Optional nur auslösen, wenn die Frage einen Begriff enthält, der als Abschnittsüberschrift im Regelwerk existiert (Eigenheit, Zauber, Artefakt). Danach Golden Set mit Abschnittsprüfung laufen lassen; Ziel: x002–x008 treffen, ohne die übrigen Abschnitt-Treffer zu verschlechtern (Stand 12.10. mit Simulationsquellen: 121/137). Der Slot gilt nur für `rules`, nicht für `simulation` — sonst verdrängen die Simulationstabellen das Regelbuch noch stärker.

**Aufwand**: Mittel. Mehrere Golden-Set-Läufe — seit dem GPU-Reranker (5.4) je ~1 min statt ~12 min.

**Befund (13.10.)**: Ein Slot *nach* dem Rerank reicht nicht — bei x002–x007 ist **kein einziger** Regel-Chunk unter den 15 Vektor-Kandidaten (die NPC-Datei füllt sie). Eine Vektorsuche nur in `rules` findet dagegen x001–x007 unter den ersten 8.

**Umgesetzt**: `retrieval.rules_slot` — zweite Vektorsuche in Gruppe `rules` (5 Kandidaten, AND mit dem Anfrage-Filter → Spieler-Modus bleibt dicht), mit gereranked, aber getrennt vom Haupt-Pool; fehlt in den Top-8 ein `rules`-Chunk, ersetzt der beste mit Rerank-Score ≥ −4 den schwächsten (max. einer). Erste Variante (Extra-Kandidaten in den Haupt-Pool gemischt) brachte 138/153, aber Regel-Chunks in 121 von 138 reinen Lore-Fragen → verworfen.

**Messung (Golden Set, alle Varianten im selben Prozess)**: aus 132/153 Abschnitte, x 1/8, Lore-Fragen mit Regel-Chunk 24/138 · −3: 133, 2/8, 48 · **−4: 136, 5/8, 67 (5 Kandidaten: 64)** · −5/−6/−8: 136, 5/8, 91/119/136. Dateien unverändert 0,986. Offen: x003 (richtiger Chunk −7,5 < falscher −5,5), x004 (Tabellen-Chunk ohne Überschrift), x008 (Heimkehrpfennig findet die Regel-Suche nicht).

**Restliche Abschnitt-Misses (separat ansehen)**: Wortähnlichkeit zieht falsche Dateien („Armbrust“ → Bargrun der Armdrücker, „Verbände“ → Balancing-Notiz) und Nachbarabschnitte gewinnen (Erlernen → 5.3 statt 5.1, Vorteil/Nachteil → 1.1 statt 1.4).

### 3.11 Spieler-/Spielleiter-Modus — ✅ Erledigt (inkl. Sicherheitstests)

**Problem**: Im Heimnetz nutzen auch Spieler LoreKeeper (z.B. am Handy). Der Vault enthält Spielleiter-Wissen (Plot, NPC-Geheimnisse wie Sol Invictus = Rhaz'Muell), das Spieler nicht sehen sollen; für sie ist nur das Regelwerk relevant.

**Lösung (umgesetzt)**: Vor dem Chat die Auswahl „🎲 Spieler“ / „📜 Spielleiter“ (`ui/roles.py`, `ui/LoreKeeper.py`), ohne Passwort (Entscheidung: Spoiler-Schutz, keine Zugangskontrolle).
- Spieler durchsuchen nur Quellen mit `player_visible` (neues Source-Feld, Spalte „🎲 Spieler“ auf der Sources-Seite; freigegeben: `pnp_regelbuch`, `charakterbogen`). Filter `{"source_id": {"$in": [...]}}`, gesetzt im Streamlit-Prozess und seit 12.10. zusätzlich vom Backend selbst erzwungen (`role`, siehe 4.2); Sidebar-Filter für Spieler ausgeblendet und ignoriert.
- „🔄 Rolle wechseln“ startet eine neue Session (kein Spielleiter-Verlauf im Spieler-Kontext).
- Ohne freigegebene Quelle ist der Spieler-Chat gesperrt.
- Live geprüft: „Wer ist Sol Invictus wirklich?“ / „Was ist mit Isolde Vaer los?“ im Spieler-Modus finden nur Regelbuch/Charakterbogen, Antwort „keine Informationen“.
- Doku: `docs/ui-ux.md` (Role selection), `docs/configuration.md` (`player_visible`), `docs/operations.md`.

**Sicherheitstests — ✅ Erledigt** (Details `docs/testing.md` → „Sicherheitsprüfungen“):
- A, deterministisch: Retrieval-Pfade mit Köder-Inhalten (Vektor, BM25/Hybrid, Rerank-Backfill, Section Expansion ohne Top-Rank-Limit, Condense) — `tests/integration/test_player_mode_security.py`.
- B, deterministisch: UI/Session (manipulierter/übriggebliebener Session-State, Folgefragen, Fail-closed bei Backend-Fehler, Freigabe-Entzug) und Login-Bruteforce — `tests/ui/test_player_mode_security.py`.
- C, live (`pytest -m live`): alle Vault-Fragen des Golden Sets im Spieler-Modus (Vektor + Hybrid) und fünf SL-Geheimnisse über den vollen Antwortpfad — `tests/live/test_player_mode_live.py`.
- Mutationsgeprüft (7 absichtlich eingebaute Lücken werden erkannt).

**Behoben im Zuge der Tests**:
- **Login-Bruteforce**: unbegrenzte Passwortversuche → jetzt 1 s Verzögerung pro Fehlversuch, nach 5 Fehlversuchen in 5 min Login-Sperre für 5 min (serverweit, beide Login-Formulare; angemeldete Sessions unberührt).
- **Veraltete Freigabe**: Entzog die SL einer Quelle die Spieler-Freigabe, suchte der Chat bis zu 30 s weiter darin (Sidebar-Cache) → Speichern auf der Sources-Seite leert jetzt den gesamten Streamlit-Cache.
- **Section Expansion verlor Chunks**: Duplikat-Erkennung nur über (Dateiname, Chunk-Nr.) — gleichnamige Dateien in zwei Quellen verdrängten sich gegenseitig → jetzt mit `source_id`.
- `/eval/preview` akzeptiert `metadata_filters` und liefert `source_id` je Chunk (für die Live-Checks).

**Aufwand**: Gering-Mittel (umgesetzt); Sicherheitstests Mittel.

## 4.0 Zugriff auf LoreKeeper aus dem Netzwerk und dem Internet

**Problem**: Der Laptop ist zu schwach für ein lokales LLM. LoreKeeper soll auf dem Gaming-PC laufen (Ollama + Backend + UI) und vom Laptop aus bedient werden — zuerst im Heimnetz, später ggf. auch unterwegs. Die Alternative, nur noch Gemini (Bezahlvariante) zu nutzen, hat ein Kostenrisiko: Budgets in Google Cloud senden nach meinem Kenntnisstand nur Warnungen und stoppen nicht automatisch; begrenzen lässt sich das Risiko nur über heruntergesetzte Quotas (Requests/Minute bzw. /Tag) oder eine selbstgebaute Budget→Billing-aus-Automatik. Gemini bleibt deshalb Fallback, nicht Hauptweg.

**Architektur-Befund**: Der Browser spricht nur mit Streamlit (Port 8501); alle API-Aufrufe laufen serverseitig im Streamlit-Prozess gegen `127.0.0.1:8000`. Es muss also **nur Port 8501** erreichbar sein, das Backend bleibt lokal.

### 4.1 Heimnetz (LAN) — ✅ Umgesetzt, Einrichtung am Gaming-PC offen

**Lösung**:
1. **Erreichbarkeit**: Streamlit lauscht bereits auf allen Interfaces. Windows-Firewall-Regel nur für Port 8501 und nur im Netzwerkprofil „Privat“; feste IP für den Gaming-PC per DHCP-Reservierung im Router. Aufruf vom Laptop: `http://<gaming-pc-ip>:8501`.
2. **Server-Modus in `start.ps1`**: Schalter (z.B. `-Server`) startet uvicorn ohne `--reload` (Code-Änderungen starten den Server nicht mitten in einer Session neu) und öffnet keinen Browser auf dem PC.
3. **Kein Autostart** (Entscheidung): LoreKeeper wird bei Bedarf manuell mit `.\start.ps1 -Server` gestartet; die Ollama-App startet ohnehin mit Windows. Ollama-Umgebungsvariablen auf dem Gaming-PC: `OLLAMA_CONTEXT_LENGTH=16384` (erledigt, siehe 5.3), `OLLAMA_KEEP_ALIVE=1h` (erledigt).
4. **Zugriffsschutz**: 3.7 (Passwort-Gate für Admin-Seiten, API-URL-Feld sperren) — Voraussetzung, sobald der Port offen ist.
5. **Doku**: Abschnitt „Betrieb im Heimnetz“ in `ARCHITECTURE.md` (Deployment).

**Aufwand**: Gering (inkl. 3.7).

**Umgesetzt**:
- `start.ps1 -Server`: uvicorn ohne `--reload`, explizit auf `127.0.0.1`; Streamlit auf `0.0.0.0`; minimierte Fenster, kein Browser; gibt die LAN-URLs aus und warnt, wenn kein Admin-Passwort gesetzt ist.
- Entwicklungsmodus (`.\start.ps1`, aber auch `streamlit run` direkt via `.streamlit/config.toml`) bindet die UI jetzt nur noch an `127.0.0.1` — vorher lauschte sie ungeschützt auf allen Interfaces.
- `scripts/setup-lan.ps1` (als Admin, einmalig, `-Uninstall` zum Rückgängigmachen): Firewall-Regel TCP 8501 nur Profil „Privat“; warnt bei Netzwerkprofil „Öffentlich“ und bei Python-Block-Regeln; kein Autostart (eine Autostart-Aufgabe aus der ersten Skript-Version wird entfernt).
- Doku: `docs/operations.md` → „Home-network server (LAN)“.

**Offen — Schritte am Gaming-PC (Nutzer)**:
1. ✅ `LOREKEEPER_ADMIN_PASSWORD` in `.env` gesetzt.
2. ✅ Heimnetz („Netzwerk 3“) auf „Privat“ umgestellt.
3. ✅ `scripts\setup-lan.ps1` als Administrator ausgeführt; Server-Modus läuft (`0.0.0.0:8501`, Backend `127.0.0.1:8000`).
4. Autostart-Aufgabe der ersten Skript-Version entfernen (braucht Admin): `Unregister-ScheduledTask -TaskName "LoreKeeper Server" -Confirm:$false`
5. DHCP-Reservierung im Router (offen); ✅ Zugriff vom Handy im Heimnetz funktioniert.
6. ✅ Startup-Messung (5.2): Baseline 17,7 s aufgenommen (12.10.).

**Befund**: Es existieren Inbound-Allow-Regeln für `python.exe` im Profil „Öffentlich“ (alle Ports) — für andere Python-Installationen (alte LoreKeeper-Kopie unter OneDrive, PortraitForge, globales Python 3.12), nicht für LoreKeeper. Nicht angefasst; ggf. selbst aufräumen.

### 4.2 Internet — später

**Lösung**: **Tailscale** (privates Mesh-VPN, kostenlos für Privatnutzung) auf Gaming-PC und Laptop → `http://gaming-pc:8501` funktioniert von überall, ohne Portfreigabe im Router und ohne öffentliche Erreichbarkeit. Alternative: Cloudflare Tunnel + Cloudflare Access (Login-Seite davor) — mehr Einrichtung.

**Offener Punkt**: Der Gaming-PC muss laufen. Wake-on-LAN von unterwegs braucht ein immer laufendes Gerät im Heimnetz (Router-Funktion oder Raspberry Pi); sonst bleibt der PC an.

**✅ Erledigt (12.10.) — Spieler-Modus im Backend durchsetzen**: Welche Quellen Spieler sehen, steht zentral in `sources.yaml` (`player_visible`), durchgesetzt wird es aber nur in der Streamlit-UI (`ui/roles.py:query_filters` baut den `source_id`-Filter, das Backend kennt keine Rollen und führt jeden `metadata_filters` aus). Im Heimnetz reicht das, weil das Backend nur auf `127.0.0.1` lauscht und nur Streamlit es erreicht. Sobald die API selbst von außen erreichbar wird (oder ein weiterer Client dazukommt), fehlt der Spieler-Schutz dort → `/query`, `/query/stream` und `/eval/preview` nehmen eine Rolle entgegen und bauen den Spieler-Filter selbst aus `player_visible`; ein mitgeschickter Filter darf ihn nur einschränken, nie erweitern. Die Sicherheitstests aus 3.11 (A/C) dann gegen die API-Seite richten.
**Umgesetzt**: `role` (`player`/`game_master`, Default `game_master` für CLI/Eval-Tools) auf `/query`, `/query/stream`, `/eval/preview`; `src/api/access.py` baut für Spieler `source_id ∈ player_visible` und AND-et mitgeschickte Filter (können nur einengen), keine freigegebene Quelle → 403. Sessions sind an die Rolle gebunden (Spieler mit SL-`session_id` bekommt eine neue Session). Die UI schickt `role` und weiterhin ihren eigenen Filter als zweite Sicherung. Tests in `tests/integration/test_player_mode_security.py` (mutationsgeprüft: 11 Fehlschläge, wenn der Backend-Filter fehlt), Live-Test schickt nur noch `role`. **Offen, wenn die API je exponiert wird**: Default auf `player` drehen bzw. Authentifizierung — der Default `game_master` vertraut dem Aufrufer.

**Aufwand**: Gering (Einrichtung, kaum Code); Spieler-Modus im Backend erledigt.

## 5.0 Performance-Optimierung und Performance-Tests

**Ziel**: Bei Weiterentwicklungen sofort sehen, wenn etwas deutlich ineffizienter wird — vor allem bei der Antwortzeit und der UI-Reaktion, beim Start nur einfache Hebel.

**Stand (2026-10-13)**:

| Bereich | Status | Ergebnis |
|---|---|---|
| Antwortzeit | ✅ große Hebel erledigt (5.1, 5.3) | Erster Token 8,7 s → **1,1 s**, gesamt 13,9 s → **3,7 s** (Überschaden, warm); kein abgeschnittener Prompt mehr; kein 16-s-Kaltstart innerhalb 1 h |
| Retrieval | ✅ erledigt (5.4) | Rerank auf der GPU via DirectML (5.4): 840 ms → 50 ms, unter CPU-Last 14 s → 70 ms |
| UI-Reaktion | ✅ erledigt (5.2, 5.3) | `localhost`-Delay von ~2 s pro Backend-Request beseitigt; Chat 1 Request pro Nachricht, Sources-Rerun 0 Requests, Ingest-Fortschritt blockiert nicht mehr (3.9) |
| Start | ✅ Baseline (5.2) | **17,7 s** bis `/health`: ~9,6 s Python-Imports (davon `sentence-transformers`/torch ~8 s), ~6,7 s Lifespan (vorher 12,2 s — Reranker auf der GPU lädt in 1,6 s statt 4,3 s), Rest uvicorn |
| Reproduzierbarkeit | ⚠️ offen (5.5) | Vektorsuche liefert nach Neustart für ~9 von 278 Golden-Fragen andere Kandidaten → Eval-Läufe schwanken ±2 Abschnitte |
| Regressionsschutz | ✅ erledigt (5.2) | Deterministische Budgets im normalen `pytest`, Zeit-Checks per `pytest -m perf` gegen `tests/perf/baseline.json`, inkl. Startup |

### 5.1 Messen: Stage-Timings — ✅ Erledigt

Pro Anfrage loggen und im `done`-Event mitliefern: Condense, Retrieval (Embedding / Vektorsuche / Rerank / Expansion), Zeit bis zum ersten Token, Gesamtzeit, Anzahl Chunks und Prompt-Größe (Zeichen bzw. `usage.tokens_in`). Grundlage für alles Weitere; die UI kann die Werte optional in der Caption zeigen.

**Aufwand**: Gering.

**Umgesetzt**: `timings` in `/query` und im `done`-Event (`condense_ms`, `retrieval_ms` + `retrieval_stages_ms` {embed, search, bm25, rerank, expand}, `first_token_ms`, `total_ms`, `chunks`, `prompt_chars`), eine Log-Zeile `Query timings: …` pro Anfrage, Caption „⏱ 1. Token … · gesamt … · Suche … · N Chunks“ unter jeder Antwort.

**Erste Messung (qwen3:14b, warm, GPU)**: Retrieval 0,6–0,8 s (davon Rerank ~0,5–0,75 s), erster Token nach 8,7–9,9 s, gesamt 14–20 s. → Der große Block vor dem ersten Token ist **nicht** Retrieval, sondern verstecktes Denken (siehe 5.3).

### 5.2 Performance-Tests (Regressionsschutz) — ✅ Erledigt

Zwei Arten von Prüfungen — **deterministische** schlagen zuverlässig an und laufen im normalen `pytest`, **zeitbasierte** sind hardwareabhängig und laufen separat:

| Prüfung | Art | Erkennt z.B. |
|---|---|---|
| Max. Chunks / Prompt-Zeichen im QA-Prompt pro Golden-Frage | deterministisch, fester Grenzwert | Kontext-Aufblähung wie die 29-Chunk-Expansion |
| Anzahl LLM-Aufrufe pro Anfrage (Condense ja/nein) | deterministisch | unnötige Zusatz-Calls |
| Anzahl Backend-Requests pro UI-Rerun je Seite (Streamlit `AppTest` mit Fake-Backend) | deterministisch | fehlendes Caching, Polling-Schleifen |
| Retrieval-Latenz p50/p95 über das Golden Set | zeitbasiert, `pytest -m perf` | langsame Reranker-/Expansion-Änderungen |
| Render-Zeit je UI-Seite (`AppTest`) | zeitbasiert, `pytest -m perf` | schwere Reruns beim Klicken / Seitenwechsel |
| Startup bis `/health` = healthy | zeitbasiert, `pytest -m perf` | zusätzliche Modell-Ladezeit |

Zeitbasierte Tests vergleichen gegen eine eingecheckte **Baseline-Datei** (`tests/perf/baseline.json`) mit Toleranz (z.B. Faktor 1,5) statt gegen absolute Werte; Baseline wird bewusst per Flag aktualisiert. LLM-Antwortzeiten werden nicht automatisch getestet (Modell-/Hardwareabhängig) — dafür liefert 5.1 die Messwerte.

**Aufwand**: Mittel.

**Umgesetzt** (Details in `docs/testing.md` → „Performance-Prüfungen“):
- Deterministisch, im normalen `pytest`: Backend-Requests pro Seiten-Render/Rerun/Chat-Nachricht (`tests/ui/test_backend_calls.py`, `AppTest` + Fake-Backend, Budgets in `BUDGETS`); LLM-Aufrufe pro Anfrage (`test_llm_calls_per_request`); keine `localhost`-URLs für lokale Dienste.
- Zeitbasiert, `pytest -m perf` gegen `tests/perf/baseline.json` (×1,5 + Slack; `PERF_UPDATE_BASELINE=1` übernimmt neue Werte): Render-Zeit je Seite; Golden Set über das laufende Backend — p50/p95 Retrieval-Latenz, Ø/max Chunks pro Frage, Abschnitt-Trefferquote darf nicht sinken.
- **Startup-Messung** (`tests/perf/test_startup.py`, 12.10.): startet ein zweites Backend auf freiem Port gegen echte Config und `chroma_data/`, misst bis zur ersten `/health`-Antwort (die Lifespan lädt Ollama-Autostart, Embedding-Warm-up, ChromaDB, Reranker vorher). Wird übersprungen, solange LoreKeeper läuft (eingebettete ChromaDB nicht doppelt öffnenbar). ✅ **Baseline (12.10.)**: 17,7 s (zweiter Lauf 18,4 s Testdauer, bestanden; Grenze ×1,5 + 3 s). Aufgenommen bei gestopptem LoreKeeper, ohne Simulation, GPU-Reranker aktiv.

**Befunde aus der ersten Messung**:
- **`localhost` kostet 2 s pro Verbindung (behoben)**: Unter Windows löst `localhost` zuerst auf IPv6 `::1` auf, uvicorn und Ollama lauschen nur auf IPv4 → jede neue Verbindung wartet ~2 s auf den Fallback (gemessen 2000 ms vs. 3 ms). Betraf jede Chat-Nachricht (~4 s: `/config` + `/query/stream`), jeden Sources-Seiten-Aufbau (~10 s) und neue Verbindungen zu Ollama nach einer Pause. Alle Standard-URLs (UI, Config, Eval) stehen jetzt auf `127.0.0.1`; ein Test verhindert die Rückkehr. **Achtung**: Eine offene Browser-Session mit `localhost` im Sidebar-Feld „API URL“ behält den Delay bis zum Neuladen.
- **UI-Code selbst ist schnell**: Rerun je Seite 0,02–0,04 s, erster Aufbau < 0,7 s (Fake-Backend) — die gefühlte Trägheit kam aus den Backend-Aufrufen.
- **Sources-Seite**: `/sources/{id}/folders` wird gecacht (60 s, Speichern leert den Cache) — ein Rerun ohne Änderung kostet 0 Requests, als Budget im Test festgehalten.

### 5.3 Bekannte Hebel (nach der Messung angehen)

- **Antwortzeit — qwen3 denkt trotz `/no_think` (gemessen, größter Hebel)**: Ollama ignoriert das an den Prompt gehängte `/no_think`; qwen3 erzeugt Reasoning-Tokens im separaten Feld `reasoning`, das LoreKeeper weder anzeigt noch zählt (`tokens_thinking` bleibt 0). Kostet ~7 s pro Antwort. Direkter Test gegen Ollama: `reasoning_effort: "none"` (bzw. `reasoning: {effort: "none"}`) schaltet es ab (0 Reasoning-Chunks), `think: false` wird ignoriert. ✅ **Erledigt**: `OllamaProvider` sendet für Qwen3 `extra_body={"reasoning_effort": "none"}` statt `/no_think`; Reasoning-Chunks im `reasoning`-Feld werden als `tokens_thinking` gezählt.
- **Antwortzeit — Kontextfenster (gemessen)**: `ollama ps` zeigt `CONTEXT 4096` für qwen3:14b. Die Krieger-Frage brauchte 3621 Eingabe- + 919 Ausgabe-Tokens — am bzw. über dem Limit; bei größerer Section Expansion wird der Prompt-Anfang (Systemprompt) still abgeschnitten. ✅ **Erledigt**: `OLLAMA_CONTEXT_LENGTH=16384` als Benutzer-Umgebungsvariable gesetzt (der OpenAI-kompatible Endpoint ignoriert `num_ctx` pro Request, getestet). Danach `CONTEXT 16384`, 11 GB von 16 GB VRAM, weiterhin 100 % GPU, keine messbare Verlangsamung (warm: erster Token 1,1 s / 2,3 s).
- **Antwortzeit — Modell entladen nach 5 Minuten (gemessen)**: Ollama entlädt das Modell nach 5 min Leerlauf; die erste Frage danach wartet ~16 s aufs Laden (erster Token 18 s statt 1,1 s). ✅ **Erledigt**: `OLLAMA_KEEP_ALIVE=1h` als Benutzer-Umgebungsvariable — das Modell bleibt bis 1 h nach der letzten Frage geladen (`ollama ps`: „59 minutes from now“). Belegt in der Zeit ~11 GB VRAM; `-1` hielte es dauerhaft geladen.
- **Präzision/Kontext — Section Expansion begrenzt** ✅ **Erledigt**: `section_expansion.top_rank: 3` — expandiert nur, wenn ein Treffer der Sektion unter den ersten 3 Chunks liegt. Golden Set (0 = altes Verhalten / 3 / 2 / 1): Ø Chunks 13,9 / 12,6 / 11,9 / 10,7; expandierte Fragen 144 / 119 / 102 / 75; Abschnitt-Treffer überall 89,1 %. Bei 3 fallen nur Fehl-Expansionen weg (z.B. Geisterwesen-Statblöcke bei „Eigenheiten Geist und Psyche“); 2 und 1 schneiden bereits relevante Übersichts-Chunks ab (Akademie-Geschichte, Szenenübersicht, Arkenfeld-Steckbrief). Offen: Gesamtbudget über alle expandierten Sektionen (max. 49 Chunks bleiben möglich). Unnötige Condense-Aufrufe sind bereits entfernt (`needs_condense`).
- **Retrieval — Reranker auf der CPU** ✅ **Erledigt** über 5.4 (DirectML): Rerank 840 ms → 50 ms. Embedding-Modell bleibt bewusst auf der CPU.
- **Kontext — Gesamtbudget für Section Expansion (offen)**: Mehrere expandierte Sektionen können zusammen noch bis zu 49 Chunks ergeben (z.B. „Wie wird die Bonusaktion im Kampf eingesetzt?“). Ein Budget über alle Sektionen (z.B. max. 20 Chunks) begrenzt das; messbar mit der Chunk-Prüfung aus 5.2.
- **UI-Reaktion (gemessen, Hauptursache behoben)**: Die 2-s-`localhost`-Verzögerung pro Request ist behoben (5.2). ✅ Reste erledigt: Der Chat holt das LLM-Timeout aus dem gecachten `/sidebar-state` statt `/config` vor jeder Nachricht (1 Request pro Nachricht statt 2); Sources-Seite cacht `/folders`; Polling blockiert nicht mehr (3.8/3.9). Kosten eines Reruns mit langer Chat-Historie und vielen Quellen-Expandern noch nicht gemessen.
- **Start (kein Handlungsbedarf)**: `--reload` im Server-Modus aus (4.1); Reranker lädt seit 5.4 auf der GPU in 1,6 s — lazy laden lohnt nicht mehr. Größter Block sind jetzt die Python-Imports (~9,6 s, `sentence-transformers`/torch); die ließen sich nur durch einen Verzicht auf torch fürs Embedding-Modell senken (Re-Index-Risiko, siehe 5.4) — für einen manuell gestarteten Dienst nicht lohnend.
- **Generierung**: Lange Antworten (Krieger: 655 Tokens ≈ 12 s) sind durch die Modellgeschwindigkeit (~50 Token/s) begrenzt — kein Performance-Bug; Stellschraube wäre ein Prompt, der kürzere Antworten verlangt (Produktentscheidung).

**Aufwand**: je Hebel Gering bis Mittel.


### 5.4 Reranker auf der GPU (DirectML) — ✅ Erledigt

**Warum überhaupt?** Der Cross-Encoder-Rerank ist der teuerste Schritt im Retrieval und läuft auf der CPU (`torch 2.14.1+cpu`). Gemessen (`logs/`, 34 Anfragen): Rerank Median 0,8 s, p90 1,5 s — aber **9,7–9,9 s, während die Simulation lief** (Retrieval gesamt 11,7 s, Embedding 1,7 s statt 0,08 s). Der eigentliche Nutzen ist deshalb nicht die halbe Sekunde im Normalfall, sondern:
1. **Unabhängig von CPU-Last**: Chat bleibt schnell, auch wenn die Simulation (oder ein Ingest) läuft — heute kostet jede Frage dann ~10 s extra.
2. **Evals und Indexierung**: Golden-Set-Lauf ~12 min (Rerank dominiert), Re-Index Regelwerk ~80 s. Auf der GPU würden 3.10 und das Section-Budget (je mehrere Läufe) billiger — und müssten nicht auf eine freie CPU warten (Eval am 12.10. abgebrochen wegen Simulation).
3. Im Normalfall: erster Token ~1,1 s → ~0,5 s. Spürbar, aber allein kein Grund.

**Optionen**

| Option | Gewinn | Kosten / Risiko | Bewertung |
|---|---|---|---|
| A. Reranker als ONNX **int8 auf der CPU** (`backend="onnx"`, `export_dynamic_quantized_onnx_model`) | weniger CPU-Arbeit pro Frage (Faktor erst messen); unter Last weiterhin Konkurrenz mit der Simulation | neue Abhängigkeit `optimum-onnx` (über `sentence-transformers[onnx]`), nutzt das **schon installierte** `onnxruntime 1.31` — kein Konflikt; Quantisierung kann Ranking leicht verschieben → Golden Set prüfen | **Erster Schritt**: kleines Risiko, sofort messbar |
| B. ONNX Runtime + **DirectML** (GPU) | Rerank weg von der CPU → Punkt 1 und 2 voll | DirectML ist laut Microsoft in *sustained engineering* (keine neuen Features, Nachfolger Windows ML); letztes Paket `onnxruntime-directml 1.24.4` (17.03.2026) vs. installiert `onnxruntime 1.31`. Beide Pakete schreiben ins selbe Modul `onnxruntime/` — chromadb, pymupdf-layout und rapidocr verlangen `onnxruntime`, also Downgrade für alle drei; jedes spätere `pip install -r requirements.txt` kann das CPU-Paket zurückholen → **stiller Rückfall auf CPU** (nur langsamer, nicht falsch). VRAM unkritisch (~0,5 GB neben 11 GB LLM) | Machbar, aber Sackgasse; nur mit Startup-Log/Test „DmlExecutionProvider aktiv“ |
| C. **Windows ML** (Nachfolger, wählt den Provider je Hardware, Python-API vorhanden) | wie B, zukunftssicher | Radeon-GPU-Pfad (MIGraphX-Provider) unter Windows aus Python nicht belegt; Doku auf NPU/Ryzen AI fokussiert | Beobachten, jetzt nicht |
| D. PyTorch + **ROCm für Windows** | wie B, ohne ONNX-Export | Bug [ROCm/TheRock #8379](https://github.com/ROCm/TheRock/issues/8379): RX 9070 XT liefert still Nullen → würde den Index zerstören | Erst nach Bugfix |
| E. Nichts tun, Evals/Ingest nur ohne Simulation | — | Status quo; Chat während Simulation ~10 s langsamer | Fallback |

**Nur den Reranker umstellen, nicht das Embedding-Modell**: Query-Embedding kostet normal ~0,08 s; ein anderes Backend ändert die Vektoren minimal → Index und Golden Set müssten neu verglichen bzw. neu indiziert werden. Der Reranker wirkt nur zur Abfragezeit, ein Wechsel ist jederzeit rückgängig zu machen.

**Empfehlung vor dem Benchmark (überholt, siehe Entscheidung unten)**: A zuerst (Config-Schalter `reranking.backend: torch | onnx`, Default `torch`; ONNX-Modell einmal exportieren und lokal speichern). Messen: Rerank-Zeit idle und unter Simulationslast, Golden Set Treffer/Abschnitte gegen heute (261/262, 121/137). Reicht das unter Last nicht, B als zweiter Schritt — mit Provider-Check beim Start. C/D halbjährlich neu prüfen.

**Benchmark (12.10., CPU frei, Ryzen 5 5600X / RX 9070 XT)**: 15 Paare in Produktionsgröße, 30 Wiederholungen; „Last“ = 12 Busy-Loop-Prozesse (≈ laufende Simulation).

| Variante | idle | unter Last | Ranking wie heute? |
|---|---|---|---|
| torch CPU (heute) | 840 ms | 14,2 s | — |
| ONNX fp32 CPU | 840 ms | 2,9 s | ✅ identisch |
| ONNX int8 CPU (`quint8_avx2`) | 710 ms | 2,8 s | ❌ Top-8 weicht ab |
| **DirectML fp32 (GPU)** | **50 ms** | **68 ms** | ✅ identisch (max. Abweichung 6·10⁻⁶) |
| DirectML fp16 (`O4`) | — | — | lädt nicht (`invalid unordered_map key`) |

int8 verworfen (kaum schneller, anderes Ranking). Modell-Laden DirectML 1,3 s idle / 10,7 s unter Last (einmalig beim Start).

**Entscheidung (12.10.): B, DirectML.** **Umgesetzt**: `reranking.backend: torch | directml` (Code-Default `torch`, `settings.yaml`: `directml`); `src/retrieval/onnx_reranker.py` lädt das veröffentlichte ONNX-Export des Modells (`onnx/model.onnx`) + `tokenizer.json` aus dem HF-Cache und läuft auf `DmlExecutionProvider` (Session-Lock, eine GPU-Queue). Kein DirectML / kein ONNX-Export / Session landet auf der CPU → **Warnung im Log** und Fallback auf torch (gleiche Scores, nur langsamer) — das ist der Schutz gegen den stillen Rückfall. Installation: nach `pip install -r requirements.txt` einmal `pip uninstall -y onnxruntime` + `pip install --force-reinstall --no-deps onnxruntime-directml` (README → Quickstart). Tests: `tests/unit/retrieval/test_onnx_reranker.py`, `TestRerankerBackend`, `tests/perf/test_reranker_gpu.py` (echtes Modell: gleiches Ranking, GPU < CPU/3).

**Abschluss (12.10.)**: Paket-Tausch in `.venv` erledigt (`onnxruntime-directml 1.24.4`; `pip check` meldet erwartungsgemäß das fehlende `onnxruntime`-Paket für chromadb/pymupdf-layout/rapidocr — das Modul ist da). Volle Testsuite grün (532), RapidOCR-Smoke-Test ok, `tests/perf/test_reranker_gpu.py`: 18 Paare CPU 118 ms → GPU 12 ms, gleiches Ranking. Golden Set mit GPU-Reranker in **1:12 min statt ~12 min**: Dateien 275/278, Abschnitte 132/153 (letzter CPU-Lauf 18:05: 274/278, 131/153). Die 9 Fragen mit anderer Trefferliste als 18:05 liefern im direkten A/B-Vergleich (gleicher Index, torch vs. DirectML) **identische** Ergebnisse (max. Score-Abweichung 3·10⁻⁶) — die Abweichung zu 18:05 kommt nicht vom GPU-Reranker. Offen: Rerank-Timings im laufenden Chat ansehen. Embedding-Modell bleibt auf der CPU (Query-Embedding ~0,08 s; Wechsel würde Re-Index erfordern). Windows ML / ROCm halbjährlich neu prüfen.

### 5.5 Vektorsuche nicht reproduzierbar (gefunden 13.10.)

**Problem**: Dieselbe Frage liefert nach einem Neustart für ~9 von 278 Golden-Fragen andere Vektor-Kandidaten (innerhalb eines Prozesses stabil; Query-Embeddings bitgleich). ChromaDBs HNSW-Suche ist approximativ — teils fehlt sogar ein besserer Treffer (p087: 0,796 in einem Lauf gefunden, im anderen nicht). Folge: Eval-Läufe in getrennten Prozessen schwanken um ±2 Abschnitte; Varianten nur im selben Prozess vergleichen.

**Lösung (zu prüfen)**: `ef_search` der Collection erhöhen (ChromaDB-HNSW-Konfiguration) oder bei der kleinen Collection (~20 k Chunks) exakt suchen; vorher messen, ob die Abweichung verschwindet und was es an Latenz kostet. Ändert die Collection-Konfiguration → nur bei gestopptem Backend.

**Aufwand**: Gering-Mittel. Last 🟡.

## 6.0 PnP-Simulation zugreifbar machen — 🔄 Umgesetzt, Abschluss nach dem nächsten Simulationslauf

**Entscheidung (12.10.)**: Minimaler Input — fünf Dateien aus `PnP_Simulation` als Einzeldatei-Sources in der neuen Gruppe **`simulation`** (Sidebar „📊 Simulation“, im Spielleiter-Modus abwählbar für reine Regelfragen). HTML, JSON, CSV, Cache, README und die übrigen `docs/` bleiben draußen (Entwickler-Doku, Duplikate des Vaults/Regelwerks oder veraltete Protokolle). Gegnerwerte und Kampagnen-Kämpfe stehen bereits über die `sim:start`-/`sim:kampf`-Blöcke im Vault.

| Source-ID | Datei | Spieler |
|---|---|---|
| `sim-begegnungen-bericht` | `output/begegnungen/llm_report.md` | ❌ (Gegner-LP/-Stärke) |
| `sim-begegnungsbaukasten` | `docs/begegnungen/begegnungsbaukasten.md` | ❌ |
| `sim-faehigkeiten-bericht` | `output/faehigkeiten/llm_report.md` | ✅ (Charakterbau) |
| `sim-faehigkeiten-analyse` | `docs/faehigkeiten/fachliche_analyse.md` | ✅ |
| `sim-balance-roadmap` | `docs/BALANCE_ROADMAP.md` | ❌ (nennt Kampagnen-Kämpfe) |

**Umgesetzt**: Gruppe `simulation` (Config, Chat-Sidebar, Sources-Seite, Doku, Tests); Fix: `max_per_source` im Reranker zählt je (`source_id`, Datei) — beide Berichte heißen `llm_report.md`; Evaluation: `expected_source_ids` und benannte Überschriften als `expected_section`; Golden Set z001–z016 (prüft Quelle und Abschnitt, Antwort-Keywords nur Methoden-Begriffe — robust gegen neue Läufe).

**Messung (12.10., Index mit SIM 50 und Begegnungsbericht vom 10.10.)**:
- Bestehendes Golden Set: Datei-Treffer unverändert 261/262; Abschnitt 122 → 121 von 137 (p061 verliert 4.8).
- Simulation z001–z016: Datei 13/16, Abschnitt 10/16. Misses z009/z012/z014 — ohne das Wort „Simulation“ gewinnt das Regelbuch.
- Verdrängung: 38 von 262 Fragen bekommen Simulations-Chunks in den Kontext (meist 1 von 8; Krit-Fragen p050/p052 4 von 8). Spielleiter: „📊 Simulation“ abwählen.
- Spieler-Modus live (`pytest -m live -k vault_question`): 278 passed — Begegnungsquellen erreichen Spieler nie.

**Offen (nach einem sauberen Simulationslauf)**:
- ⚠️ **Vorher nicht inkrementell indizieren**: Der fehlerhafte SIM-51-Lauf (12.10.) hat `output/faehigkeiten/llm_report.md` überschrieben; im Index steht noch SIM 50.
- Nach dem Lauf: `python begegnungen.py` (Begegnungsbericht passt sonst nicht zu den neuen Baukasten-Konstanten), dann „⚡ Nur Änderungen indizieren“, Eval wiederholen.
- Messen, wie viele Simulations-Chunks Spieler bei reinen Regelfragen bekommen (sie haben keinen Quellenfilter) → entscheiden, ob der Fähigkeitsbericht für Spieler freigegeben bleibt oder nur `fachliche_analyse.md`.

**Aufwand**: Gering (umgesetzt); Rest Messung.

---

## 7.0 Testlücken schließen — ✅ Erledigt

**Problem**: Einzelne Module sind kaum durch Tests abgedeckt (Stand Coverage-Lauf 10.10.): `src/api/prompt_routes.py` ~54 %, `src/generation/providers/gemini.py` ~42 %, `src/api/eval_routes.py` ~77 %. Fehler dort (Prompt-Varianten speichern/aktivieren, Gemini-Streaming/Token-Usage/Fehlerpfade, Eval-Jobs) fallen erst im Betrieb auf.

**Lösung**: Erst `pytest --cov=src --cov-report=term-missing` laufen lassen und die ungetesteten Zeilen ansehen, dann gezielt Tests ergänzen — Gemini mit gemocktem `google-genai`-Client (Streaming, Usage, Timeout, fehlender Key), Prompt-/Eval-Routes über `TestClient` wie die übrigen Integrationstests. Kein LLM- oder API-Aufruf in Tests.

**Umgesetzt (12.10.)**: alle drei Module auf 100 % — `gemini.py` 45 → 100 (Key-Herkunft, Generate, 429-Retry/Backoff, Rate-Limit, Streaming-Usage, Health-Check; gemockter Client), `prompt_routes.py` 54 → 100 (aktive Prompts, Varianten, Aktivieren, Vorschau gegen echten `ConfigManager` auf tmp-Dateien), `eval_routes.py` 77 → 100 (Start-Validierung, Fehlerpfade, E2E-Lauf, Ergebnisliste).

**Dabei gefunden und behoben**: Die Prompts-Seite markierte eine aktivierte Variante **nie** als „✅ aktiv“ — `is_active` verglich Datei-Hashes, die Variante enthält aber `_meta`, die aktive `prompts.yaml` nie. Jetzt Vergleich der vier Templates.

**Aufwand**: Gering-Mittel (umgesetzt).

### 7.1 Kleine Fixes (gefunden 12.10.) — ✅ Erledigt

- **Eval-CLI bricht in der Windows-Konsole ab**: `python -m evaluation.evaluate_retrieval` stürzt beim Ausgeben der Misses mit `UnicodeEncodeError` (cp1252) ab, sobald eine Überschrift ein Emoji enthält (z.B. 🎭). Der JSON-Report ist dann schon geschrieben, die Zusammenfassung fehlt. Fix: Ausgabe auf UTF-8 umstellen (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`). Workaround: `PYTHONIOENCODING=utf-8`. ✅ Behoben, Test `test_cli_summary_survives_a_cp1252_console`.

**Aufwand**: Gering (umgesetzt).

## 7.2 Code-Review der Änderungen vom 12./13.10. — ✅ Erledigt (inkl. Nachbesserung)

**Ergebnis**: Kein Weg am Spieler-Filter vorbei (Filter-Verknüpfung, BM25, Section Expansion, Regelwerk-Slot, Preview, Sessions geprüft). Bewusste Grenze: das Backend vertraut der Rolle des Clients (Default `game_master`), `/sessions`, `/feedback`, `/eval/chunks` ohne Login — reicht, solange nur Streamlit `127.0.0.1:8000` erreicht; vor 4.2 angehen.

**Behoben (13.10.)**:
- Regelwerk-Slot konnte bei `top_k_rerank` 1–2 fast den ganzen Kontext ersetzen (über Settings-Regler erreichbar) → Slot erst ab 3 Plätzen.
- GPU-Ausfall im Betrieb (Treiber-Reset, VRAM voll) ließ jede Frage bis zum Neustart scheitern → Reranker wechselt mit Warnung dauerhaft auf die CPU.
- Reranker-Laden beim ersten Rerank lief auf dem Event-Loop (wenn Reranking erst zur Laufzeit eingeschaltet wird) → Threadpool.
- Feedback → Golden Set übernahm Folgefragen ohne Kontext → umgeschriebene Frage (`search_query`), Chat-Frage in `notes`.
- Antwort-Prompt kürzte nur frühere Antworten, nicht lange (eingefügte) Fragen → beide gekürzt.
- Feedback-Tab: Häkchen konnten nach 👍-Schalter auf andere Antworten rutschen → Tabelle neu je angezeigter Liste.
- Ungültige `metadata_filters` gaben 500 → Prüfung an der API, 422 (`validate_filters` in `src/api/access.py`).
- Chunk-Inspector zeigte bei ChromaDB-Fehlern „keine Dateien“ → eigene Abfrage `list_files`, Fehler 503 und Meldung in der UI.
- Testlücken: Sicherheitstests jetzt mit Regelwerk-Slot (ohne Mindest-Score) und einer Spielleiter-Regel-Quelle als Köder, rollenbasierte Tests auch mit Hybrid-Suche; mutationsgeprüft (Slot-Suche ohne Anfrage-Filter → beide Varianten schlagen fehl).

**Kein Fehler**: „Regelwerk-Kandidaten gehen verloren, wenn die Hauptsuche leer ist“ — die Regelwerk-Suche ist die Hauptsuche plus Filter; erreicht dort kein Treffer 0,5, kann es kein Regel-Chunk.

**Zweites Review (13.10., über die noch nicht committeten Fixes)**: Kein Spieler-Leck; 422 greift vor jedem Seiteneffekt; die App-eigenen Filter (Gruppenfilter der UI, Spieler-Filter, Live-Tests) bestehen die neue Prüfung; der neue Slot-Sicherheitstest schlägt bei entferntem Filter fehl. Gefunden:

| # | Befund | Schwere | Geprüft |
|---|--------|---------|---------|
| N1 | `validate_filters` ist lockerer als ChromaDB: mehrere Schlüssel in einer `$and`-Bedingung, `$gt`/`$lt` mit Text, `$in` mit gemischten Typen gehen durch → weiter 500. Umgekehrt lehnt sie `$contains` ab (sendet die App nie) | Mittel | Skript gegen ChromaDB 1.5.9 |
| N2 | BM25 (`_matches_filter`) kennt nur `$eq`/`$ne`/`$in`/`$and`: `$nin`/`$gt`/… werden ignoriert (Bedingung wirkungslos), `$or` verwirft alle Treffer. Die Doku nennt sie als erlaubt | Niedrig (Spieler nicht betroffen — ihre Einschränkung ist `$in`) | Code + Skript |
| N3 | GPU→CPU-Wechsel entscheidet über das geteilte Flag statt über das Modell des Aufrufs: eine zweite, noch auf der GPU laufende Frage scheitert trotzdem; zwei gleichzeitige Ausfälle laden das CPU-Modell doppelt | Niedrig-Mittel | Skript (beide Fälle nachgestellt) |
| N4 | Chunk-Inspector zeigt bei 503 Fehler **und** „Keine indizierten Dateien“ | Niedrig | Code |
| N5 | Regelwerk-Suche läuft auch bei `top_k_rerank` < 3, obwohl der Slot dann nicht greift (verschenkte Abfrage) | Sehr niedrig | Code |
| N6 | Testlücke: der Slot-Sicherheitstest ist rein negativ — kein Test zeigt, dass der Slot für Spieler innerhalb einer freigegebenen Regel-Quelle auch feuert | Niedrig | Mutation |
| N7 | Kommentar in `config/settings.yaml` („earlier answers are cut here“) — jetzt werden auch Fragen gekürzt | Kosmetik | Code |

**Korrekturplan**:
1. **N1 + N2 zusammen**: `validate_filters` auf die Schnittmenge beschränken, die ChromaDB **und** BM25 verstehen — `$eq`, `$ne`, `$in` (nicht-leere Liste, ein Typ), `$and` (≥ 2 Bedingungen, je **genau ein** Schlüssel). `$or`, `$nin`, `$gt`/`$gte`/`$lt`/`$lte`, `$contains` → 422. Die App sendet nichts davon. Tests: neue Fälle in `TestValidateFilters`, je ein Fall pro Lücke gegen eine echte `EphemeralClient`-Collection, dass alles, was die Prüfung durchlässt, ChromaDB auch annimmt. Doku: Filterabschnitt in `configuration.md`.
2. **N3**: in `_predict` vor dem Aufruf festhalten, ob dieses Modell das GPU-Modell ist; beim Fehler unter einem `asyncio.Lock` nur dann das CPU-Modell laden, wenn noch niemand gewechselt hat, sonst das bereits geladene nutzen. Test: zwei gleichzeitige Fragen bei ausfallender GPU → beide beantwortet, CPU-Modell genau einmal geladen.
3. **N4**: Fehler-Flag im Chunk-Tab, Hinweis „Keine indizierten Dateien“ nur ohne Fehler. UI-Test mit 503 aus dem Fake-Backend (dessen `raise_for_status` muss dafür bei Fehlercodes werfen).
4. **N5**: Regelwerk-Suche in `retrieve` nur, wenn `top_k_rerank` (Anfrage oder Config) ≥ 3. Test: keine zweite Vektorabfrage bei `top_k_rerank=1`.
5. **N6**: Positiver Sicherheitstest — Spieler mit Zugriff auf `regeln` + `bogen`, Frage, bei der der Charakterbogen die Top-Plätze belegt, `top_k_rerank` klein genug → der Slot holt einen Chunk aus `regeln`, nie aus `sl-regeln`.
6. **N7**: Kommentar anpassen.
7. Danach komplette Suite, erneuter Mutationscheck des Slots, Roadmap 7.2 auf ✅, dann Commit (zusammen mit der gelöschten `CONTRIBUTING.md` und `docs/testing.md`).

**Nachbesserung umgesetzt (13.10.)**:
- N1/N2: `validate_filters` lässt nur noch `$eq`, `$ne`, `$in` (ein Typ), `$and` (≥ 2 Bedingungen mit je einem Schlüssel) durch; `$or`, `$nin`, `$gt`/… → 422. Neuer Vertragstest `tests/integration/test_filter_contract.py`: jeder erlaubte Filter liefert in ChromaDB (`EphemeralClient`) und BM25 dieselben Chunks.
- N3: `_predict` merkt sich, ob *dieser* Aufruf das GPU-Modell nutzte, und wechselt unter einem `asyncio.Lock` nur einmal; spätere Fehlschläge nutzen das schon geladene CPU-Modell. Test mit zwei gleichzeitigen Fragen (Barrier) — mutationsgeprüft.
- N4: Chunk-Inspector zeigt bei Fehler nur die Fehlermeldung; das Fake-Backend der UI-Tests wirft jetzt bei Fehlercodes wie `requests` — mutationsgeprüft.
- N5: Regelwerk-Suche erst ab 3 finalen Chunks.
- N6: Positiver Sicherheitstest — der Slot holt für Spieler einen Chunk aus der freigegebenen Regel-Quelle, nie aus der Spielleiter-Regel-Quelle. Mutation (Slot-Suche ohne Anfrage-Filter) → 5 Sicherheitstests schlagen fehl.
- N7: Kommentar in `settings.yaml`.
- Komplette Suite grün.

**Aufwand**: Gering (umgesetzt).

## 7.3 Inline-Doku (Docstrings) vervollständigen — ✅ Erledigt (inkl. 7.4 D1)

**Problem**: Die Code-Doku ist uneinheitlich. Stand 13.10.: 27 Module ohne Modul-Docstring — darunter Kernmodule wie `src/retrieval/retriever.py`, `vectorstore.py`, `embeddings.py`, `src/api/routes.py`, `src/ingestion/orchestrator.py`, `chunking.py`, die Parser, `src/config/manager.py`, `src/conversation/manager.py`, `src/generation/*` und `ui/LoreKeeper.py`. 43 von 358 Funktionen/Klassen ohne Docstring, vor allem die Config-Modelle in `src/config/manager.py` (17), `src/api/eval_schemas.py` (11) und `prompt_schemas.py` (5). Viele Config-Felder haben keinen Kommentar zu Bedeutung, Einheit und Wirkungszeitpunkt.

**Lösung** (Regeln in `CLAUDE.md` → „Inline docs“): je Modul ein Docstring mit Aufgabe, Platz in der Pipeline, wichtigen Entscheidungen und Stolperfallen; Docstrings für die fehlenden Klassen/Funktionen; kurze Kommentare an den Config-Feldern; Kommentare an sicherheitsrelevanten Stellen (Spieler-Filter) und an gemessenen Zahlen mit Quelle (Roadmap-Punkt). Nur Doku, kein Verhalten ändern; Englisch wie der übrige Code. Danach prüfen, dass `ruff`/`mypy`/Tests unverändert grün sind.

**Aufwand**: Gering-Mittel (reine Doku, mehrere Dateien). Last 🟢.

**Umgesetzt (13.10.)**: Modul-Docstrings für alle 25 Module ohne (plus `src/__init__.py`): Aufgabe, Platz in der Pipeline, Stolperfallen (z.B. „CLI nur bei gestopptem Backend“, Spieler-Filter in Retriever und BM25, Cosine-Pflicht im Vectorstore). Docstrings für alle 42 fehlenden Klassen/Funktionen (Config-Modelle, Eval-/Prompt-Schemas, verschachtelte UI-Funktionen). Kommentare an allen Config-Feldern ohne (Bedeutung, Einheit, wann es greift); `retrieve()`-Docstring an den heutigen Ablauf angepasst (Deckel, Slot, Expansion). Prüfskript (AST): kein Modul und keine Funktion/Klasse in `src/`, `ui/`, `evaluation/` mehr ohne Docstring. Nur Doku, kein Verhalten geändert; Suite 632 grün.

**7.4 D1 mit erledigt** — falsche Texte korrigiert: `ConfigManager` („legacy migrations“ → Shim gibt es seit Paket B nicht mehr), `Settings`-Reihenfolge ohne `settings.local.yaml`, `logging.suppress` „WARNING“ → ERROR (`settings.yaml`, `configuration.md`), `distance_metric` (`l2`/`ip` als gleichwertig dokumentiert — nur Cosine rechnet richtig), `qa`-Hilfe der Prompts-Seite ohne `history`, `answer_history_max_chars` (kürzt jede Nachricht, nicht nur Antworten; `configuration.md`, `prompts.md`), „Language Learning Model“, deutscher Kommentar an `SourceGroup`. Außerdem in `configuration.md`: Die Default-Spalte zeigte bei `rules_slot.candidates`/`min_score` und Ollama `max_tokens`/`timeout` die Werte aus `settings.yaml` statt der Code-Defaults — jetzt beide wie bei `reranking.backend`. Die übrigen Punkte der Liste (TTL 30 s, `evaluate.py`-Usage, `window_size`-Hilfe) waren schon in A/B erledigt.

## 7.4 Code-Qualität: Dead Code, Vereinfachung, Clean Code (Review 13.10.) — ✅ Erledigt (D1 mit 7.3)

**Ergebnis des Reviews** (Agent, nur lesend; Suite grün, ruff/mypy sauber): Alle Funktionen in `src/`, `ui/`, `evaluation/` haben Aufrufer — toter Code steckt in Config-Feldern, Datenfeldern, Parametern, Zweigen und Altlasten. Komplexeste Stellen laut ruff: `ui/LoreKeeper.py:_render_sidebar` (Komplexität 29), `orchestrator.run` (19), `OllamaProvider.generate_stream` (14), `_rerank` (13), `validate_filters` (13). Geschätzt ~400–450 Zeilen weniger, wenn alle risikoarmen Punkte umgesetzt werden.

**Beim Review gefundene Fehler** (selbst nachgeprüft):
- **Chunking hängt** bei `chunk_overlap ≥ max_chunk_size`: die Wortschleifen (`fixed_size` und der Notfall-Split in `_split_with_separators`) kommen nicht voran — über die Settings-Seite einstellbar.
- **„Zum Golden Set hinzufügen“ in „Frage testen“ funktioniert nie**: der Knopf steckt im `if st.button("🔍 Testen")`-Block; sein Klick lädt die Seite neu, der äußere Block ist dann falsch (`2_Evaluation.py:358-402`).
- **„Gespeichert.“ auf der Settings-Seite ist unsichtbar**: `st.success()` direkt vor `st.rerun()` (4×; die Sources-Seite löst das mit `_flash`).
- **Datenbankfehler beim Hash-Abgleich wird verschluckt** (`get_content_hashes_for_source` → `{}`): die Ingestion hält dann alle Dateien für neu, alte Chunks geänderter Dateien und verwaiste Chunks bleiben liegen. `wipe_collection` meldet „Wiped“ auch bei Fehlern.
- **Eval-Polling** (`_poll_eval_job`) läuft bei totem Backend endlos (Sources-Seite hat eine Fehlergrenze).
- **End-to-End-CLI** schreibt `eval_*.json` — die Aufräumlogik löscht nur `e2e_*`; die Treffer-Logik von `evaluate.py` ignoriert `expected_source_ids` (abgedriftet von `evaluate_retrieval.py`).
- **Doku/Hilfetexte falsch**: u.a. „persistiert nach settings.yaml“ (ist `settings.local.yaml`), „TTL = 30s“ (Schleife 60 s), `evaluate.py`-Usage mit nicht existierendem `--settings`, `logging.suppress` „auf WARNING“ (setzt ERROR), Hilfe zu `window_size` („auch für den LLM-Prompt“ — nein), `qa`-Hilfe ohne `history`, `distance_metric: l2|ip` dokumentiert, aber nur Cosine rechnet richtig.

**Korrekturplan** (vier Pakete, je eigener Commit):

*Paket A — Fehler* (zuerst):
1. `ChunkingConfig`: Validator `chunk_overlap < max_chunk_size` (422 über `/config`, Fehlermeldung in der UI); Wortschleifen zusätzlich mit `max(1, size - overlap)` absichern. Test.
2. „Zum Golden Set hinzufügen“ in „Frage testen“: Vorschau-Ergebnis in `st.session_state` halten, damit der Knopf wirkt (oder Block entfernen — **Entscheidung**). UI-Test.
3. Settings-Seite: ein lokales `_save(...)` statt 4 Kopien (Diff → PUT → Cache leeren → Flash → Rerun), „Gespeichert.“ über `_flash` sichtbar; Strategieliste als Konstante.
4. `get_content_hashes_for_source` wirft bei DB-Fehlern (Job endet mit Fehler statt still falsch); `list_files` daraus ableiten; `wipe_collection` nur „nicht gefunden“ abfangen. Tests.
5. `_poll_eval_job` mit Fehlergrenze wie die Sources-Seite.
6. Evaluation: ein `load_qa_pairs`, ein `source_hit` (mit `expected_source_ids`), ein `_save_report`; E2E-CLI schreibt `e2e_*.json`.

**Paket A umgesetzt (13.10.)** — Entscheidung A2: reparieren:
1. `ChunkingConfig` lehnt `chunk_overlap ≥ max_chunk_size` ab (Speichern → 422 mit Begründung, nichts geschrieben); beide Wortschleifen rücken zusätzlich mindestens ein Wort vor. Tests: Validator, `save_settings`, Schleifen mit ungültiger Config (`model_construct`).
2. „Frage testen“ hält das Ergebnis im Session-State; „➕ Hinzufügen“ wirkt jetzt, doppelte IDs werden abgelehnt. UI-Test (`tests/ui/test_question_preview.py`).
3. Settings-Seite: ein `_save(...)` statt vier Kopien, „Gespeichert.“ über `_flash` sichtbar, Backend-Begründung bei 422 in der Fehlermeldung; Hilfetexte zu `window_size` und Min-Chunk-Size korrigiert. UI-Tests (`tests/ui/test_settings.py`, mutationsgeprüft).
4. `get_content_hashes_for_source` wirft bei DB-Fehlern (Ingest-Job endet mit Fehler, nichts geschrieben); `list_files` daraus abgeleitet; `wipe_collection` fängt nur `NotFoundError`. Tests.
5. `_poll_eval_job` gibt nach 45 Fehlversuchen in Folge mit Hinweis auf. UI-Test.
6. `load_qa_pairs`, `source_hit`, `save_report` einmal in `evaluation/evaluate_retrieval.py`, genutzt von CLI, E2E-Skript und `/eval`-API; E2E-CLI schreibt `e2e_*.json`; Usage von `evaluate.py` korrigiert. Tests.

Suite: 620 grün (vorher 599).

*Paket B — Toter Code* (Verhalten unverändert, außer markiert):
1. Ungelesene Config: `ingestion.watch_for_changes`, `api.*`, `ui.*`, `vectorstore.provider` samt YAML-Blöcken und Doku.
2. Client-Modus von ChromaDB (`HttpClient`, `chroma_host`/`chroma_port`) — nie genutzt.
3. Migrations-Shim `ingestion.document_paths` — **Entscheidung** (alte Installationen ohne `sources.yaml` starten dann ohne Quellen statt migriert).
4. Tote Felder/Parameter: `LLMResponse.raw_response` (Ollama kopiert pro Aufruf die ganze Antwort), `latency_ms`-Buchhaltung, `Message.timestamp`, `_health_cache["ts"]`, ungenutztes `source_path`, `CALLOUT_CONTINUATION`, nie genutzte `**kwargs`-Overrides in den Providern, `if pairs or True`.
5. Altlast in `recategorize._match_source` (+ `source_collection`) — erst bei gestopptem Backend prüfen, ob noch Chunks ohne `source_id` im Index sind.
6. Endpoints ohne Aufrufer: `GET /provider/gemini/status` (Daten stecken in `/sidebar-state`), `GET /provider` (nur Tests) — **Entscheidung** (öffentliche API behalten oder entfernen).
7. **Bildpfad** (Bild-Chunks erreichen nie eine Antwort, PDF-Bilder zeigen auf gelöschte Temp-Dateien): nicht entfernen — Roadmap 2 baut darauf auf. Jetzt nur falsche Docstrings/Kommentare korrigieren und den Temp-Pfad in Roadmap 2 vermerken.

**Paket B umgesetzt (13.10.)** — Entscheidungen: `document_paths`-Shim entfernt, die zwei Endpoints bleiben, Bildpfad bleibt:
1. Entfernt: `ingestion.watch_for_changes`, `api.*` (`APIConfig`), `ui.*` (`UIConfig`), `vectorstore.provider` samt YAML-Blöcken; alte YAML-Dateien laden weiter (`extra="ignore"`).
2. ChromaDB nur noch eingebettet: Client-Modus (`HttpClient`, `mode`, `chroma_host`, `chroma_port`) entfernt; `/config` zeigt im Nur-lesen-Block nur die Collection (keine Pfade in der API).
3. Migrations-Shim `ingestion.document_paths` entfernt — ohne `sources.yaml` startet LoreKeeper ohne Quellen (Warnung).
4. Tote Felder/Parameter: `LLMResponse.raw_response` (Ollama kopierte pro Aufruf die ganze Antwort), `latency_ms`-Buchhaltung, nie genutzte `**kwargs`-Overrides der Provider (Werte jetzt direkt aus der Config), `Message.timestamp`, `_health_cache["ts"]` (+ falscher Kommentar „TTL = 30s“), Parameter `source_path` in `_extract_image_docs`, `CALLOUT_CONTINUATION`, `if pairs or True`.
5. `recategorize._match_source` nur noch über `source_id` (Index geprüft 13.10.: 0 von 4745 Chunks ohne `source_id`); `source_collection` wird nicht mehr geschrieben (bestehende Chunks behalten das Feld, stört nicht).
6. Bildpfad: falsche Kommentare/Docstrings korrigiert (Bild-Chunks werden heute nie gefunden); Hinweis in Roadmap 2.

Suite: 618 grün (zwei Tests der alten Zuordnungs-Heuristik entfernt).

*Paket C — Vereinfachungen* (gleiches Verhalten, je Testabdeckung prüfen):
1. `ConfigManager.editable_snapshot` aus `_EDITABLE_KEYS` ableiten statt zweiter Handbaum (−50 Zeilen, Sync-Gefahr weg).
2. `_get_services()`-7-Tupel, `_get_retriever()`, `_get_vectorstore()` → überall `main_module.<dienst>` (−25).
3. Chat-Seite: „Neue Session“ nutzt `_reset_chat()`, ein `_render_extras(msg)` für Verlauf und neue Antwort, ein Gemini-Key-Block (**Entscheidung**: leere Eingabe löscht den Laufzeit-Key oder wird abgelehnt), ein Sidebar-State-Helfer — senkt die Komplexität von `_render_sidebar`.
4. Evaluation-Seite: ein Fehler-Handler statt 3, ein `_flipped(...)` statt 2, `_csv(...)`; Sources-Seite: Spaltenkonfiguration und `category_map`-Umwandlung je einmal (die beiden Kopien sind abgedriftet), „Recategorize (alle)“ einmal statt pro Quelle.
5. `_spawn_eval_task` = `_spawn_background` wiederverwenden; Prompt-Routen mit `_templates()`/`_variant_from_raw()`; Chunking mit einer Strategie-Tabelle und gemeinsamem Wort-Split; Retriever-Kleinigkeiten (`_to_chunk` in der Expansion, Backfill per Slice); `IngestionResult` als Dataclass; Provider-Doppelungen (Ollama-Nachrichten, Gemini-Key-Auflösung); `asyncio.get_event_loop().run_in_executor` → `run_in_threadpool`.
6. Tests: `tests/integration/conftest.py` mit einem session-weiten Embedding-Service (Modell nur einmal laden) und einem `parse_sse()`-Helfer.

**Paket C umgesetzt (13.10.)** — Entscheidung C3: leere Eingabe wird abgelehnt, eigener Knopf „UI-Key entfernen“:
1. `editable_snapshot` = `_EDITABLE_KEYS` + `_READ_ONLY_KEYS` (Ausgabe gegen den alten Stand verglichen: identisch).
2. `_get_services()`, `_get_retriever()`, `_get_vectorstore()`, `_spawn_eval_task` entfernt — Handler lesen `main_module.<dienst>`, Eval-Jobs nutzen `_spawn_background`. Hintergrundjobs laufen über `asyncio.to_thread` statt `run_in_executor` (`run_in_threadpool` hing im TestClient).
3. Chat-Seite: „Neue Session“ = `_reset_chat()`, `_render_extras(msg)` für Verlauf und neue Antwort, ein Gemini-Key-Block (leere Eingabe abgelehnt, „UI-Key entfernen“ nur bei Laufzeit-Key, Bestätigung über Flash), `_sidebar_state()` statt dreimal try/except; Docstring von `_fetch_sidebar_state` korrigiert. `_render_sidebar` bleibt mit Komplexität 29 der komplexeste Block (Provider-Wechsel, Status, Filter) — weiteres Aufteilen wäre reines Verschieben.
4. Evaluation-Seite: `_show_eval_error` (409 → Warnung) statt 3 Kopien, `_flipped(...)` statt 2, `_csv(...)` statt 4. Sources-Seite: `_FOLDER_MAP_COLUMNS`, `_cmap_from_rows` als eine Regel (vorher schrieb „Zuordnung speichern“ jeden Ordner in die `category_map`, „Neue Source“ nur abweichende — gleiches Ergebnis, jetzt einheitlich nur abweichende), ein „🏷 Recategorize“ oben neben den Indizier-Knöpfen statt einer pro Quelle.
5. Prompt-Routen mit `_templates()`/`_variant_from_raw()`; Chunking mit Strategie-Tabelle, `_word_split` und `_split_prose` (Ausgabe auf 2.018 echten Dokumenten, alle Strategien, gegen den alten Stand verglichen: identisch); Retriever (`_to_chunk` in der Expansion, Backfill per Slice); `IngestionResult` als Dataclass; `_messages()` in Ollama, `_resolve_api_key()` in Gemini; `_ensure_ollama` mit `time.monotonic()`; `_PROGRESS_FIELDS` im Ingest-Fortschritt.
6. `tests/integration/conftest.py`: session-weites `embed_service` (Modell einmal statt dreimal geladen), `parse_sse()` statt 5 Kopien. Neue UI-Tests `tests/ui/test_simplified_pages.py` (Gemini-Key, Neue Session, Ordner-Zuordnung, Recategorize, Ergebnisvergleich); Tests für `_templates`/`_variant_from_raw`.

Suite: 626 grün (vorher 618).

*Paket D — Clean Code*:
1. Falsche Docstrings, Kommentare, Hilfetexte (Liste oben) — zusammen mit 7.3.
2. Nutzer sichtbare `detail`-Meldungen der API einheitlich deutsch (heute gemischt), Logs bleiben englisch.
3. Ein Begriff, eine Stelle: `access.Role` statt mehrfach `Literal[...]`/`str`; `player_source_ids()` statt Inline-Kopie in `/sidebar-state`; `/sidebar-state` nutzt `health()`; `_raise_for_status` (zeigt `detail` des Backends) für alle Seiten statt nur Sources; Variantennamen URL-kodieren; Lösch-Aufrufe prüfen den Status.
4. Benannte Konstanten: 3,5 Zeichen/Token (2×), Chroma-Batch 5000 (2×), `max_length=512` (2×), Health-Intervall, Vorschau-Längen. Eval-Defaults 15/8/3 hart kodiert an 4 Stellen → `None` = Config-Wert (**Entscheidung**: ändert Verhalten, wenn `settings.yaml` abweicht).
5. Kein Griff in private Attribute anderer Module: `VectorStoreService.get_all()`, `ConfigManager.prompts_path`, `_source_base` als Modulfunktion.
6. `Literal` statt Handprüfung für `provider`/`eval_type` — **Entscheidung** (Statuscode 400 → 422).

**Paket D umgesetzt (13.10.)** — Entscheidungen D4 (Config-Wert) und D6 (422):
2. API-Meldungen (`detail`, dazu `error`/`note` von `/sources/…/folders` und `/sources/scan`) durchgehend deutsch; Logs bleiben englisch.
3. `access.Role` in `QueryRequest`, `RetrievalPreviewRequest` und `FeedbackRequest` (dort vorher freier String, jetzt `player`/`game_master`, Default `game_master` wie bei `/query`); `/sidebar-state` nutzt `player_source_ids()` und `health()`. Neu `ui/backend.py` mit `raise_for_status` → `BackendError` (Statuscode + `detail` des Backends, bei 422 nur die Meldungen) auf allen Seiten statt nur Sources; die Settings-Seite nutzt es statt ihrer eigenen Kopie, die Evaluation-Seite erkennt 409 darüber. Variantennamen URL-kodiert; Löschen von Varianten und Eval-Ergebnissen prüft den Status (vorher sah ein Fehlschlag wie Erfolg aus).
4. Benannte Konstanten: `CHARS_PER_TOKEN` (Chunking + Chat-Verlauf), `_CHROMA_BATCH`, `_RERANK_MAX_TOKENS`, `_HEALTH_INTERVAL_S`, `_SOURCE_PREVIEW_CHARS`/`_PREVIEW_CHARS`. Eval-Parameter `None` = Wert aus `settings.yaml` in `/eval/preview`, `/eval/run` und `run_evaluation*`; der Bericht nennt die tatsächlich benutzten Werte. Die Regler der Evaluation-Seite starten bei den Config-Werten (`GET /config`, gecacht; Budget der Seite 4 → 5). Nebenbei behoben: Das CLI `evaluate_retrieval` lief immer mit `max_per_source=3` — `RETRIEVAL__RERANKING__MAX_PER_SOURCE` wirkte nicht.
5. `VectorStoreService.get_all()` statt `_get_collection()` in BM25 und Recategorize; `ConfigManager.prompts_path`; `source_base()` als Modulfunktion im Orchestrator.
6. `provider` und `eval_type` als `Literal` → 422 vor jeder Änderung (Handprüfungen entfernt).

Tests: `tests/ui/test_backend_errors.py` (Regler-Defaults, 409 als Warnung, fehlgeschlagenes Löschen, URL-Kodierung — gegen die alten Varianten mutationsgeprüft), Eval-Defaults in `tests/unit/test_evaluate_retrieval.py`, 422-Fälle in den Integrationstests; das Fake-Backend der UI-Tests kennt jetzt `DELETE`. Suite: 632 grün (vorher 626).

Offen: `recategorize` importiert weiterhin das private `_resolve_category` aus dem Orchestrator (im Review nicht genannt).

**Aufwand**: Paket A gering-mittel, B gering, C mittel, D gering. Last 🟢 (nur Code + Tests).

## 8.0 Was passiert wenn man an den Server mehrere Anfragen gleichzeitig bzw. nahe zu glecihzeitig schickt?
- Relevant wenn ich einen Abend mache und sowohl ich als auch meine spieler Lorekeeper nutzen wollen.

**Bewertung (13.10.)**: Impact Mittel-Hoch — das ist der Einsatzfall am Spielabend, und bisher testet nichts gleichzeitige Anfragen. Aus dem Code bekannt: Ingest- und Eval-Jobs sind gegen Parallelstart gesperrt (409); Embedding und Rerank laufen im Threadpool; ein Ingest-Job läuft in einem Thread **neben** laufenden Fragen auf derselben ChromaDB; ein Provider-Wechsel ersetzt `main_module.provider/generator` mitten in laufenden Anfragen. Zu prüfen: (1) Antwortzeit bei 2–4 gleichzeitigen Fragen — arbeitet Ollama Anfragen nacheinander ab (`OLLAMA_NUM_PARALLEL`), wartet die zweite Frage die ganze erste Generierung; (2) Rerank gleichzeitig auf einer DirectML-Session; (3) Fragen während eines Ingests (BM25 wird erst am Jobende invalidiert); (4) Sessions zweier Rollen gleichzeitig. Vorgehen: Lasttest mit 2–4 parallelen Streams gegen das laufende Backend (eigener `perf`-Test), Messwerte festhalten, erst dann entscheiden, ob z.B. eine Warteschlange oder ein Hinweis in der UI nötig ist.

**Aufwand**: Gering-Mittel (Messung), Fixes je nach Befund. Last 🟡 (Backend + LLM).

## 9.0 Testabdeckung prüfen

**Messung (13.10., `pytest --cov=src --cov=ui --cov=evaluation`)**: gesamt 85 %. `src/` fast überall ≥ 88 %, Eval- und Prompt-Routen 100 %. Lücken: `evaluation/evaluate.py` 15 % (E2E-Skript), `ui/pages/3_Prompts.py` 53 %, `ui/pages/2_Evaluation.py` 55 %, `ui/pages/1_Sources.py` 69 %, `src/main.py` 71 % (Lifespan), `ui/LoreKeeper.py` 78 %.

**Vorgehen**: Nicht auf eine Prozentzahl optimieren, sondern die ungetesteten Pfade der Admin-Seiten ansehen und die mit echtem Fehlerrisiko testen (Speichern/Aktivieren von Varianten, A/B-Lauf, E2E-Tab, Ergebnisvergleich); `evaluate.py` mit gemocktem HTTP. Lifespan nur, wo Logik steckt (Seed des Health-Caches, Aufräumen beim Beenden).

**Aufwand**: Gering-Mittel. Last 🟢.

## 10.0 Logging prüfen und wenn nötig erweitern
- sind alle wichtigen events geloggt
- sind alle wichtigen fehler geloggt
- sind unnötige dinge geloggt
- unterschied zwischen normalem und debug logging gemacht?

**Bewertung (13.10.)**: Impact Mittel — am Spielabend ist das Log die einzige Quelle, um einen Fehler hinterher zu verstehen. Bekannt: Fehlerpfade loggen teils ohne Traceback (`logger.error(f"…{e}")` statt `exc_info`), z.B. Streaming- und Eval-Fehler; ein fehlgeschlagener Ingest-Job schreibt den Fehler nur in den Job-Status, nicht ins Log. Vorgehen: alle `logger.`-Aufrufe und `except`-Zweige durchgehen (Agent-Review mit Fokus „error handling“), Ergebnis als Liste hier, dann Fixes in einem Paket.

**Aufwand**: Gering. Last 🟢.
## 11.0 Architektur-Review: Passt der Aufbau noch, was sollte modularer werden?

**Wann**: nach 10.0 und der Messung aus 8.0, **vor** 9.0 und vor dem Wiederaufnehmen der Features. Begründung:
- 8.0 kann Umbauten erzwingen (z.B. atomarer Provider-Wechsel, Job-Verwaltung, Warteschlange fürs LLM) — die gehören in denselben Entwurf, nicht in einen zweiten Umbau danach.
- 9.0 danach, weil die größten Abdeckungslücken genau in den Teilen liegen, die ein Umbau anfassen würde (Admin-Seiten, `main.py`). Tests für Code, der gleich umgebaut wird, wären doppelte Arbeit; die bestehende Suite (632 Tests) sichert den Umbau ab.
- Vor den Features, weil Bilder (2), Wikilink-Expansion (3.5) und Internet-Zugriff (4.2) neue Suchwege bzw. Zugriffswege bringen — genau dort muss der Spieler-Filter heute an jeder Stelle von Hand mitgegeben werden.
- 10.0 davor ist unabhängig und billig; ordentliches Logging hilft auch bei 8.0.

**Beobachtungen, die das Review prüfen soll** (Stand 13.10., gezählt bzw. aus dem Code, nicht bewertet):
- **Dienste als Modul-Globale in `src.main`**: 35 Stellen holen sie per `import src.main as main_module`. Tests müssen `main_module` patchen; ein Provider-Wechsel ersetzt `provider`, `condense_provider` und `generator` einzeln (nicht atomar, relevant für 8.0). Alternative: ein Dienste-Container im `app.state` bzw. per FastAPI-Dependency.
- **`src/api/routes.py` mit 722 Zeilen** für Query, Ingest-Jobs, Quellen, Provider, Config, Health und Feedback — Kandidat für eigene Router je Bereich (wie schon `eval_routes`/`prompt_routes`).
- **Job-Verwaltung doppelt**: `_ingest_jobs` (routes) und `_eval_jobs` (eval_routes), beide In-Memory-Dicts ohne Aufräumen; `eval_routes` importiert das private `_spawn_background` aus `routes`.
- **Spieler-Filter als Konvention**: jeder neue Suchweg muss den Filter selbst durchreichen (CLAUDE.md, Sicherheitstests). Prüfen, ob eine gefilterte Sicht auf den Vectorstore/BM25 das strukturell erzwingen kann.
- **`retriever.py` (469 Zeilen)** vereint Suche, Fusion, Rerank (inkl. GPU-Fallback), Deckel, Regelwerk-Slot und Expansion — Kandidat für eigene Bausteine (z.B. Reranker, Nachbearbeitungs-Schritte).
- **UI**: `2_Evaluation.py` 859 Zeilen, Chat und Sources je ~550; 52 direkte `requests`-Aufrufe über alle Seiten. `ui/backend.py` ist der Anfang eines API-Clients — prüfen, ob alle Aufrufe dorthin gehören (ein Ort für URLs, Timeouts, Fehler).
- **`ConfigManager`** lädt Settings, Quellen und Prompts und schreibt alle drei; Prompt-Varianten verwaltet dagegen `prompt_routes` direkt auf der Platte.
- Kleinere Kopplungen über private Namen: `recategorize` → `_resolve_category`, `main` → `_health_cache`/`_health_loop`.

**Vorgehen**: Review über `.claude/agents/code-reviewer.md` mit Fokus „architecture“ über die ganze Codebasis → Befunde bewerten (was bringt Nutzen bei einem Einzelnutzer-Projekt, was ist nur Lehrbuch) → Umbauplan in Paketen hier eintragen → erst nach Zustimmung umsetzen. Maßstab aus CLAUDE.md: keine Abstraktion für Einmal-Logik; ein Umbau muss ein konkretes Problem lösen (Testbarkeit, Sicherheit des Spieler-Filters, Nebenläufigkeit, Dateigröße/Übersicht).

**Aufwand**: Review Gering; Umbau je nach Ergebnis Mittel. Last 🟢.

---

## Priorisierungsvorschlag

Erledigt: 7.3 Inline-Doku, 7.4 Pakete A–D (Fehler, toter Code, Vereinfachung, Clean Code), 7.2 Code-Review + Nachbesserung, 3.10 Regelwerk-Slot, 5.4 Reranker auf der GPU (DirectML), 5.2 Startup-Baseline, 3.3 Chunk-Inspector, 3.6 Chat-Verlauf im Antwort-Prompt, 3.4 Antwort-Feedback, Spieler-Modus im Backend (aus 4.2), 3.8/3.9 Ingest-Fortschritt, 7.0/7.1 Testlücken + Eval-CLI-Fix, 5.3 UI-Reste (`/config`, `/folders`), 3.11 Spieler-/Spielleiter-Modus (inkl. Sicherheitstests), 4.1 Heimnetz + 3.7 Zugriffsschutz, 1 PDF-Parsing, 3.1 Hybrid Search, 5.1 Stage-Timings, 5.2 Performance-Tests (+ `localhost`-Fix), 5.3 Antwortzeit (Qwen3-Thinking aus, Kontext 16k, Keep-alive 1 h, Section Expansion `top_rank`).

**Offen am Gaming-PC (Nutzer)**: alte Autostart-Aufgabe „LoreKeeper Server“ als Admin entfernen (`Unregister-ScheduledTask -TaskName "LoreKeeper Server" -Confirm:$false`); DHCP-Reservierung für `192.168.178.34` im Router.

**Last** = Rechenlast beim Umsetzen/Messen: 🟢 nur Code + Unit-/UI-Tests (geht parallel zur Simulation) · 🟡 braucht laufendes Backend oder LLM · 🔴 Indizierung oder Vision-Modell (CPU/GPU, mehrere Minuten, nicht parallel zur Simulation). Golden-Set-Läufe sind seit dem GPU-Reranker (5.4) ~1 min und belasten die CPU nur noch mit dem Query-Embedding → 🟡.

**Fokus ab 13.10.: Qualität vor neuen Features.** Features (3.2, 2, 4.2, 3.5) ruhen, bis die Qualitätspunkte erledigt sind.

| Reihenfolge | Punkt | Impact | Aufwand | Last | Begründung |
|---|---|---|---|---|---|
| 1 | 10.0 Logging prüfen | Mittel — Fehler am Spielabend hinterher nachvollziehbar | Gering | 🟢 | Fehler ohne Traceback, Ingest-Fehler nicht im Log; Review mit Fokus „error handling“ |
| 2 | 8.0 Gleichzeitige Anfragen | Mittel-Hoch — Spielabend mit mehreren Nutzern | Gering-Mittel | 🟡 | Ungetestet; erst messen (2–4 parallele Streams), dann entscheiden |
| 3 | 11.0 Architektur-Review | Mittel — Grundlage für 9.0 und die ruhenden Features | Gering (Review), Umbau Mittel | 🟢 | Nach 8.0 (dessen Befunde fließen ein), vor 9.0 (keine Tests für Code, der gleich umgebaut wird) und vor neuen Suchwegen |
| 4 | 9.0 Testabdeckung | Gering-Mittel — Lücken in den Admin-Seiten und `evaluate.py` | Gering-Mittel | 🟢 | 85 % gesamt; gezielt Pfade mit Fehlerrisiko, keine Prozentjagd |
| 5 | Offene Prüfungen im laufenden Betrieb | Mittel — bestätigt, dass die Neuerungen im echten Chat wirken | Gering | 🟡 | Mit gestartetem LoreKeeper: 3.6 Folgefragen, 👍/👎 + Tabs „Feedback“/„Chunks“, Rerank-Timings (5.4), Regelwerk-Slot an x002/x005 (3.10), `pytest -m live`; gut mit 8.0 kombinierbar |
| 6 | 5.5 Vektorsuche reproduzierbar machen | Gering-Mittel — verlässliche Eval-Vergleiche | Gering-Mittel | 🟡 | ±2 Abschnitte Schwankung zwischen Prozessen; `ef_search` erhöhen oder exakt suchen |
| 7 | 5.3 Gesamtbudget Section Expansion | Gering-Mittel — max. 49 Chunks pro Frage | Gering | 🟡 | Antwortqualität/Prompt-Größe; mit der Chunk-Prüfung aus 5.2 messbar |
| 8 | 6.0 PnP-Simulation — Abschluss | Mittel — Balance-/Begegnungsfragen im Chat | Gering | 🔴 | Wartet auf einen sauberen Simulationslauf (extern); dann Re-Index, Eval, Spieler-Messung |
| — | 3.2 Session-History — ⏸ | Unklar | Mittel | 🟢 | Feature, ruht; erst entscheiden, ob gespeicherte Sessions Nutzen haben |
| — | 2 Bilder im Chat (Phase 1/2) — ⏸ | Mittel / Nice-to-have | Mittel / Mittel-Hoch | 🔴/🟡 | Feature, ruht |
| — | 4.2 Internet-Zugriff (Tailscale) — ⏸ | Mittel | Gering | 🟢 | Feature, ruht; sinnvoll erst nach 8.0 |
| — | 3.5 Wikilink-Expansion — ⏸ | Mittel | Mittel | 🔴 | Feature, ruht; im Spieler-Modus darf sie keine SL-Quellen nachladen |
