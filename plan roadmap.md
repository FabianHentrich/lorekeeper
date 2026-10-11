# LoreKeeper — Roadmap & Bearbeitungsliste

Stand: 2026-10-11

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

### 3.7 Zugriffsschutz für die UI

**Problem**: Die API selbst ist nicht exponiert — uvicorn bindet auf `127.0.0.1`, und alle API-Aufrufe macht der Streamlit-Prozess serverseitig. Exponiert ist die **Streamlit-UI**: sie lauscht standardmäßig auf allen Interfaces und hat keinen Login. Wer im selben Netz `http://<pc>:8501` öffnet, kann über die Admin-Seiten indizieren, die Collection löschen (`/admin/wipe`), Config und Gemini-Key ändern. Zusätzlich lässt das Feld „API URL“ in der Sidebar den Streamlit-Server beliebige URLs abfragen.

**Lösung**:
- Passwort-Gate (Secret aus `.env`) vor den Admin-Seiten (Sources, Settings, Prompts, Evaluation); der Chat bleibt frei oder hängt optional am selben Gate
- „API URL“-Feld durch Config-Wert ersetzen (nicht im Browser editierbar)
- Backend bleibt auf `127.0.0.1` — ein API-Key wird erst nötig, wenn die API selbst je exponiert wird

**Aufwand**: Gering. Teil von 4.1.

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

## 4.0 Zugriff auf LoreKeeper aus dem Netzwerk und dem Internet

**Problem**: Der Laptop ist zu schwach für ein lokales LLM. LoreKeeper soll auf dem Gaming-PC laufen (Ollama + Backend + UI) und vom Laptop aus bedient werden — zuerst im Heimnetz, später ggf. auch unterwegs. Die Alternative, nur noch Gemini (Bezahlvariante) zu nutzen, hat ein Kostenrisiko: Budgets in Google Cloud senden nach meinem Kenntnisstand nur Warnungen und stoppen nicht automatisch; begrenzen lässt sich das Risiko nur über heruntergesetzte Quotas (Requests/Minute bzw. /Tag) oder eine selbstgebaute Budget→Billing-aus-Automatik. Gemini bleibt deshalb Fallback, nicht Hauptweg.

**Architektur-Befund**: Der Browser spricht nur mit Streamlit (Port 8501); alle API-Aufrufe laufen serverseitig im Streamlit-Prozess gegen `127.0.0.1:8000`. Es muss also **nur Port 8501** erreichbar sein, das Backend bleibt lokal.

### 4.1 Heimnetz (LAN) — jetzt

**Lösung**:
1. **Erreichbarkeit**: Streamlit lauscht bereits auf allen Interfaces. Windows-Firewall-Regel nur für Port 8501 und nur im Netzwerkprofil „Privat“; feste IP für den Gaming-PC per DHCP-Reservierung im Router. Aufruf vom Laptop: `http://<gaming-pc-ip>:8501`.
2. **Server-Modus in `start.ps1`**: Schalter (z.B. `-Server`) startet uvicorn ohne `--reload` (Code-Änderungen starten den Server nicht mitten in einer Session neu) und öffnet keinen Browser auf dem PC.
3. **Autostart**: `start.ps1 -Server` als Aufgabe in der Aufgabenplanung bei Anmeldung; Ollama startet als Windows-Dienst ohnehin mit.
4. **Zugriffsschutz**: 3.7 (Passwort-Gate für Admin-Seiten, API-URL-Feld sperren) — Voraussetzung, sobald der Port offen ist.
5. **Doku**: Abschnitt „Betrieb im Heimnetz“ in `ARCHITECTURE.md` (Deployment).

**Aufwand**: Gering (inkl. 3.7).

### 4.2 Internet — später

**Lösung**: **Tailscale** (privates Mesh-VPN, kostenlos für Privatnutzung) auf Gaming-PC und Laptop → `http://gaming-pc:8501` funktioniert von überall, ohne Portfreigabe im Router und ohne öffentliche Erreichbarkeit. Alternative: Cloudflare Tunnel + Cloudflare Access (Login-Seite davor) — mehr Einrichtung.

**Offener Punkt**: Der Gaming-PC muss laufen. Wake-on-LAN von unterwegs braucht ein immer laufendes Gerät im Heimnetz (Router-Funktion oder Raspberry Pi); sonst bleibt der PC an.

**Aufwand**: Gering (Einrichtung, kaum Code).

## 5.0 Performance-Optimierung und Performance-Tests

**Ziel**: Bei Weiterentwicklungen sofort sehen, wenn etwas deutlich ineffizienter wird — vor allem bei der Antwortzeit und der UI-Reaktion, beim Start nur einfache Hebel.

**Problem**: Es wird nichts gemessen. Bekannt ist nur aus Stichproben: qwen3:14b ~17 s bis zum ersten Token (mit 29 Chunks im Kontext, weil die Section Expansion den ganzen Krieger-Abschnitt nachgeladen hat), Retrieval ~1 s, Start ~12 s (Modelle laden). Ob die Ollama-Kontextgröße (`num_ctx`) den Prompt abschneidet, ist ungeprüft.

### 5.1 Messen: Stage-Timings — ✅ Erledigt

Pro Anfrage loggen und im `done`-Event mitliefern: Condense, Retrieval (Embedding / Vektorsuche / Rerank / Expansion), Zeit bis zum ersten Token, Gesamtzeit, Anzahl Chunks und Prompt-Größe (Zeichen bzw. `usage.tokens_in`). Grundlage für alles Weitere; die UI kann die Werte optional in der Caption zeigen.

**Aufwand**: Gering.

**Umgesetzt**: `timings` in `/query` und im `done`-Event (`condense_ms`, `retrieval_ms` + `retrieval_stages_ms` {embed, search, bm25, rerank, expand}, `first_token_ms`, `total_ms`, `chunks`, `prompt_chars`), eine Log-Zeile `Query timings: …` pro Anfrage, Caption „⏱ 1. Token … · gesamt … · Suche … · N Chunks“ unter jeder Antwort.

**Erste Messung (qwen3:14b, warm, GPU)**: Retrieval 0,6–0,8 s (davon Rerank ~0,5–0,75 s), erster Token nach 8,7–9,9 s, gesamt 14–20 s. → Der große Block vor dem ersten Token ist **nicht** Retrieval, sondern verstecktes Denken (siehe 5.3).

### 5.2 Performance-Tests (Regressionsschutz)

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

### 5.3 Bekannte Hebel (nach der Messung angehen)

- **Antwortzeit — qwen3 denkt trotz `/no_think` (gemessen, größter Hebel)**: Ollama ignoriert das an den Prompt gehängte `/no_think`; qwen3 erzeugt Reasoning-Tokens im separaten Feld `reasoning`, das LoreKeeper weder anzeigt noch zählt (`tokens_thinking` bleibt 0). Kostet ~7 s pro Antwort. Direkter Test gegen Ollama: `reasoning_effort: "none"` (bzw. `reasoning: {effort: "none"}`) schaltet es ab (0 Reasoning-Chunks), `think: false` wird ignoriert. ✅ **Erledigt**: `OllamaProvider` sendet für Qwen3 `extra_body={"reasoning_effort": "none"}` statt `/no_think`; Reasoning-Chunks im `reasoning`-Feld werden als `tokens_thinking` gezählt.
- **Antwortzeit — Kontextfenster (gemessen)**: `ollama ps` zeigt `CONTEXT 4096` für qwen3:14b. Die Krieger-Frage brauchte 3621 Eingabe- + 919 Ausgabe-Tokens — am bzw. über dem Limit; bei größerer Section Expansion wird der Prompt-Anfang (Systemprompt) still abgeschnitten. Fix: `OLLAMA_CONTEXT_LENGTH` (z.B. 8192–16384, VRAM prüfen) oder `num_ctx` über die native API.
- **Antwortzeit**: Section Expansion begrenzen (nur Abschnitte mit Top-Treffern, `max_per_source` respektieren); unnötige Condense-Aufrufe sind bereits entfernt (`needs_condense`).
- **UI-Reaktion**: Chat holt vor jeder Nachricht `/config` (cachen); Sources-Seite blockiert beim Polling (3.8/3.9); Kosten eines Reruns mit langer Chat-Historie und Quellen-Expandern prüfen.
- **Start**: nur einfache Hebel — Reranker/BM25 erst bei erster Anfrage laden (lazy), `--reload` im Server-Modus aus (4.1).

**Aufwand**: je Hebel Gering bis Mittel.


## 6.0 PnP Simulation zugreifbar machen. 
 reichen die HTML-Dateien als zugriff? und welche daten sollte ich dem LLM geben? wie wird das ganze indexiert?

---

## Priorisierungsvorschlag

Erledigt: 1 PDF-Parsing, 3.1 Hybrid Search.

| Reihenfolge | Punkt | Impact | Aufwand | Begründung |
|---|---|---|---|---|
| 1 | ~~5.1 Stage-Timings~~ | | | ✅ Erledigt |
| 2 | 5.3 Hebel Antwortzeit (qwen3-Thinking aus, `num_ctx`, Section Expansion) | Hoch — ~7 s weniger pro Antwort, kein abgeschnittener Prompt | Gering | Gemessen, Fix ist klein |
| 3 | 4.1 Heimnetz + 3.7 Zugriffsschutz | Hoch — Gaming-PC als Server nutzbar | Gering | 3.7 muss vor dem Öffnen des Ports stehen |
| 4 | 5.2 Performance-Tests | Mittel — Regressionsschutz | Mittel | Deterministische Prüfungen zuerst, zeitbasierte danach |
| 5 | 3.8 + 3.9 Ingest-Job / Fortschritt | Mittel — Sources-Seite bleibt bedienbar | Gering-Mittel | Zusammen, Teil der UI-Reaktion |
| 6 | 3.10 Regelwerk-Slot im Retrieval | Mittel — Vault↔Regel-Fragen | Mittel | Mit Abschnittsprüfung messbar |
| 7 | 3.6 Chat-Verlauf im Antwort-Prompt | Mittel — bessere Folgefragen | Gering-Mittel | Kontextlänge nach 5.3 im Griff |
| 8 | 3.4 Antwort-Feedback | Gering-Mittel | Gering | Quick Win |
| 9 | 2 Bilder im Chat (Phase 1: Beschreibung) | Mittel — Karten/Handouts suchbar | Mittel | |
| 10 | 3.2 Session-History | Mittel | Mittel | |
| 11 | 4.2 Internet-Zugriff (Tailscale) | Mittel | Gering | Später, nach 4.1 |
| 12 | 3.5 Wikilink-Expansion | Mittel | Mittel | Experimentell |
| 13 | 3.3 Chunk-Inspector | Gering (Dev-Tool) | Gering-Mittel | Wenn nötig |
| 14 | 2 Bilder im Chat (Phase 2: Inline) | Nice-to-have | Mittel-Hoch | Später |
