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

### 3.7 Zugriffsschutz für die UI — ✅ Erledigt

**Problem**: Die API selbst ist nicht exponiert — uvicorn bindet auf `127.0.0.1`, und alle API-Aufrufe macht der Streamlit-Prozess serverseitig. Exponiert ist die **Streamlit-UI**: sie lauscht standardmäßig auf allen Interfaces und hat keinen Login. Wer im selben Netz `http://<pc>:8501` öffnet, kann über die Admin-Seiten indizieren, die Collection löschen (`/admin/wipe`), Config und Gemini-Key ändern. Zusätzlich lässt das Feld „API URL“ in der Sidebar den Streamlit-Server beliebige URLs abfragen.

**Lösung**:
- Passwort-Gate (Secret aus `.env`) vor den Admin-Seiten (Sources, Settings, Prompts, Evaluation); der Chat bleibt frei oder hängt optional am selben Gate
- „API URL“-Feld durch Config-Wert ersetzen (nicht im Browser editierbar)
- Backend bleibt auf `127.0.0.1` — ein API-Key wird erst nötig, wenn die API selbst je exponiert wird

**Aufwand**: Gering. Teil von 4.1.

**Umgesetzt** (`ui/admin_gate.py`): `LOREKEEPER_ADMIN_PASSWORD` in `.env` schützt Sources, Evaluation, Prompts, Settings sowie Provider-Wechsel und Gemini-Key in der Chat-Sidebar; der Chat bleibt offen (Entscheidung: nur Admin-Seiten schützen). Login gilt pro Browser-Session. Ohne Passwort alles offen wie bisher. „API URL“-Feld entfernt, Backend-URL aus `LOREKEEPER_API_URL`. Tests: `tests/ui/test_admin_gate.py`.

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

### 3.11 Spieler-/Spielleiter-Modus — ✅ Umgesetzt, Sicherheitstests offen

**Problem**: Im Heimnetz nutzen auch Spieler LoreKeeper (z.B. am Handy). Der Vault enthält Spielleiter-Wissen (Plot, NPC-Geheimnisse wie Sol Invictus = Rhaz'Muell), das Spieler nicht sehen sollen; für sie ist nur das Regelwerk relevant.

**Lösung (umgesetzt)**: Vor dem Chat die Auswahl „🎲 Spieler“ / „📜 Spielleiter“ (`ui/roles.py`, `ui/LoreKeeper.py`), ohne Passwort (Entscheidung: Spoiler-Schutz, keine Zugangskontrolle).
- Spieler durchsuchen nur Quellen mit `player_visible` (neues Source-Feld, Spalte „🎲 Spieler“ auf der Sources-Seite; freigegeben: `pnp_regelbuch`, `charakterbogen`). Filter `{"source_id": {"$in": [...]}}`, gesetzt im Streamlit-Prozess; Sidebar-Filter für Spieler ausgeblendet und ignoriert.
- „🔄 Rolle wechseln“ startet eine neue Session (kein Spielleiter-Verlauf im Spieler-Kontext).
- Ohne freigegebene Quelle ist der Spieler-Chat gesperrt.
- Live geprüft: „Wer ist Sol Invictus wirklich?“ / „Was ist mit Isolde Vaer los?“ im Spieler-Modus finden nur Regelbuch/Charakterbogen, Antwort „keine Informationen“.
- Doku: `docs/ui-ux.md` (Role selection), `docs/configuration.md` (`player_visible`), `docs/operations.md`.

**Offen — Sicherheitstests** (nächster Schritt): systematisch prüfen, dass im Spieler-Modus kein Spielleiter-Inhalt in Antwort oder Quellenliste landen kann — Retrieval-Pfade (Vektor, BM25/Hybrid, Section Expansion, Rerank-Backfill), Condense/Session-Verlauf, Rollenwechsel, Cache, manipulierte Session-State-Werte, Fehlerfälle (Backend nicht erreichbar, leere Freigabe), Golden-Set-Fragen mit SL-Wissen im Spieler-Modus.

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
6. Danach: Startup-Messung (5.2) mit dem Server-Modus.

**Befund**: Es existieren Inbound-Allow-Regeln für `python.exe` im Profil „Öffentlich“ (alle Ports) — für andere Python-Installationen (alte LoreKeeper-Kopie unter OneDrive, PortraitForge, globales Python 3.12), nicht für LoreKeeper. Nicht angefasst; ggf. selbst aufräumen.

### 4.2 Internet — später

**Lösung**: **Tailscale** (privates Mesh-VPN, kostenlos für Privatnutzung) auf Gaming-PC und Laptop → `http://gaming-pc:8501` funktioniert von überall, ohne Portfreigabe im Router und ohne öffentliche Erreichbarkeit. Alternative: Cloudflare Tunnel + Cloudflare Access (Login-Seite davor) — mehr Einrichtung.

**Offener Punkt**: Der Gaming-PC muss laufen. Wake-on-LAN von unterwegs braucht ein immer laufendes Gerät im Heimnetz (Router-Funktion oder Raspberry Pi); sonst bleibt der PC an.

**Aufwand**: Gering (Einrichtung, kaum Code).

## 5.0 Performance-Optimierung und Performance-Tests

**Ziel**: Bei Weiterentwicklungen sofort sehen, wenn etwas deutlich ineffizienter wird — vor allem bei der Antwortzeit und der UI-Reaktion, beim Start nur einfache Hebel.

**Stand (2026-10-11)**:

| Bereich | Status | Ergebnis |
|---|---|---|
| Antwortzeit | ✅ große Hebel erledigt (5.1, 5.3) | Erster Token 8,7 s → **1,1 s**, gesamt 13,9 s → **3,7 s** (Überschaden, warm); kein abgeschnittener Prompt mehr; kein 16-s-Kaltstart innerhalb 1 h |
| Retrieval | ⚠️ offen | 0,6–0,8 s, davon Rerank 0,5–0,75 s — läuft auf der **CPU** (PyTorch-CPU-Build), siehe 5.3 |
| UI-Reaktion | ✅ Hauptursache behoben (5.2) | `localhost`-Delay von ~2 s pro Backend-Request beseitigt (Chat −4 s pro Nachricht, Sources-Seite −10 s); UI-Code selbst 0,02–0,04 s pro Rerun; Rest nur noch Millisekunden (5.3) |
| Start | ❌ offen | ~12 s (Embedding-Modell ~7 s, Reranker ~4 s); nur einfache Hebel geplant; Messung kommt mit dem Server-Modus (4.1) |
| Regressionsschutz | ✅ erledigt (5.2) | Deterministische Budgets im normalen `pytest`, Zeit-Checks per `pytest -m perf` gegen `tests/perf/baseline.json`; Startup-Messung offen (mit 4.1) |

### 5.1 Messen: Stage-Timings — ✅ Erledigt

Pro Anfrage loggen und im `done`-Event mitliefern: Condense, Retrieval (Embedding / Vektorsuche / Rerank / Expansion), Zeit bis zum ersten Token, Gesamtzeit, Anzahl Chunks und Prompt-Größe (Zeichen bzw. `usage.tokens_in`). Grundlage für alles Weitere; die UI kann die Werte optional in der Caption zeigen.

**Aufwand**: Gering.

**Umgesetzt**: `timings` in `/query` und im `done`-Event (`condense_ms`, `retrieval_ms` + `retrieval_stages_ms` {embed, search, bm25, rerank, expand}, `first_token_ms`, `total_ms`, `chunks`, `prompt_chars`), eine Log-Zeile `Query timings: …` pro Anfrage, Caption „⏱ 1. Token … · gesamt … · Suche … · N Chunks“ unter jeder Antwort.

**Erste Messung (qwen3:14b, warm, GPU)**: Retrieval 0,6–0,8 s (davon Rerank ~0,5–0,75 s), erster Token nach 8,7–9,9 s, gesamt 14–20 s. → Der große Block vor dem ersten Token ist **nicht** Retrieval, sondern verstecktes Denken (siehe 5.3).

### 5.2 Performance-Tests (Regressionsschutz) — ✅ Erledigt (Startup-Messung offen)

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

**Umgesetzt** (Details in `CONTRIBUTING.md` → „Performance checks“):
- Deterministisch, im normalen `pytest`: Backend-Requests pro Seiten-Render/Rerun/Chat-Nachricht (`tests/ui/test_backend_calls.py`, `AppTest` + Fake-Backend, Budgets in `BUDGETS`); LLM-Aufrufe pro Anfrage (`test_llm_calls_per_request`); keine `localhost`-URLs für lokale Dienste.
- Zeitbasiert, `pytest -m perf` gegen `tests/perf/baseline.json` (×1,5 + Slack; `PERF_UPDATE_BASELINE=1` übernimmt neue Werte): Render-Zeit je Seite; Golden Set über das laufende Backend — p50/p95 Retrieval-Latenz, Ø/max Chunks pro Frage, Abschnitt-Trefferquote darf nicht sinken.
- **Offen**: Startup-Messung — ein zweites Backend kann nicht parallel zum laufenden starten (Port, eingebettete ChromaDB); sinnvoll erst mit dem Server-Modus aus 4.1.

**Befunde aus der ersten Messung**:
- **`localhost` kostet 2 s pro Verbindung (behoben)**: Unter Windows löst `localhost` zuerst auf IPv6 `::1` auf, uvicorn und Ollama lauschen nur auf IPv4 → jede neue Verbindung wartet ~2 s auf den Fallback (gemessen 2000 ms vs. 3 ms). Betraf jede Chat-Nachricht (~4 s: `/config` + `/query/stream`), jeden Sources-Seiten-Aufbau (~10 s) und neue Verbindungen zu Ollama nach einer Pause. Alle Standard-URLs (UI, Config, Eval) stehen jetzt auf `127.0.0.1`; ein Test verhindert die Rückkehr. **Achtung**: Eine offene Browser-Session mit `localhost` im Sidebar-Feld „API URL“ behält den Delay bis zum Neuladen.
- **UI-Code selbst ist schnell**: Rerun je Seite 0,02–0,04 s, erster Aufbau < 0,7 s (Fake-Backend) — die gefühlte Trägheit kam aus den Backend-Aufrufen.
- **Sources-Seite**: `/sources/{id}/folders` wird bei jedem Rerun neu geladen (1× pro Source, ungecacht) — seit dem `127.0.0.1`-Fix nur noch Millisekunden, als Budget im Test festgehalten.

### 5.3 Bekannte Hebel (nach der Messung angehen)

- **Antwortzeit — qwen3 denkt trotz `/no_think` (gemessen, größter Hebel)**: Ollama ignoriert das an den Prompt gehängte `/no_think`; qwen3 erzeugt Reasoning-Tokens im separaten Feld `reasoning`, das LoreKeeper weder anzeigt noch zählt (`tokens_thinking` bleibt 0). Kostet ~7 s pro Antwort. Direkter Test gegen Ollama: `reasoning_effort: "none"` (bzw. `reasoning: {effort: "none"}`) schaltet es ab (0 Reasoning-Chunks), `think: false` wird ignoriert. ✅ **Erledigt**: `OllamaProvider` sendet für Qwen3 `extra_body={"reasoning_effort": "none"}` statt `/no_think`; Reasoning-Chunks im `reasoning`-Feld werden als `tokens_thinking` gezählt.
- **Antwortzeit — Kontextfenster (gemessen)**: `ollama ps` zeigt `CONTEXT 4096` für qwen3:14b. Die Krieger-Frage brauchte 3621 Eingabe- + 919 Ausgabe-Tokens — am bzw. über dem Limit; bei größerer Section Expansion wird der Prompt-Anfang (Systemprompt) still abgeschnitten. ✅ **Erledigt**: `OLLAMA_CONTEXT_LENGTH=16384` als Benutzer-Umgebungsvariable gesetzt (der OpenAI-kompatible Endpoint ignoriert `num_ctx` pro Request, getestet). Danach `CONTEXT 16384`, 11 GB von 16 GB VRAM, weiterhin 100 % GPU, keine messbare Verlangsamung (warm: erster Token 1,1 s / 2,3 s).
- **Antwortzeit — Modell entladen nach 5 Minuten (gemessen)**: Ollama entlädt das Modell nach 5 min Leerlauf; die erste Frage danach wartet ~16 s aufs Laden (erster Token 18 s statt 1,1 s). ✅ **Erledigt**: `OLLAMA_KEEP_ALIVE=1h` als Benutzer-Umgebungsvariable — das Modell bleibt bis 1 h nach der letzten Frage geladen (`ollama ps`: „59 minutes from now“). Belegt in der Zeit ~11 GB VRAM; `-1` hielte es dauerhaft geladen.
- **Präzision/Kontext — Section Expansion begrenzt** ✅ **Erledigt**: `section_expansion.top_rank: 3` — expandiert nur, wenn ein Treffer der Sektion unter den ersten 3 Chunks liegt. Golden Set (0 = altes Verhalten / 3 / 2 / 1): Ø Chunks 13,9 / 12,6 / 11,9 / 10,7; expandierte Fragen 144 / 119 / 102 / 75; Abschnitt-Treffer überall 89,1 %. Bei 3 fallen nur Fehl-Expansionen weg (z.B. Geisterwesen-Statblöcke bei „Eigenheiten Geist und Psyche“); 2 und 1 schneiden bereits relevante Übersichts-Chunks ab (Akademie-Geschichte, Szenenübersicht, Arkenfeld-Steckbrief). Offen: Gesamtbudget über alle expandierten Sektionen (max. 49 Chunks bleiben möglich). Unnötige Condense-Aufrufe sind bereits entfernt (`needs_condense`).
- **Retrieval — Reranker/Embeddings auf der CPU (gemessen, offen)**: `torch 2.14.1+cpu` ist installiert, `torch.cuda.is_available()` = False. Rerank kostet 0,5–0,75 s pro Frage, Re-Indexierung des Regelwerks ~80 s. Mit CUDA-PyTorch liefen beide auf der GPU (Platz: Modell 11 GB + Reranker/Embedding ~1–2 GB von 16 GB). **Neue Abhängigkeit** (CUDA-Build von torch, ~2–3 GB Download) → vorher entscheiden; danach Timings und VRAM messen.
- **Kontext — Gesamtbudget für Section Expansion (offen)**: Mehrere expandierte Sektionen können zusammen noch bis zu 49 Chunks ergeben (z.B. „Wie wird die Bonusaktion im Kampf eingesetzt?“). Ein Budget über alle Sektionen (z.B. max. 20 Chunks) begrenzt das; messbar mit der Chunk-Prüfung aus 5.2.
- **UI-Reaktion (gemessen, Hauptursache behoben)**: Die 2-s-`localhost`-Verzögerung pro Request ist behoben (5.2). Rest, jetzt nur noch Millisekunden pro Request: Chat holt vor jeder Nachricht `/config` (cachen); Sources-Seite lädt `/folders` bei jedem Rerun; Sources-Seite blockiert beim Polling (3.8/3.9). Kosten eines Reruns mit langer Chat-Historie und vielen Quellen-Expandern noch nicht gemessen.
- **Start (offen)**: nur einfache Hebel — `--reload` im Server-Modus aus (4.1); Reranker wird in `src/main.py` beim Start vorgeladen (~4 s) — lazy laden verschiebt die Zeit nur auf die erste Frage, lohnt also nur mit CUDA (dann deutlich schneller geladen).
- **Generierung**: Lange Antworten (Krieger: 655 Tokens ≈ 12 s) sind durch die Modellgeschwindigkeit (~50 Token/s) begrenzt — kein Performance-Bug; Stellschraube wäre ein Prompt, der kürzere Antworten verlangt (Produktentscheidung).

**Aufwand**: je Hebel Gering bis Mittel.


## 6.0 PnP-Simulation zugreifbar machen

**Fragen (zu klären)**: Reichen die HTML-Dateien als Zugriff? Welche Daten soll das LLM bekommen? Wie wird das Ganze indexiert?

**Nächster Schritt**: HTML-Dateien und `pnp_sim`-Exporte gemeinsam ansehen (was steht drin, was ist generiert, was ändert sich wie oft), dann Lösung und Aufwand festlegen. Bezug: Die Vault-Gegnerseiten enthalten bereits `sim:start`/`sim:end`-Blöcke aus `pnp_sim` (`scripts/begegnungen/vault_export.py`).

**Aufwand**: offen.

---

## Priorisierungsvorschlag

Erledigt: 3.11 Spieler-/Spielleiter-Modus (ohne Sicherheitstests), 4.1 Heimnetz + 3.7 Zugriffsschutz, 1 PDF-Parsing, 3.1 Hybrid Search, 5.1 Stage-Timings, 5.2 Performance-Tests (+ `localhost`-Fix), 5.3 Antwortzeit (Qwen3-Thinking aus, Kontext 16k, Keep-alive 1 h, Section Expansion `top_rank`).

| Reihenfolge | Punkt | Impact | Aufwand | Begründung |
|---|---|---|---|---|
| 1 | ~~5.2 Performance-Tests~~ | | | ✅ Erledigt (Startup-Messung mit 4.1) — dabei `localhost`-Delay gefunden und behoben |
| 2 | ~~4.1 Heimnetz + 3.7 Zugriffsschutz~~ | | | ✅ Umgesetzt und eingerichtet (Handy-Test erfolgreich); alte Autostart-Aufgabe noch per Admin entfernen |
| 2a | 3.11 Spieler-/Spielleiter-Modus — Sicherheitstests | Hoch — Spieler dürfen kein SL-Wissen sehen | Mittel | Feature umgesetzt; Tests sind der nächste Schritt |
| 3 | 3.8/3.9 Ingest-Fortschritt + 5.3 UI-Reste (`/config` cachen, `/folders` cachen) | Gering-Mittel — Sources-Seite bleibt bedienbar | Gering-Mittel | Hauptursache der UI-Trägheit ist behoben; Rest sind Komfort und Millisekunden |
| 4 | 5.3 Reranker/Embeddings auf GPU (CUDA-PyTorch) | Mittel — ~0,5 s pro Frage, schnellere Indexierung | Gering (Code) | Neue Abhängigkeit — erst entscheiden |
| 5 | 6.0 PnP-Simulation zugreifbar machen | offen | offen | Erst Daten ansehen und klären |
| 6 | 3.10 Regelwerk-Slot im Retrieval | Mittel — Vault↔Regel-Fragen | Mittel | Mit Abschnittsprüfung messbar |
| 7 | 5.3 Gesamtbudget Section Expansion | Gering-Mittel | Gering | Zusammen mit 5.2-Chunk-Prüfung |
| 8 | 3.6 Chat-Verlauf im Antwort-Prompt | Mittel — bessere Folgefragen | Gering-Mittel | Kontext jetzt 16k |
| 9 | 3.4 Antwort-Feedback | Gering-Mittel | Gering | Quick Win |
| 10 | 2 Bilder im Chat (Phase 1: Beschreibung) | Mittel — Karten/Handouts suchbar | Mittel | |
| 11 | 3.2 Session-History | Mittel | Mittel | |
| 12 | 4.2 Internet-Zugriff (Tailscale) | Mittel | Gering | Später, nach 4.1 |
| 13 | 3.5 Wikilink-Expansion | Mittel | Mittel | Experimentell |
| 14 | 3.3 Chunk-Inspector | Gering (Dev-Tool) | Gering-Mittel | Wenn nötig |
| 15 | 2 Bilder im Chat (Phase 2: Inline) | Nice-to-have | Mittel-Hoch | Später |
