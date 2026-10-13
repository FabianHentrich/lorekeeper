# LoreKeeper — Roadmap

Stand: 13.10.2026

Hier steht **offene Arbeit**. Erledigtes ist unten eine Zeile mit Verweis; Entscheidungen, Messwerte
und Begründungen stehen in der zuständigen Doku (Tabelle in `CLAUDE.md` → „Documentation“), die
Umsetzung im `git log`. Die Nummern bleiben stabil — die Doku verweist auf sie.

**Fokus seit 13.10.: Qualität vor neuen Features.** Features (⏸) ruhen, bis die Qualitätspunkte
erledigt sind.

---

## Priorisierung

**Last** = Rechenlast beim Umsetzen/Messen: 🟢 nur Code + Unit-/UI-Tests (geht parallel zur
Simulation) · 🟡 braucht laufendes Backend oder LLM · 🔴 Indizierung oder Vision-Modell (CPU/GPU,
mehrere Minuten, nicht parallel zur Simulation). Ein Golden-Set-Lauf dauert ~1 min (🟡).

| # | Punkt | Impact | Aufwand | Last | Begründung |
|---|---|---|---|---|---|
| 1 | 3.10 Restliche Abschnitt-Misses | Gering-Mittel — Regelfragen präziser | Mittel | 🟡 | Einzelfälle, je Golden-Set-Lauf prüfbar |
| — | 3.2 Session-History ⏸ | Unklar | Mittel | 🟢 | Erst entscheiden, ob gespeicherte Sessions Nutzen haben |
| — | 3.4 Kommentar zur Bewertung ⏸ | Gering | Gering | 🟢 | Feature |
| — | 2 Bilder im Chat (Phase 1/2) ⏸ | Mittel / Nice-to-have | Mittel / Mittel-Hoch | 🔴/🟡 | Feature |
| — | 4.2 Internet-Zugriff (Tailscale) ⏸ | Mittel | Gering | 🟢 | Vorher die Vertrauensgrenze der API klären |
| — | 3.5 Wikilink-Expansion ⏸ | Mittel | Mittel | 🔴 | Neuer Suchweg — muss in den Spy-Test |
| — | 11.0 P8 Dienste-/Reranker-Modul (optional) | Gering | Mittel | 🟢 | Nur zusammen mit Feature-Arbeit |

**Offen am Gaming-PC (Nutzer)**: alte Autostart-Aufgabe „LoreKeeper Server“ als Admin entfernen
(`Unregister-ScheduledTask -TaskName "LoreKeeper Server" -Confirm:$false`); DHCP-Reservierung für
`192.168.178.34` im Router. Nebenbei gesehen: Inbound-Allow-Regeln für `python.exe` im Profil
„Öffentlich“ gehören zu anderen Python-Installationen (alte LoreKeeper-Kopie unter OneDrive,
PortraitForge, globales Python) — ggf. selbst aufräumen.

**Offen im Browser** (beim nächsten Re-Index, 6.0): Ingest-Fortschritt durchklicken (Job starten,
auf der Seite klicken, Seite wechseln und zurück — 3.8/3.9); 👍/👎 und die Tabs „Feedback“/„Chunks“
einmal von Hand bedienen (API-Seite live geprüft, siehe unten).

---

## Offene Punkte

### 3.10 Restliche Abschnitt-Misses

**13.10. umgesetzt:** `top_k` 15 → 20 — Vorteil/Nachteil (k002) und Heimkehrpfennig (x008) gelöst
(Abschnitte 136 → 138/153); Hybrid gemessen und verworfen (überall schlechter). Messung:
`docs/embedding-strategy.md` → „Warum `top_k` 20?“.

**Offen, je mit Diagnose** (Rang des richtigen Chunks in der Vektorsuche, Reranker-Score richtig
gegen gewählt; Skript-Muster: Rang in den Top 200 + `_predict` je Paar):
- **Zu tief für mehr Kandidaten** — Waffentabelle 7.1.1: „Langschwert“ (p086) Rang 37, „Armbrust“
  (p087) Rang 50 hinter „Armdrücker“; der Reranker würde die Armbrust-Tabelle vorn sehen. Ansatz:
  Embed-Text von Tabellen-Chunks (Zeilennamen stärker gewichten) — braucht Neu-Indizierung.
- **Reranker wählt anders** — Würfelsystem (p003, 1.1 auf Rang 6, −5,6 gegen −4,7), Erlernen
  neuer Zauber (p010, 5.1 gegen 5.3), Verbände (p099, Rang 1, verliert im Rerank), Flammenheilung
  (p108, Rang 22).
- **Entität + Regel** — x003 (Mordlust bei Malek), x004 (Phönixfeder bei Thalion).
- **Simulation** — z003, z006, z009, z010, z012, z014, z015 (Fragen zu Berichtsabschnitten).
- Die 22-Chunk-Sektion „7.3 Legendäre Artefakte > Monolith“ landet bei fremden Regelfragen in den
  Top 3 („Ass-Regel“ p011, „Zauberspeicher“ p101); das Budget aus 5.6 streicht sie nur.

**Inhalt (Nutzer)**: 59 Gegner-/Dämonen-Dateien haben seit dem Simulations-Export einen Abschnitt
„## Alte Vault-Werte“ („nicht mehr maßgeblich“) im eingeklappten Callout — LoreKeeper indiziert ihn
als normalen Text, alte und neue Werte konkurrieren (z010 zieht „Alte Vault-Werte > Fähigkeiten“).

### 2 Bilder im Chat ⏸

Heute werden Bilder nur als Dateiname + Ordner indiziert, und der Retriever schließt Bild-Chunks
komplett aus ([embedding-strategy.md](docs/embedding-strategy.md) → „Warum Bilder aus der Suche
ausschließen?“). Ziel: „Wie sieht die Karte von Arkenfeld aus?“ zeigt das Bild im Chat.

**Phase 1 — Bilder beschreiben**: optionaler Vision-Schritt beim Ingest (`ingestion.image_description:
none | local | gemini`), bevorzugt lokal über Ollama (Qwen2.5-VL / Gemma 3 — verdrängt beim
Indizieren das Chat-Modell aus dem VRAM, also nur als Ingest-Schritt); Gemini Flash als
kostenpflichtige Alternative; Beschreibung als Inhalt des Bild-Chunks, Dateiname als Rückfall. Dann
Bild-Ausschluss im Retriever aufheben, Golden-Set-Fragen i001–i010 prüfen. **Vorher klären**: aus PDFs
extrahierte Bilder liegen in einem temporären Ordner, der nach dem Parsen gelöscht wird — die
Extraktion muss an einen dauerhaften Ort schreiben. Verworfen: CLIP-Embeddings (eigener
Vektorraum, eigene Collection), manuelle Captions (selten vorhanden).

**Phase 2 — Inline-Anzeige**: `GET /files/{path}` (nur innerhalb der Quellordner) statt lokaler
Pfade — nötig, sobald die UI von einem anderen Gerät geöffnet wird; System-Prompt erlaubt Bild-
Verweise, die UI ersetzt sie durch `st.image()`; Rückfall: Bild im Quellen-Bereich.

Optional, für PDFs mit schwierigem Layout: Docling statt pymupdf4llm (braucht PyTorch).

### 3.2 Session-History ⏸

Sessions leben nur im RAM und sind nach einem Neustart weg. Idee: in SQLite/JSON speichern,
Liste „Letzte Gespräche“ in der Sidebar, Export als Markdown. Zurückgestellt (12.10.): erst
entscheiden, ob gespeicherte Sessions bei einem Wissenstool Nutzen haben.

### 3.4 Kommentar zur Bewertung ⏸

👍/👎 um einen kurzen Freitext-Kommentar erweitern (landet in `logs/feedback.jsonl` und im Tab
„Feedback“).

### 3.5 Wikilink-Expansion ⏸

Wikilinks (`[[Malek]]`) stehen als Metadaten am Chunk, werden aber nicht genutzt. Idee: nach dem
Retrieval die Wikilinks der Top-Chunks auflösen und die verlinkten Chunks nachladen. Ist ein neuer
Suchweg: muss den Anfrage-Filter tragen und in den Spy-Test der Sicherheitstests — dann ggf. einen
gefilterten Store statt Konvention ([ARCHITECTURE.md](ARCHITECTURE.md#8-architektur-entscheidungen)).

### 4.2 Internet-Zugriff ⏸

**Tailscale** (privates Mesh-VPN) auf Gaming-PC und Laptop → `http://gaming-pc:8501` von überall,
ohne Portfreigabe. Alternative: Cloudflare Tunnel + Access. Offen: Der Gaming-PC muss laufen
(Wake-on-LAN braucht ein immer laufendes Gerät im Heimnetz). **Vorher**: Die API vertraut der
Rolle des Aufrufers und hat keinen eigenen Login ([ARCHITECTURE.md](ARCHITECTURE.md#8-architektur-entscheidungen)
→ „Bewusste Grenze“) — solange nur Streamlit sie erreicht, reicht das; Tailscale ändert daran
nichts, eine direkt erreichbare API schon.

### 11.0 P8 — optional

Dienste-Modul `src/services.py` (Globale, `install_llm`, Health-Loop; beseitigt die Lazy-Imports
und vereinfacht die Test-Fixtures) und Reranker + GPU-Rückfall als `src/retrieval/reranker.py`. Nur
zusammen mit Feature-Arbeit, die diese Stellen ohnehin anfasst.

---

## Erledigt

| Nr | Was | Wann | Wo dokumentiert |
|---|---|---|---|
| 1 | PDF-Parsing: OCR (RapidOCR), Bildextraktion, Überschriften aus dem Inhaltsverzeichnis | 14.04. | `docs/parsing.md` |
| 3.1 | Hybrid-Suche (BM25 + Vektor, RRF), Schwelle vor der Fusion | 14.04. | `docs/embedding-strategy.md` |
| 3.3 | Chunk-Inspector (Tab „Chunks“) | 12.10. | `docs/ui-ux.md` |
| 3.4 | 👍/👎 unter jeder Antwort, `logs/feedback.jsonl`, Übernahme ins Golden Set | 12.10. | `docs/ui-ux.md`, `docs/operations.md` |
| 3.6 | Letztes Frage/Antwort-Paar im Antwort-Prompt | 12.10. | `docs/prompts.md`, `docs/configuration.md` |
| 3.7 | Admin-Passwort für die UI, Bruteforce-Bremse | 11.10. | `docs/ui-ux.md`, `ARCHITECTURE.md` 4.8 |
| 3.8/3.9 | Ingest-Fortschritt ohne Blockieren, Anhängen an laufende Jobs | 12.10. | `docs/ui-ux.md` |
| 3.10 | Regelwerk-Slot (eigene Suche in `rules`, Mindest-Score −4) | 13.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 3.11 | Spieler-/Spielleiter-Modus inkl. Sicherheitstests (deterministisch, live, Spy-Test) | 11./12.10. | `ARCHITECTURE.md` 4.8, `docs/testing.md` |
| 3.12 | Folgefragen ohne Inhaltswort („Fass das kürzer zusammen“, „Gib mir ein Beispiel“) gehen durch Condense — Suche und Quellenliste passen zum Thema; Wortliste nur ergänzend (unbekanntes Wort = Inhaltswort), Unit-Tests in beide Richtungen, mutationsgeprüft. Offen: einmal live prüfen, dass Condense das Thema einsetzt | 13.10. | `docs/prompts.md` |
| 4.1 | Heimnetz-Betrieb (`start.ps1 -Server`, Firewall-Skript, kein Autostart) | 11.10. | `docs/operations.md` |
| 4.2 (Teil) | Spieler-Filter im Backend erzwungen (`role`) | 12.10. | `ARCHITECTURE.md` 4.8 |
| 4.3 | Sauberes Beenden: `stop.ps1` (Strg+C an beide Fenster, nach 20 s hart, samt Ollama; Sperre während Ingest) und **⏻ LoreKeeper beenden** in der Sidebar (Admin). Gemessen: Strg+C ist in allen Lagen sauber (Backend 1,3–1,6 s, UI 0,2 s, auch bei Stream/Eval) — 7 von 10 Starts endeten durch Fenster-Schließen ohne Shutdown; die verzögerte Strg+C-Reaktion war nicht reproduzierbar (Verdacht: markierter Text im Konsolenfenster hält den Prozess an) | 13.10. | `docs/operations.md`, `docs/ui-ux.md` |
| 5.1 | Stage-Timings je Anfrage (Log, `done`-Event, Chat-Zeile) | 11.10. | `docs/operations.md` |
| 5.2 | Performance-Tests: Anfrage-Budgets, `pytest -m perf`, Startup-Baseline 17,7 s | 11./12.10. | `docs/testing.md` |
| 5.3 | Antwortzeit: Qwen3-Denken aus, Kontext 16k, Keep-alive 1 h, `top_rank` 3 — erster Token 8,7 → 1,1 s | 11./12.10. | `docs/provider-strategy.md`, `docs/configuration.md` |
| 5.4 | Reranker auf der GPU (DirectML): 840 → 50 ms, unter Last 14 s → 70 ms | 12.10. | `docs/embedding-strategy.md` |
| 5.5 | Vektorsuche exakt und reproduzierbar: `hnsw_ef_search` 100 → 1000 (0 statt 11 abweichende Golden-Fragen zwischen Prozessen, +5 ms/Frage, p105 gewonnen); Stolperfalle l2 beim Anlegen behoben; Prüfung „🎯 Suchgenauigkeit“ in der UI und Anleitung auf der Evaluation-Seite | 13.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 5.6 | Gesamtbudget der Section Expansion `max_total_chunks: 30`: max. 49 → 30 Chunks pro Frage, Treffer unverändert (276/278, 137/153); trifft nur die 3 Fragen mit gestapelten Sektionen und streicht dort die fremde. 25 und kleiner schnitten Klassen-/Magie-Sektionen (Sweep 0–12 in einem Prozess) | 13.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 6.0 | Simulation im Index: Quellen-Gruppe `simulation` (12.10.); Neu-Indizierung nach SIM 52 und Begegnungsbericht 13.10. (81 Dateien, 2957 Chunks, Index 4733). Eval: Datei 276/278, Abschnitt 136/153 (z010 verloren), z001–z016 Datei 14/16, Abschnitt 9/16; 55 von 278 Fragen mit Simulations-Chunks. Spieler bekommen bei 47 von 124 reinen Regelfragen Ø 2,3 Chunks aus dem Fähigkeitsbericht (6,7 % aller Chunks, 3× in den Top 3) | 12./13.10. | `docs/configuration.md`, `docs/evaluation.md` |
| 7.0/7.1 | Testlücken Gemini/Prompts/Eval auf 100 %, Eval-CLI in der Windows-Konsole | 12.10. | `docs/testing.md` |
| 7.2 | Code-Review: Filter-Vertrag (nur `$eq`/`$ne`/`$in`/`$and`, sonst 422), GPU-Rückfall bei gleichzeitigen Fragen | 13.10. | `docs/configuration.md`, `docs/testing.md` |
| 7.3 | Inline-Doku: Docstrings für alle Module, Klassen, Funktionen; Config-Felder kommentiert | 13.10. | Code |
| 7.4 | Clean-Code-Review, Pakete A–D (Fehler, toter Code, Vereinfachung, Clean Code) | 13.10. | `git log` |
| 8.0 | Gleichzeitige Fragen gemessen, `OLLAMA_NUM_PARALLEL` bleibt 1, Warte-Hinweis im Chat | 13.10. | `docs/provider-strategy.md`, `docs/ui-ux.md` |
| 9.0 | Testabdeckung nach Risiko 86 → 93 % | 13.10. | `docs/testing.md` |
| 10.0 | Logging: Fehler mit Traceback in der Datei, wichtige Ereignisse, weniger Rauschen, eine Datei pro Tag, `logs/ollama.log` | 13.10. | `docs/operations.md` |
| 11.0 | Architektur-Review + Umbau P1–P7 (Recategorize, Fallback-Schalter, Index-Konsistenz, Filter erzwungen, UI-Fehlerpfad, atomares Speichern, Router aufgeteilt) | 13.10. | `ARCHITECTURE.md` 8 |
| 12.0 | Code-Review seit a3b5d77 + Korrekturen: `stop.ps1` nimmt sich selbst aus dem Prozessbaum (Button wartete 20 s und beendete sich selbst) und lässt fremde Server auf 8000/8501 in Ruhe; `hnsw_ef_search` ≥ 1, `max_total_chunks` ≥ 0; Collection wird beim Start geöffnet (ef_search wirklich beim Start); p50/p95 der Evaluation-Seite wie in `pytest -m perf`; Exaktheitsprüfung zählt Gleichstände an der Top-K-Grenze nicht; Tests für verschachtelte Sektionen und Duplikate (je mutationsgeprüft) | 13.10. | `docs/operations.md`, `docs/evaluation.md`, `docs/testing.md` |

**Live-Prüfungen (13.10., echtes LLM und echter Index)**: Ollama-Autostart findet die Modelle
(`OLLAMA_MODELS`); Folgefrage „fass das kürzer zusammen“ liefert eine kürzere Antwort zum selben
Thema (Verlauf wirkt; Quellenliste siehe 3.12); Regelwerk-Slot im Chat bei x002 und x005 —
Regelbuch unter den Quellen, Antworten nennen die Regel; Rerank-Zeiten in der Chat-Zeile, warm
~70–250 ms auf der GPU; Chunk-Inspector und Feedback-Liste über die API; Warte-Hinweis bei zwei
gleichzeitigen Fragen; `pytest -m live` 283 grün (2:21 min).
