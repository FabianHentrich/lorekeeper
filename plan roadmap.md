# LoreKeeper — Roadmap

Stand: 14.10.2026

Hier steht **offene Arbeit**. Erledigtes ist unten eine Zeile mit Verweis; Entscheidungen, Messwerte
und Begründungen stehen in der zuständigen Doku (Tabelle in `CLAUDE.md` → „Documentation“), die
Umsetzung im `git log`. Die Nummern bleiben stabil — die Doku verweist auf sie.

**Qualitätsphase 13.–14.10. abgeschlossen.** Retrieval Datei 272/274, Abschnitt 143/149; Antworten
~90 % in zwei End-to-End-Läufen; Holdout-Listen 9/10; Code-Review 13.0 und Performance-Review
erledigt. Was an Qualität offen ist, sind Einzelfälle (3.10, 3.15) oder bewusst zurückgestellt (3.14)
— sie werden nur bei Anlass im Spiel angefasst. **Regressionsprüfung** nach größeren
Inhaltsänderungen (Regelbuch, Vault): Golden Set + End-to-End-Lauf + Holdout
([evaluation.md](docs/evaluation.md)), kein Tuning. Features sind wieder dran — Reihenfolge unten.

---

## Priorisierung

**Last** = Rechenlast beim Umsetzen/Messen: 🟢 nur Code + Unit-/UI-Tests (geht parallel zur
Simulation) · 🟡 braucht laufendes Backend oder LLM · 🔴 Indizierung oder Vision-Modell (CPU/GPU,
mehrere Minuten, nicht parallel zur Simulation). Ein Golden-Set-Lauf dauert ~1 min, ein
End-to-End-Lauf ~35 min (🟡).

| # | Punkt | Impact | Aufwand | Last | Begründung |
|---|---|---|---|---|---|
| 1 | 2 Bilder im Chat (Einbettung + Dateiname) | Mittel-Hoch — Karten, Charaktere am Tisch | Mittel | 🟡 | Ohne Vision-Schritt; Parser-Änderung teilt sich 3.5 |
| 2 | 3.4 Kommentar zur Bewertung | Gering | Gering | 🟢 | Schneller Gewinn, bessere Rückmeldung für spätere Fixes |
| 3 | 3.5 Wikilink-Expansion | Mittel | Mittel | 🔴 | Neuer Suchweg (Spy-Test); nutzt dieselbe Parser-Änderung wie 2 |
| — | 4.2 Internet-Zugriff (Tailscale) ⏸ | Mittel | Gering | 🟢 | Nur zusammen mit einem Login für die API |
| — | 3.2 Session-History ⏸ | Gering | Mittel | 🟢 | Empfehlung: streichen — Nutzerentscheidung offen |
| — | 3.10 / 3.15 Einzelfälle | Gering | — | 🟡 | Nur bei Anlass im Spiel |
| — | 3.14 Rechenfragen ⏸ | Mittel | Mittel-Hoch | 🟡 | Nur mit Denken zuverlässig (14–41 s); später mit größerem Modell neu prüfen |
| — | 12.3 Auto-Reload im Entwicklungsmodus hängt | Gering | Gering | 🟢 | Nach Code-Änderungen startete das Backend zweimal nicht neu (14.10.) |
| — | 11.0 P8 Dienste-/Reranker-Modul | Gering | Mittel | 🟢 | Nur zusammen mit Feature-Arbeit an diesen Stellen |

**Entschieden 14.10.**: Der Fähigkeitsbericht bleibt für Spieler freigegeben (Entscheidungshilfe).
`Spielleiter-Tools/potenzielle_anpassungen.md` wird nicht indiziert — Vorschläge sind keine geltenden
Regeln. `HF_HUB_OFFLINE=1` (in `src/__init__.py`): Modelle nur aus dem lokalen Cache, Downloads mit
`HF_HUB_OFFLINE=0` ([operations.md](docs/operations.md#modelle-fehlen-beim-ersten-start)).

**Offen beim Nutzer**:
- **Gaming-PC**: alte Autostart-Aufgabe „LoreKeeper Server“ einmal in einer Admin-PowerShell prüfen
  (`Get-ScheduledTask -TaskName "LoreKeeper*"`); Inbound-Allow-Regeln für `python.exe` im Profil
  „Öffentlich“ gehören zu anderen Python-Installationen — ggf. aufräumen. (DHCP-Reservierung
  `192.168.178.34` eingestellt, 14.10.)
- **Vault**: `Organisationen/Schattenorden/Schattenorden.md` nennt noch „fünf Magiepfade“ (Einleitung,
  Grundsatz 6, Hierarchie-Tabelle), `Meister der Pfade.md` hat vier — das LLM sagt „fünf“ und zählt
  vier auf. Optional: Simulationsberichte mit fragenahen Überschriften und „Kosten je Effekt“ statt
  „Preis“ (3.15).
- **Browser (Handprüfung)**: **⏻ LoreKeeper beenden** einmal klicken — `logs/stop.log` muss nach
  Sekunden mit „stopped … (clean)“ enden (12.0); bei der nächsten Indizierung über die UI den
  Fortschritt durchklicken (Job starten, Seite wechseln und zurück — 3.8/3.9); 👍/👎 und die Tabs
  „Feedback“/„Chunks“ einmal von Hand bedienen (👍/👎 vom Nutzer am 14.10. getestet, erste Bewertungen liegen
  in `logs/feedback.jsonl`). Neu zu sehen: die Chunk-Ausklapper unter den Quellen (3.16).

---

## Offene Punkte

### 2 Bilder im Chat

Ziel: „Wie sieht die Karte von Arkenfeld aus?“ oder eine Frage zu einem Charakter zeigt das passende
Bild im Chat. Heute werden Bilder nur als Dateiname + Ordner indiziert und aus der Suche
ausgeschlossen ([embedding-strategy.md](docs/embedding-strategy.md) → „Warum Bilder aus der Suche
ausschließen?“) — als eigene Chunks verdrängten sie richtige Treffer.

**Neuer Ansatz (Nutzer, 14.10.): ohne Bildbeschreibung.** Bilder werden nicht mehr gesucht, sondern
**zugeordnet**:
1. **Einbettung in der Notiz** — Notizen binden Bilder mit `![[Karte Arkenfeld.png]]` ein; der
   `MarkdownParser` entfernt die Zeile heute nur. Stattdessen als Metadatum am Chunk speichern
   (z. B. `images`) — ein gefundener Chunk bringt sein Bild mit, genau das, das der Autor dorthin
   gesetzt hat.
2. **Dateiname als Rückfall** für Bilder, die nirgends eingebettet sind: „Karte Arkenfeld.png“ passt
   zu Fragen, die „Arkenfeld“ nennen — ähnlich dem Eintrags-Slot (Namenssuche, Filter der Anfrage).

Verworfen: Vision-Beschreibungen beim Ingest (eigenes Modell, GPU-Zeit je Bild, neue
Fehlerquellen), CLIP-Embeddings (eigener Vektorraum), manuelle Captions. Braucht einen Re-Ingest
des Vaults.

**Anzeige**: `GET /files/{path}` statt lokaler Pfade — nötig, sobald die UI von einem anderen Gerät
geöffnet wird. Neue Angriffsfläche: nur Dateien innerhalb der Quellordner, nur Bilder, und im
Spieler-Modus nur Bilder aus freigegebenen Quellen (Sicherheitstest + Spy-Test). Rückfall: Bild im
Quellen-Bereich. **Vorher klären**: Aus PDFs extrahierte Bilder liegen in einem temporären Ordner, der
nach dem Parsen gelöscht wird.

### 12.3 Auto-Reload im Entwicklungsmodus hängt

`.\start.ps1` startet uvicorn mit `--reload`. Am 14.10. blieb das Backend nach einer Code-Änderung
zweimal hängen statt neu zu laden: um 14:13 fuhr es herunter (samt Ollama) und kam nicht wieder hoch
— ein laufender End-to-End-Job ging verloren —, um ~17:15 lud es die Änderung nicht und ließ sich nur
mit `.\stop.ps1` nach 20 s hart beenden. Ursache offen (Verdacht: Reload während GPU-Session oder
Hintergrund-Job, oder das Herunterfahren von Ollama im Lifespan). Bis dahin: nach Code-Änderungen
`.\stop.ps1` + `.\start.ps1`, und nie Code ändern, während ein Job im Backend läuft.

### 3.4 Kommentar zur Bewertung

👍/👎 um einen kurzen Freitext-Kommentar erweitern (landet in `logs/feedback.jsonl` und im Tab
„Feedback“).

### 3.5 Wikilink-Expansion

Wikilinks (`[[Malek]]`) stehen als Metadaten am Chunk, werden aber nicht genutzt. Idee: nach dem
Retrieval die Wikilinks der Top-Chunks auflösen und die verlinkten Chunks nachladen — könnte
Einzelfälle aus 3.15 lösen (NPC → Gegenstand, Entität → Regel). Ist ein neuer Suchweg: muss den
Anfrage-Filter tragen und in den Spy-Test der Sicherheitstests. Teilt sich mit 2 die Parser-Änderung
(Links und Einbettungen als Metadaten). Messen mit Golden Set, End-to-End und Holdout.

### 4.2 Internet-Zugriff ⏸

**Tailscale** (privates Mesh-VPN) auf Gaming-PC und Laptop → `http://gaming-pc:8501` von überall,
ohne Portfreigabe. Alternative: Cloudflare Tunnel + Access. Offen: Der Gaming-PC muss laufen
(Wake-on-LAN braucht ein immer laufendes Gerät im Heimnetz). **Vorher**: Die API vertraut der
Rolle des Aufrufers und hat keinen eigenen Login ([ARCHITECTURE.md](ARCHITECTURE.md#8-architektur-entscheidungen)
→ „Bewusste Grenze“) — solange nur Streamlit sie erreicht, reicht das; Tailscale ändert daran
nichts, eine direkt erreichbare API schon.

### 3.2 Session-History ⏸

Sessions leben nur im RAM und sind nach einem Neustart weg. Idee: in SQLite/JSON speichern, Liste
„Letzte Gespräche“ in der Sidebar, Export als Markdown. Empfehlung (14.10.): streichen — bei einem
Wissenstool steht die Antwort im Regelbuch/Vault, nicht im Verlauf. Nutzerentscheidung offen.

### 3.10 Restliche Abschnitt-Misses — Einzelfälle

Golden Set 14.10.: Datei 272/274, Abschnitt 143/149, Suche exakt. Umgesetzt und verworfen: Tabelle
„Erledigt“ und [embedding-strategy.md](docs/embedding-strategy.md). Offen, nur bei Anlass:
- **Simulationsberichte** z003, z006, z009, z012 (und z001 mit richtiger Antwort) — die Berichts-Chunks
  sagen weder „Schwierigkeit“ noch „Simulation“; besser in den Berichten selbst lösen (Überschriften).
- **p009** Zauberpunkte — Abschnitt-Miss im Golden Set, die Antwort ist seit dem Regelkern richtig.
- **Name in fremder Identität** — „Wer ist Arkenfeld?“ holt vor allem „Der Armdrücker von Arkenfeld“;
  Antwort trotzdem richtig.
- **Monolith-Sektion** (22 Chunks) landet bei fremden Regelfragen in den Top 3 (p011, p101).
- **Tabellen**: „Dolch, STÄ 4, Wurfzahl 12“ und „Unterschied Dolch und Kurzschwert“ (Eintrags-Slot holt
  die Tabelle, jina wertet anderes höher); „Langbogen“ (PDF-Zelle `**l**<br>**Langbogen**`).
- **Schattenorden-Kontext** (14.10., Bug-Review): `Organisationen/Schattenorden/Schattenorden.md` hat nach der
  Vault-Änderung 27 Chunks, über `section_expansion.max_chunks` (25) — m003 „Was ist der Schattenorden?“ und
  m055 „Wie ist er aufgebaut?“ bekommen die Datei nicht mehr ganz; Treffer-Metriken zeigen das nicht.
- **AP-Pool**: „Wie viele Aktionspunkte habe ich?“ → „50 AP“ — die 50 sind ein gemeinsamer Pool für AP
  und MP (2.5).

### 3.15 Antwortfehler aus dem End-to-End-Lauf — Einzelfälle

Feintuning beendet (14.10.): zwischen zwei gleichen Läufen kippen ~15 Fragen, die übrigen Fehler haben
je eine eigene Ursache, und der Holdout zeigt, dass Listen allgemein funktionieren (9/10). Messungen,
Gründe und die angepassten Golden-Set-Stichwörter: [evaluation.md](docs/evaluation.md#evaluatepy--end-to-end).
Offen, nur bei Anlass:
- **Aufzählungen über große Kapitel** — der richtige Abschnitt wird gefunden, aber nur 8 Chunks
  erreichen das LLM: p008 Klassen, p036 Basis-Skills, p037 Basis-Zauber, p114 Eigenheiten Magie, p014
  Benchmark, a030 Beweis-Tabelle, p049 Überheilung.
- **Fakten**: k001 Langschwert „20,5 Gold“ (Spalte „Preis“ der Simulations-Rangliste statt Preistabelle
  7.4); m048 Deckname „Solmar“ (steht ohne Überschrift neben einem Bild — würde von 2 mit profitieren);
  p091 Tränke-Fehlwurf (1W8-Tabelle fehlt); x008 Heimkehrpfennig („Kultpriester“ gilt, Antwort nennt
  Thalion).

### 3.14 Rechenfragen zum Regelwerk ⏸

Fragen, bei denen das Modell Regeln **anwenden und rechnen** muss (höchste Wurfzahl mit Rang 1,
Dolch-Schaden bei Wurfzahl 12, Schattenklinge auf Rang 6), beantwortet qwen3:14b ohne Denken oft
falsch, auch wenn Regel und Werte im Prompt stehen. Mit Qwen3-Denken richtiger, aber 14–41 s pro
Antwort und teils leere Antworten (Token-Budget) — als Standard ungeeignet. Messungen:
[prompts.md](docs/prompts.md#regelkern). **Entschieden 14.10.: vorerst so lassen.** Später: Denken nur
für Rechenfragen (braucht eine Erkennung und mehr `max_tokens`) oder ein größeres Modell; vorher eine
feste Liste von Rechenfragen mit Sollwerten anlegen.

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
| 3.10 (Teil) | Regelwerk-Slot (eigene Suche in `rules`, Mindest-Score −4) | 13.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 3.10 (Teil) | `top_k` 20 (+2 Abschnitte, Hybrid verworfen), `ingestion.skip_sections` gegen „Alte Vault-Werte“ (Full-Reindex, 4419 Chunks), `ef_search` 1000 → 2000 (nach den Re-Index-Läufen wieder exakt), verwaiste Chunks entfernt, Golden Set 274 Fragen — Datei 272/274, Abschnitt 137/149 | 13.10. | `docs/embedding-strategy.md`, `docs/parsing.md`, `docs/configuration.md`, `docs/evaluation.md` |
| 3.10 (Teil) | Tabellen-Chunks: Zeilennamen im Embed-Text („Einträge: Dolch, …“) — Langschwert/Armbrust gelöst, Abschnitt 137 → 139/149, nichts verloren (Vektor-Rang 22 → 14, 47 → 2); Fließtext-Variante verworfen; Full-Reindex, Suche exakt | 13.10. | `docs/embedding-strategy.md` |
| 3.10 (Teil) | Reranker `jina-reranker-v2-base-multilingual` (fp16 auf der GPU, fp32-ONNX als CPU-Rückfall statt torch): Abschnitt 139 → 142/149, Platz-1-Treffer 110 → 121, Datei unverändert 272/274; bge-m3, MiniLM en-de, gte verworfen; VRAM neben qwen3:14b 14,4 statt 14,6 GB, LLM-Tempo gleich | 14.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 3.10 (Teil) | Eintrags-Slot: nennt die Frage eine Tabellenzeile („Dolch-Angriff“, „Hammer“, „Glücksstein“), kommt die Tabelle per Namenssuche in den Rerank-Pool (Filter der Anfrage, max. 3, kein reservierter Platz). Golden Set Abschnitt 142 → 143/149 (+x004), nichts verloren; Tabellenfragen 5 → 8/11; ~75 ms. Verworfen vorher: `top_k` 25–40 (verliert m062/k006/z001), Zeilen-Chunks (142 → 141, Listenfragen 9/9 → 2/9). Spy-Test + Köder-Tabelle, mutationsgeprüft | 14.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 13.0 | Code-Review seit 194be1b (jina, Regelkern, Batches, Eintrags-Slot) + Korrekturen: **kritisch** — eine lange Frage machte den Eintrags-Slot zu einer `$contains`-Abfrage ohne Grenze (4.000 Zeichen → 500, 12.000 → Backend-Prozess stirbt ohne Log, im Heimnetz für jeden Spieler auslösbar); jetzt höchstens 60 Namen (Golden Set braucht ≤ 48, nichts ändert sich) und `question` ≤ 4.000 Zeichen an `/query`, `/query/stream`, `/eval/preview` (422) sowie im Chat-Feld — beides mutationsgeprüft, 50.000 Zeichen laufen durch. Dazu: Kostenmodell Namen × Chunks dokumentiert, Schwellen-Umgehung des Slots dokumentiert und per Test festgehalten, Docstrings/Kommentare nachgezogen, Rerank-Zeiten mit Kontext, Regelkern nur mit spielerfreigegebenem Inhalt. Aus dem Performance-Review: `start.ps1` hielt `TimeWait`-Reste für einen laufenden Server und startete das Backend nicht (jetzt `-State Listen`). Performance 14.10.: Retrieval p50 278 ms / p95 364 ms (Baseline 784/1.156), erster Token ~2 s (Regelkern +0,56 s), Antwort p50 7,6 s je nach Länge | 14.10. | `docs/embedding-strategy.md`, `docs/prompts.md`, `docs/provider-strategy.md`, `ARCHITECTURE.md` |
| 3.15 (Teil) | End-to-End-Basis: zwei Läufe (Answer Contains 0,905 / 0,901, ~15 Fragen kippen zwischen Läufen), `evaluate.py` speichert ganze Antwort, `id` und Stichwörter (Evaluation-Seite zeigt sie), Holdout `evaluation/holdout_lists.yaml` (10 Listenfragen, ungetunt 9/10), acht Golden-Set-Stichwörter angepasst (k001, p095 an die Regelbuch-Änderung, sechs zu enge), Hybrid mit jina neu gemessen (bleibt aus), Eintrags-Slot für Obsidian-Tabellen gemessen und verworfen; Feintuning danach beendet | 14.10. | `docs/evaluation.md`, `docs/embedding-strategy.md`, `docs/ui-ux.md` |
| 3.16 | Chunks unter der Quelle einsehbar (Idee des Nutzers): `sources` liefern zusätzlich `content` (ganzer Chunk-Text), der Chat zeigt je Chunk einen Ausklapper „↳ Überschrift · Score“ mit dem Text (Überschriftenzeile gekürzt, `<br>` → Umbruch, ältere Nachrichten: Vorschau). Spieler-Sicherheitstest prüft auch `content` auf Geheimnisse; UI-, API- und Langtext-Test. Keine zusätzliche Backend-Anfrage | 14.10. | `docs/ui-ux.md`, `ARCHITECTURE.md` |
| 3.11 | Spieler-/Spielleiter-Modus inkl. Sicherheitstests (deterministisch, live, Spy-Test) | 11./12.10. | `ARCHITECTURE.md` 4.8, `docs/testing.md` |
| 3.12 | Folgefragen ohne Inhaltswort („Fass das kürzer zusammen“, „Gib mir ein Beispiel“) gehen durch Condense — Suche und Quellenliste passen zum Thema; Wortliste nur ergänzend (unbekanntes Wort = Inhaltswort), Unit-Tests in beide Richtungen, mutationsgeprüft. Live 13.10.: „Fass das kürzer zusammen“ → „Fasse die Beschreibung von Arkenfeld kurz zusammen.“ (1,2 s), Quellen `Orte/Arkenfeld.md` statt Venthir/Regelbuch | 13.10. | `docs/prompts.md` |
| 3.13 | Regelkern im `qa`-Prompt (15 Stichpunkte zum Regelwerk, direkt vor der Frage; „Ressourcen vs. Punkte“ ergänzt: p009 Zauberpunkte 0/3 → 3/3): mit mMiniLMv2 Rang-1-Wurfzahl und Rüstung je 0/3 → 3/3, Lore-Fragen unverändert; im Systemprompt wirkungslos. Live mit jina: Rüstung 3/3, Rang-1-Wurfzahl wieder 0/3 (Lead „höchstens 13“). Offen: Rang 1 mit jina, Dolch, Schattenklinge (3.10) | 14.10. | `docs/prompts.md` |
| 4.1 | Heimnetz-Betrieb (`start.ps1 -Server`, Firewall-Skript, kein Autostart) | 11.10. | `docs/operations.md` |
| 4.2 (Teil) | Spieler-Filter im Backend erzwungen (`role`) | 12.10. | `ARCHITECTURE.md` 4.8 |
| 4.3 | Sauberes Beenden: `stop.ps1` (Strg+C an beide Fenster, nach 20 s hart, samt Ollama; Sperre während Ingest) und **⏻ LoreKeeper beenden** in der Sidebar (Admin). Gemessen: Strg+C ist in allen Lagen sauber (Backend 1,3–1,6 s, UI 0,2 s, auch bei Stream/Eval) — 7 von 10 Starts endeten durch Fenster-Schließen ohne Shutdown; die verzögerte Strg+C-Reaktion war nicht reproduzierbar (Verdacht: markierter Text im Konsolenfenster hält den Prozess an) | 13.10. | `docs/operations.md`, `docs/ui-ux.md` |
| 5.1 | Stage-Timings je Anfrage (Log, `done`-Event, Chat-Zeile) | 11.10. | `docs/operations.md` |
| 5.2 | Performance-Tests: Anfrage-Budgets, `pytest -m perf`, Startup-Baseline 17,7 s | 11./12.10. | `docs/testing.md` |
| 5.3 | Antwortzeit: Qwen3-Denken aus, Kontext 16k, Keep-alive 1 h, `top_rank` 3 — erster Token 8,7 → 1,1 s | 11./12.10. | `docs/provider-strategy.md`, `docs/configuration.md` |
| 5.4 | Reranker auf der GPU (DirectML): 840 → 50 ms, unter Last 14 s → 70 ms | 12.10. | `docs/embedding-strategy.md` |
| 5.5 | Vektorsuche exakt und reproduzierbar: `hnsw_ef_search` 100 → 1000 (später 2000, siehe 3.10) (0 statt 11 abweichende Golden-Fragen zwischen Prozessen, +5 ms/Frage, p105 gewonnen); Stolperfalle l2 beim Anlegen behoben; Prüfung „🎯 Suchgenauigkeit“ in der UI und Anleitung auf der Evaluation-Seite | 13.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
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
| 12.2 | Reranker rechnet in Batches zu 32 Paaren: 200 Kandidaten füllten neben qwen3:14b die GPU (Rerank 3–8 s, einmal DirectML-Fehler → bis zum Neustart CPU); jetzt ~0,8 s bei +0,5 GB, `top_k` 20 unverändert ein Batch (~90 ms) | 14.10. | `docs/embedding-strategy.md` |
| 12.1 | Ingest-Lücke: Ein Lauf über alle Quellen löscht Chunks von Quellen, die nicht mehr in `sources.yaml` stehen (vorher blieben sie durchsuchbar, 13.10.: 36 Chunks); nie bei leerer Quellenliste, nicht bei Einzel-Reindex; beide Schutzbedingungen mutationsgeprüft | 13.10. | `ARCHITECTURE.md` 4, `docs/data-flow.md`, `docs/configuration.md` |

**Live-Prüfungen (13.10., echtes LLM und echter Index)**: Ollama-Autostart findet die Modelle
(`OLLAMA_MODELS`); Folgefrage „fass das kürzer zusammen“ liefert eine kürzere Antwort zum selben
Thema (Verlauf wirkt; seit 3.12 passt auch die Quellenliste — live geprüft); Regelwerk-Slot im Chat bei x002 und x005 —
Regelbuch unter den Quellen, Antworten nennen die Regel; Rerank-Zeiten in der Chat-Zeile, warm
~70–250 ms auf der GPU; Chunk-Inspector und Feedback-Liste über die API; Warte-Hinweis bei zwei
gleichzeitigen Fragen; `pytest -m live` 283 grün (2:21 min).
