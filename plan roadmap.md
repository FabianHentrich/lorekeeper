# LoreKeeper — Roadmap

Stand: 14.10.2026

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
| 1 | 13.0 Korrekturen aus dem Code-Review (Befund 1 kritisch) | Hoch — lange Frage legt das Backend lahm | Gering | 🟢 | Kappung + API-Grenze, Tests; danach Golden Set (🟡) |
| 2 | 3.10 Restliche Abschnitt-Misses | Gering-Mittel — Regelfragen präziser | Mittel | 🟡 | Einzelfälle, je Golden-Set-Lauf prüfbar |
| 2 | 3.15 Antwortfehler aus dem End-to-End-Lauf | Mittel — 19 Fragen scheitern stabil | Mittel | 🟡 | Größte Gruppe: Aufzählungsfragen (Retrieval) |
| — | 3.14 Rechenfragen zum Regelwerk ⏸ | Mittel — falsche Zahlen bei Regel-Rechnungen | Mittel-Hoch | 🟡 | Vorerst so gelassen (14.10.); nur mit Denken zuverlässiger, das kostet 15–40 s |
| — | 3.2 Session-History ⏸ | Unklar | Mittel | 🟢 | Erst entscheiden, ob gespeicherte Sessions Nutzen haben |
| 3 | 3.4 Kommentar zur Bewertung ⏸ | Gering | Gering | 🟢 | Feature |
| — | 2 Bilder im Chat (Phase 1/2) ⏸ | Mittel / Nice-to-have | Mittel / Mittel-Hoch | 🔴/🟡 | Feature |
| — | 4.2 Internet-Zugriff (Tailscale) ⏸ | Mittel | Gering | 🟢 | Vorher die Vertrauensgrenze der API klären |
| 4 | 3.5 Wikilink-Expansion ⏸ | Mittel | Mittel | 🔴 | Neuer Suchweg — muss in den Spy-Test |
| — | 11.0 P8 Dienste-/Reranker-Modul (optional) | Gering | Mittel | 🟢 | Nur zusammen mit Feature-Arbeit |

**Entschieden 14.10.**: Der Fähigkeitsbericht bleibt für Spieler freigegeben — er ist eine
Entscheidungshilfe (Spieler bekommen bei 47 von 124 reinen Regelfragen Ø 2,3 von 8 Chunks daraus).
`Spielleiter-Tools/potenzielle_anpassungen.md` (Änderungsvorschläge/Errata) wird nicht mehr
indiziert (`exclude_patterns` in `sources.yaml`, 14 Chunks entfernt) — Vorschläge sind keine
geltenden Regeln. `HF_HUB_OFFLINE=1` ist gesetzt (in `src/__init__.py`, weil `.env` die
Modell-Bibliotheken nicht erreicht): Modelle nur aus dem lokalen Cache, kein Hub-Zugriff beim Start;
Downloads mit `HF_HUB_OFFLINE=0` ([operations.md](docs/operations.md#modelle-fehlen-beim-ersten-start)).

**Offen am Gaming-PC (Nutzer)**: (DHCP-Reservierung für `192.168.178.34` eingestellt, 14.10.) Die alte
Autostart-Aufgabe „LoreKeeper Server“ war am 13.10. ohne Admin-Rechte nicht mehr zu sehen — einmal
in einer Admin-PowerShell bestätigen (`Get-ScheduledTask -TaskName "LoreKeeper*"`). Nebenbei
gesehen: Inbound-Allow-Regeln für `python.exe` im Profil „Öffentlich“ gehören zu anderen
Python-Installationen (alte LoreKeeper-Kopie unter OneDrive, PortraitForge, globales Python) — ggf.
selbst aufräumen.

**Offen im Browser** (Handprüfung): **⏻ LoreKeeper beenden** einmal klicken — `logs/stop.log` muss
nach Sekunden mit „stopped … (clean)“ enden (der korrigierte Startweg wurde nur mit einem
Ersatz-Prozess geprüft, 12.0); bei der nächsten Indizierung über die UI den Fortschritt
durchklicken (Job starten, Seite wechseln und zurück — 3.8/3.9); 👍/👎 und die Tabs
„Feedback“/„Chunks“ einmal von Hand bedienen (API-Seite live geprüft).

---

## Offene Punkte

### 3.10 Restliche Abschnitt-Misses

Stand 14.10.: Datei 272/274, Abschnitt **143/149**, Suche exakt. Bisher umgesetzt (Tabelle
„Erledigt“): Regelwerk-Slot, `top_k` 20, „Alte Vault-Werte“ aus dem Index, Zeilennamen für
Tabellen-Chunks, Reranker jina-reranker-v2, Eintrags-Slot; verworfen: Hybrid-Suche, Tabellen als
Fließtext, Tabellenzeilen als eigene Chunks, `top_k` > 20.

**Offen, je mit Diagnose** (Rang des richtigen Chunks in der Vektorsuche, Reranker-Score richtig
gegen gewählt; Skript-Muster: Rang in den Top 200 + `_predict` je Paar):
- **Seit jina verloren** (diagnostiziert 14.10., Kandidaten mit jina und mMiniLMv2 nachgerechnet):
  - **p009 Zauberpunkte — Antwort war falsch, über den Regelkern behoben (14.10.).** jina setzt „5.1
    Fähigkeitspunkte“ auf Rang 15 von 20 (mMiniLMv2: 7) und füllt den Kontext mit Ressourcen-Chunks
    (2.5 AP/MP, LP, RP); die Antwort hielt Zauberpunkte für verbrauchte, regenerierte Ressourcen.
    Neuer Regelkern-Stichpunkt „Ressourcen vs. Punkte“: Antwort 0/3 → 3/3, MP-/Talentfragen
    unverändert richtig ([prompts.md](docs/prompts.md#regelkern)). Im Golden Set (nur Retrieval) bleibt
    p009 ein Abschnitt-Miss — der Chunk nennt „Zauberpunkte“ nur in Klammern.
  - **z001 Begegnungsschwierigkeit — Antwort trotzdem richtig** (leicht < 15 % … tödlich ≥ 50 %, aus
    Begegnungsbaukasten/Bericht). jina setzt Abenteuer-Szenen („Gefährlicher Wald“) nach vorn; die
    Berichts-Chunks sagen weder „Schwierigkeit“ noch „Simulation“, und „4. Die Einstufung“ steht in
    einem anderen Chunk als die Tabelle mit den Schwellen.
- **AP-Pool** (14.10., nebenbei gefunden): „Wie viele Aktionspunkte habe ich?“ → „50 AP“ (0/3, mit
  und ohne Regelkern) — die 50 sind ein gemeinsamer Pool für AP und MP (2.5).
- **Simulation** — z003, z006, z009, z012 (Fragen zu Berichtsabschnitten).
- **Name in fremder Identität** — „Wer ist Arkenfeld?“ (live 13.10.): 12 Chunks von „Bargrun
  ‚Zwerg‘ Erkenbrandt“ („Der Armdrücker von Arkenfeld“), nur 5 von `Orte/Arkenfeld.md`. Die
  Identitäts-Schicht macht den Ortsnamen zum Suchsignal für die NPC-Datei; Antwort trotzdem richtig.
- Die 22-Chunk-Sektion „7.3 Legendäre Artefakte > Monolith“ landet bei fremden Regelfragen in den
  Top 3 („Ass-Regel“ p011, „Zauberspeicher“ p101); das Budget aus 5.6 streicht sie nur.
- **Tabelleneinträge** — seit 14.10. über den Eintrags-Slot (Tabelle „Erledigt“). Offen: „Dolch, STÄ 4,
  Wurfzahl 12“ und „Unterschied Dolch und Kurzschwert“ (Slot holt die Tabelle, jina wertet anderes
  höher), „Langbogen“ (PDF-Zelle `**l**<br>**Langbogen**`, Parser-Artefakt). Der Rechenteil gehört zu 3.14.

### 3.15 Antwortfehler aus dem End-to-End-Lauf (14.10.)

Zwei Läufe: Answer Contains 0,905 und 0,901, aber 15 Fragen kippen zwischen den Läufen (Rauschen bei
Temperatur 0,3). Die 19 Fragen, die in **beiden** scheitern, mit ganzen Antworten nachgefragt
([evaluation.md](docs/evaluation.md#evaluatepy--end-to-end)):

- **Stichwort zu eng, Antwort richtig** — am 14.10. angepasst (a007 „Ephraskal“, a012 „Monolith“, m042
  „physisch“, p009 „Abenteuer“, p010 „Rang 1“, p079 „halbe“ + „10+“; geprüft gegen die gespeicherten
  Antworten). Vorher: a007 („Mandate der Macht“ — Antwort erklärt
  das Ziel ohne den Akt-Titel), a012 („Gewahrsam“ — Abenteuername), m042 („Berechtigung“ — Phasen richtig
  beschrieben), p009 („Fähigkeitspunkte“ — Antwort richtig, sagt „Zauberpunkte … nicht Manapunkte“),
  p010 („Fähigkeitspunkt“ — Antwort sagt „Zauberpunkt“, wie das Regelbuch auch), p079 („keine Kosten“ —
  Antwort „keinen Kosten“, inhaltlich vollständig).
- **Aufzählungen unvollständig** (größte Gruppe, Retrieval): p008 Klassen („keine konkreten Klassen
  genannt“), p036 Basis-Skills (nur einer), p037 Basis-Zauber („Magischer Schlag“ fehlt), p114 Eigenheiten
  Magie („Manaschwäche“ beschrieben, aber ohne Namen), p014 Benchmark-Werte fehlen, a030 Beweis-Tabelle
  nicht gefunden, p049 Überheilung (Seelenentzug fehlt, dafür geratene Zauber aus NPC-Dateien).
- **Fakten falsch**: k001 Langschwert „20,5 Gold“ — Zahl aus der Fähigkeiten-Rangliste der Simulation
  (Spalte „Preis“ = Kosten je Effekt), die Preistabelle 7.4 fehlt im Kontext; m048 Deckname „Solmar“
  (steht ohne Überschrift neben einem Bild); p091 Tränke-Fehlwurf (1W8-Tabelle fehlt).
- **x008 Heimkehrpfennig** — Antwort „bei Thalion, Endzustand B“ (Kampagnenstruktur), erwartet
  „Kultpriester“ (Beutetabelle I-2); Nutzer 14.10.: „Kultpriester“ gilt — echter Antwortfehler. z009 ist ein bekannter
  Retrieval-Miss (3.10).

### 3.14 Rechenfragen zum Regelwerk unzuverlässig ⏸

Entdeckt 14.10. beim Regelkern (3.13). Fragen, bei denen das Modell Regeln **anwenden und rechnen**
muss, beantwortet qwen3:14b ohne Denken oft falsch, auch wenn Regel und Werte im Prompt stehen. Live
mit jina, je 3 Läufe ([prompts.md](docs/prompts.md#regelkern)):
- „Welche Wurfzahl kann ich mit Rang 1 höchstens erreichen?“ — Lead „höchstens 4“ bzw. „13“, richtig
  wäre „nach oben offen“ (Ass-Regel). Das Modell hängt sich an „höchstens“ + W4; das Ass-Regel-Beispiel
  im Chunk ist nicht die Ursache (ohne es 0/3). Starke Streuung zwischen Läufen (0–3/3).
- „Dolch, STÄ 4, Wurfzahl 12“ — 0/3: Krit-Schwelle (15 statt 10 bei leichten Waffen) und Grundschaden
  verwechselt; dazu fehlt die Waffentabelle im Kontext (3.10).
- „Schattenklinge auf Rang 6 maximal?“ — mit jina kommt die Fähigkeitskarte (1W10 + X), aber X wird
  falsch oder gar nicht bestimmt (richtig: + 10). Mit mMiniLMv2 kam stattdessen der
  Simulations-Durchschnitt 29,94 als „Maximum“.
- Richtig dagegen: Rüstungswerte (3/3), Erklärung des Würfelsystems (3/3).

**Gemessen: Qwen3-Denken** (nur im Test): Rang 1 3/3 richtig, Dolch 1/3 richtig und 2/3 leere
Antworten (Token-Budget aufgebraucht), Schattenklinge 3/3 „X nicht angegeben“; 14–41 s statt
2–5 s pro Antwort — als Standard ungeeignet.

**Entschieden 14.10.: vorerst so lassen.** Mögliche Wege für später: Denken nur für Rechenfragen
(braucht eine Erkennung solcher Fragen und mehr `max_tokens`, Antworten 15–40 s); größeres Modell;
Waffentabelle für Waffenfragen zuverlässig in den Kontext (3.10). Vorher eine feste Liste von
Rechenfragen mit Sollwerten anlegen, damit Varianten über mehrere Läufe vergleichbar sind.

### 13.0 Code-Review seit 194be1b — Korrekturplan (14.10.)

Review der vier Commits (jina, Regelkern, Batches, Eintrags-Slot) durch den Review-Agenten; Befund 1
selbst nachgestellt (eigener Prozess, In-Memory-Chroma).

| # | Befund | Schwere | Status | Korrektur |
|---|---|---|---|---|
| 1 | Lange Frage legt das Backend lahm: der Eintrags-Slot macht aus jedem Wort und jeder 2–3-Wort-Folge eine `$contains`-Klausel, ohne Obergrenze. 4.000 Zeichen (1.081 Namen) → `InternalError: Expression tree is too large` (500), 12.000 Zeichen → Prozess stirbt ohne Log (Stack Overflow) — im Heimnetz-Modus für jeden Spieler auslösbar | **Kritisch** | nachgestellt | Namen kappen (z. B. 60) + `max_length` für `question` in `QueryRequest`/`RetrievalPreviewRequest` (Grenze an der API, 422), Test mit sehr langer Frage; Golden Set einmal nachmessen |
| 2 | Kosten der Namenssuche ∝ Namen × Chunks (je Klausel ein Durchlauf über die Collection): bei 20k Chunks ~1 s für eine normale Frage (synthetisch gemessen) | Mittel | gemessen (synthetisch) | Kappung aus 1 begrenzt es; in `embedding-strategy.md` festhalten, bei wachsendem Index neu messen |
| 3 | Eintrags-Treffer umgehen `score_threshold` — gewollt (sie werden gefunden, weil ihr Vektor schwach ist), aber undokumentiert | Gering | Code gelesen | Satz in der Doku + Kommentar, Test mit Score unter der Schwelle |
| 4 | Veraltete Docstrings: `retriever.py` (Modul, `retrieve()` — Eintrags-Slot und Stufe `entry` fehlen), Lock-Kommentar in `onnx_reranker.py` („GPU-Queue“, gilt jetzt auch für die CPU) | Gering | Code gelesen | nachziehen |
| 5 | Doku-Kleinigkeiten: `onnx_file` fehlt in der UI-Tabelle, drei verschiedene Rerank-Zeiten ohne Kontext (26/90/110 ms), „7 Tests rot“ (Agent maß 5 + 1) | Gering | Code gelesen | angleichen |
| 6 | Regelkern geht an jede Rolle — heute unkritisch (Regelbuch spielerfreigegeben), ein künftiger Spielleiter-Inhalt dort würde aber ungeprüft an Spieler gehen | Info | — | Satz in `prompts.md`: nur spielerfreigegebene Inhalte |

Als Fehlalarm ausgeschlossen: Sonderzeichen in `$contains` (keine Injektion, `%`/`_` keine Platzhalter),
Lock pro Batch, GPU→CPU-Rückfall-Race, Reihenfolge von `HF_HUB_OFFLINE`, alter Wert `backend: torch`.

Dazu aus dem Performance-Review: `start.ps1` hielt `TimeWait`-Reste geschlossener Verbindungen für einen
laufenden Server und startete das Backend nicht — behoben (nur `-State Listen`, wie `stop.ps1`).

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
Vektorraum, eigene Collection), manuelle Captions (selten vorhanden). -> ist das überhaupt nötig? beschreiben der Bilder scheint mir ein unnötiger schritt zu sein. einfacher wäre es durch Metadaten und dateiname die ingesten zu machen und dann das zuzurodnen. Zb datei heißt Karte Arkenfeld. dann sollte es bei fragen zur Karte von AKrtenfeld oder generell bei Arken feld auftauchen.  Genauso bei Charakteren die dann im dateinamen auftauchen oder gegeständen.

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


## xx.0 unter quelle die Chunks öffnen und einsehbar machen
Idee wenn man es nochmal aus der Quelle nachlesen will kann man das einfach in den quellen amchen. dazu brauchen wir keinen eigenen viewer da nur die chunks, die sowieso geladen wurden, da einsehbar sind. macht das sinn?
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
