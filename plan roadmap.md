# LoreKeeper — Roadmap

Stand: 17.10.2026

Hier steht **offene Arbeit**. Erledigtes ist unten eine Zeile mit Verweis; Entscheidungen, Messwerte
und Begründungen stehen in der zuständigen Doku (Tabelle in `CLAUDE.md` → „Documentation“), die
Umsetzung im `git log`. Die Nummern bleiben stabil — die Doku verweist auf sie.

**Qualitätsphase 13.–14.10. abgeschlossen**, Nacharbeit 13.1–13.4 (Ultrareview, leere Chunks,
Regressionsprüfung, zwei Chat-Bugs) am 17.10. ebenfalls. Retrieval Datei 272/274, Abschnitt 143/149;
Antworten 245/274 im dritten End-to-End-Lauf (im Rauschen der ersten beiden); Holdout-Listen 9/10.
**Regressionsprüfung** nach größeren Inhaltsänderungen (Regelbuch, Vault): Golden Set +
End-to-End-Lauf + Holdout ([evaluation.md](docs/evaluation.md)), kein Tuning.

Neu priorisiert am 17.10.: **jeder offene Punkt unten steht in der Tabelle** — auch die Einzelfälle
aus 3.10/3.15, als eigene Zeile oder in einer Sammelzeile.

---

## Priorisierung

**Last** = Rechenlast beim Umsetzen/Messen: 🟢 nur Code + Unit-/UI-Tests (geht parallel zur
Simulation) · 🟡 braucht laufendes Backend oder LLM · 🔴 Indizierung oder Vision-Modell (CPU/GPU,
mehrere Minuten, nicht parallel zur Simulation). Ein Golden-Set-Lauf dauert ~1 min, ein
End-to-End-Lauf ~35 min (🟡).

| # | Punkt | Impact | Aufwand | Last | Begründung |
|---|---|---|---|---|---|
| 2 | 3.10 Rerank-Paar mit Dateiname/Überschrift — **nächster Punkt** (1 erledigt 17.10.) | Mittel (ungewiss) | Gering | 🟡 | Der Reranker sieht nur den Chunk-Text; kurze Chunks ohne Namen sind chancenlos (m048 „Deckname: Solmar“ — Chunk existiert, Antwort falsch). Allgemeiner Hebel, nur mit Golden Set + Holdout + E2E; `min_score` −4 des Regelwerk-Slots neu prüfen |
| 3 | 3.10 Schattenorden-Kontext (27 > 25 Chunks) | Gering-Mittel | Gering | 🟡 | Kernfrage zu einer Hauptfraktion (m003, m055) bekommt die Datei nicht mehr ganz; Treffer-Metriken sehen es nicht. Danach nach Punkt 2, weil 2 die Pool-Reihenfolge ändert |
| 4 | 3.15 Aufzählungen über große Kapitel | Mittel | Mittel-Hoch | 🟡 | Größte Gruppe echter Antwortfehler (7 Fragen: p008, p036, p037, p114, p014, a030, p049): Abschnitt gefunden, aber nur 8 Chunks erreichen das LLM. Holdout zeigt, dass kleine Listen gehen — Lösung muss gezielt sein (z. B. ganze Sektion für Listenfragen), Kontextlänge und Antwortzeit messen |
| 5 | 3.15 Tippfehler-Toleranz | Gering-Mittel | Mittel | 🟡 | Am Spieltisch wird auf dem Handy getippt; einziger offener 👎-Fall („üebrschaden“). Riskant (falsche Korrekturen) — nur gemessen, mit Wörterbuch der Schreibweisen |
| 6 | 14.1 Frage erneut senden | Gering | Gering | 🟢 | Idee des Nutzers. Echter Anwendungsfall: nach Fehler/Timeout (Ollama lädt, Backend-Neustart) dieselbe Frage ohne Abtippen; als neue Antwort wegen Temperatur 0,3 eher nicht. Vorschlag: Knopf nur unter Fehlermeldungen |
| 7 | 3.10 Tabellen-Einzelfälle (Dolch/Wurfzahl 12, Dolch vs. Kurzschwert, Langbogen-PDF-Zelle) | Gering | Mittel | 🟡 | Eintrags-Slot holt die Tabelle, jina wertet anderes höher; evtl. durch Punkt 2 mit erledigt — danach neu ansehen |
| 8 | 3.10 Monolith-Sektion in fremden Regelfragen (p011, p101) | Gering | Mittel | 🟡 | 22 Chunks, landet in den Top 3; Antworten bisher trotzdem richtig. Evtl. durch Punkt 2 mit erledigt |
| 9 | 2 Bilder — Rest (Vor-/Spitzname, Dateiname-Rückfall, PDF-Bilder, neues Bild ohne Notiz-Änderung) | Gering | Gering–Mittel | 🟢–🔴 | Einbettungen laufen. Vor-/Spitzname (13.4) über `aliases` der Notiz wäre klein (🟢); Dateiname-Rückfall braucht Spy-Test, PDF-Bilder dauerhafte Ablage + Re-Index (🔴) |
| 10 | 3.10/3.15 übrige Einzelfälle: AP-Pool „50 AP“, p009 (Abschnitt-Miss, Antwort richtig), „Wer ist Arkenfeld?“, k001 Preis-Spalte, p091 Tränke-Tabelle, x008 Heimkehrpfennig | Gering | je Gering | 🟡 | Nur bei Anlass im Spiel; Antworten teils schon richtig |
| 11 | Beim nächsten Re-Index: Suchgenauigkeit prüfen (m039) | Gering | Gering | 🔴 | Nach zwei In-place-Re-Indizes 273/274 exakt; bei weiteren Abweichungen Vault einmal komplett neu aufbauen (3.10) |
| — | 3.10 Simulationsberichte z001, z003, z006, z009, z012 | Gering | — | 🟢 | Liegt beim Nutzer: Überschriften in den Berichten (siehe „Offen beim Nutzer“), danach Golden Set |
| — | 3.5 Wikilink-Expansion ⏸ | Gering | Mittel | 🔴 | Gemessen 14.10.: kein Gewinn, −1/−2 Dateien; nur bei konkreten Link-Fragen neu prüfen |
| — | 4.2 Internet-Zugriff (Tailscale) ⏸ | Mittel | Gering | 🟢 | Nur zusammen mit einem Login für die API |
| — | 3.14 Rechenfragen ⏸ | Mittel | Mittel-Hoch | 🟡 | Nur mit Denken zuverlässig (14–41 s); später mit größerem Modell neu prüfen |
| — | 11.0 P8 Dienste-/Reranker-Modul | Gering | Mittel | 🟢 | Nur zusammen mit Feature-Arbeit an diesen Stellen (Punkt 2 berührt das Reranker-Modul — dann mitprüfen) |

**Entschieden 14.10.**: 3.2 Session-History gestrichen (Nutzer) — Sessions bleiben im RAM; bei einem
Wissenstool steht die Antwort im Regelbuch/Vault, nicht im Verlauf. Der Fähigkeitsbericht bleibt für Spieler freigegeben (Entscheidungshilfe).
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
  „Feedback“/„Chunks“ einmal von Hand bedienen (👍/👎 und Bilder vom Nutzer am 14./15.10. getestet).
  Neu zu sehen (13.4): „wer ist lady lilith?“ zeigt nur noch Lyssaras Bilder, nicht Lioras; der
  Kommentar wird erst mit Enter oder „Kommentar speichern“ gespeichert — Wegklicken speichert nichts.

---

## Offene Punkte

### 2 Bilder im Chat — Rest

Einbettungen in Notizen laufen seit 14.10. (siehe „Erledigt“ 2). Offen, nur bei Bedarf:
- **Dateiname als Rückfall** für Bilder, die nirgends eingebettet sind (40 Bilddateien, 20 Einbettungen):
  „Karte Arkenfeld.png“ passt zu Fragen, die „Arkenfeld“ nennen — wie der Eintrags-Slot (Namenssuche,
  Filter der Anfrage, Spy-Test).
- **Bilder aus PDFs** liegen nur in einem temporären Ordner, der nach dem Parsen gelöscht wird — zum
  Anzeigen müssten sie dauerhaft abgelegt werden.
- **Vor-/Spitzname** (13.4): Bilder erscheinen nur, wenn Frage oder Antwort den vollen Notiznamen
  nennt — „Lyssara“ allein reicht nicht. Weg: die `aliases` der Notiz mitprüfen.
- Ein neu abgelegtes Bild erscheint erst nach einer Änderung der Notiz oder einem vollen Re-Index der
  Quelle (der Inhalts-Hash deckt nur die Notiz ab).

Verworfen: Vision-Beschreibungen beim Ingest, CLIP-Embeddings, manuelle Captions; ein Endpunkt
`/files/{path}` ist unnötig, weil `st.image` die Datei auf dem Streamlit-Server liest (auch im Heimnetz).

### 3.5 Wikilink-Expansion ⏸

Gemessen am 14.10. (Index-Kopie, Golden Set + Holdout, Messung und Begründung in
[embedding-strategy.md](docs/embedding-strategy.md#warum-keine-wikilink-expansion)): Die verlinkten
Notizen der besten Treffer im Rerank-Pool bringen nichts — Golden Set Datei 272 → 270/271, Abschnitt
unverändert 143/149, Holdout unverändert. Alle sechs Fehlschläge liegen in Regelbuch und
Simulationsberichten ohne Links. **Nicht umgesetzt.** Neu prüfen nur, wenn Fragen über einen Link
(NPC → Gegenstand) nachweislich scheitern — dann mit Links je Abschnitt statt je Notiz.

### 4.2 Internet-Zugriff ⏸

**Tailscale** (privates Mesh-VPN) auf Gaming-PC und Laptop → `http://gaming-pc:8501` von überall,
ohne Portfreigabe. Alternative: Cloudflare Tunnel + Access. Offen: Der Gaming-PC muss laufen
(Wake-on-LAN braucht ein immer laufendes Gerät im Heimnetz). **Vorher**: Die API vertraut der
Rolle des Aufrufers und hat keinen eigenen Login ([ARCHITECTURE.md](ARCHITECTURE.md#8-architektur-entscheidungen)
→ „Bewusste Grenze“) — solange nur Streamlit sie erreicht, reicht das; Tailscale ändert daran
nichts, eine direkt erreichbare API schon.

### 3.10 Restliche Abschnitt-Misses — Einzelfälle

Golden Set 14.10. abends: Datei 272/274, Abschnitt 143/149. Suche 273/274 exakt — nach zwei
In-place-Re-Indizes des Vaults weicht m039 ab (Treffer unverändert); nach dem nächsten Re-Index erneut
prüfen, bei weiteren Abweichungen den Vault einmal komplett neu aufbauen. Umgesetzt und verworfen: Tabelle
„Erledigt“ und [embedding-strategy.md](docs/embedding-strategy.md). Offen, nur bei Anlass:
- **Simulationsberichte** z003, z006, z009, z012 (und z001 mit richtiger Antwort) — die Berichts-Chunks
  sagen weder „Schwierigkeit“ noch „Simulation“; besser in den Berichten selbst lösen (Überschriften).
- **p009** Zauberpunkte — Abschnitt-Miss im Golden Set, die Antwort ist seit dem Regelkern richtig.
- **Rerank-Paar ohne Namen** (Priorität 2): `_rerank` bewertet `[Frage, Chunk-Text]`; Dateiname und
  Aliase stehen nur im Embed-Text. Ein kurzer Kopfabschnitt wie „Deckname: Solmar · Faktion: Bund der
  Sonne“ (Escanor) nennt die Person nicht und verliert gegen jeden Text, der „Escanor“ enthält. Variante:
  `Dateiname | Überschrift` vor den Text im Rerank-Paar. Risiko: verschiebt alle Rerank-Scores
  (Regelwerk-Slot `min_score` −4 neu prüfen).
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
Offen:
- **Aufzählungen über große Kapitel** (Priorität 4) — der richtige Abschnitt wird gefunden, aber nur 8 Chunks
  erreichen das LLM: p008 Klassen, p036 Basis-Skills, p037 Basis-Zauber, p114 Eigenheiten Magie, p014
  Benchmark, a030 Beweis-Tabelle, p049 Überheilung.
- **👎 aus dem Chat (14.10.)** — 5 Bewertungen, 4 Fälle: drei behoben (siehe „Erledigt“ 3.15 Teil, Feedback);
  offen **Tippfehler** (Priorität 5) „was ist üebrschaden?“ — findet nichts Passendes, das Modell erfindet
  „Überschusschaden“ aus dem Simulationsbericht. Möglicher Weg: unbekannte Wörter auf das nächste Wort
  aus dem Wörterbuch der Schreibweisen abbilden (Abstand 1–2, lange Wörter) — riskanter, nur gemessen.
- **Fakten** (Priorität 10, nur bei Anlass): k001 Langschwert „20,5 Gold“ (Spalte „Preis“ der Simulations-Rangliste statt Preistabelle
  7.4); m048 Deckname „Solmar“ (Kopfabschnitt ohne Überschrift und Namen — 2 half nicht, siehe 3.10
  „Rerank-Paar ohne Namen“);
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


### 14.0 UX

- **14.1 Frage erneut senden** (Priorität 6, Idee des Nutzers 15.10.): Knopf, der die letzte Frage
  noch einmal schickt, statt sie neu zu tippen. Sinnvoll vor allem nach einem Fehler oder Timeout
  (Ollama lädt noch, Backend-Neustart) — dort steht heute nur die Fehlermeldung. Für „noch eine
  Antwort zur selben Frage“ eher nicht: die Antworten unterscheiden sich bei Temperatur 0,3 kaum,
  und Folgefragen decken „ausführlicher“/„kürzer“ ab. Vorschlag: Knopf nur unter Fehlermeldungen.

## Eingang (neue Bugs und Ideen, noch nicht bewertet)
- —

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
| 3.15 (Teil) | 👎 aus dem Chat ausgewertet (5 Bewertungen, 4 Fälle, alle live nachgestellt): **Folgefragen** wie „bitte mehr infos“ wurden nicht umgeschrieben — „Info/Infos/Informationen/erzähl/hast“ ergänzt, Tests in beide Richtungen, keine Golden-Frage neu als Folgefrage; **Kleinschreibung** kostete systematisch (Golden Set klein 143 → 135/149) — klein getippte Wörter vor der Suche in die Schreibweise der Quellen (`src/retrieval/truecase.py`, nur intern): klein 135 → 142, wie geschrieben unverändert 143, Holdout klein 6 → 7/8; **Klassenübersicht** vom Nutzer im Regelbuch ergänzt (5.4/5.5), neu indiziert: „welche klassen gibt es?“ nennt alle 12. Offen: Tippfehler | 14.10. | `docs/embedding-strategy.md`, `docs/prompts.md`, `ARCHITECTURE.md` |
| 12.4 | `stop.ps1` beendete den Browser: Windows vergibt PIDs neu, der Browser, den `start.ps1` geöffnet hatte, galt als Kind des UI-Prozesses und wurde nach 20 s hart beendet. Kinder zählen nur noch, wenn sie nach dem Elternprozess gestartet wurden; neu `.\stop.ps1 -List` zeigt die Prozesse, ohne zu stoppen | 14.10. | `docs/operations.md` |
| 12.3 | Auto-Reload hing (dreimal am 14.10.), nachgestellt mit drei Ursachen: **(1)** Uvicorn startet den Worker unter Windows per Strg+C an die ganze Konsole neu — das traf auch das vom Backend gestartete Ollama (laufende Antwort → 503). Jetzt eigene Prozessgruppe, Beenden per Strg+Pause samt `llama-server` (1 s, Modell nicht mehr im VRAM). **(2)** Ein laufender End-to-End-Lauf (`asyncio.to_thread`) hielt den alten Worker am Leben — Python wartet beim Beenden auf Pool-Threads, der Lauf fragt den eigenen, nicht mehr antwortenden Server mit 120 s je Frage, und der Reloader wartet ohne Grenze: kein neuer Worker. Jetzt Daemon-Thread (`run_in_daemon_thread`, mutationsgeprüft); Indizierung bleibt bewusst ein Pool-Thread (wird fertig geschrieben). **(3)** `stop.ps1` während eines Neustarts: der Reloader startete noch einen Worker, der kein Strg+C bekam und als Waise Port 8000 hielt — `stop.ps1` sucht vor dem harten Beenden erneut nach Kindprozessen. Live geprüft: Neustart während E2E-Lauf mit geladenem Modell → Antwort fertig, Ollama + Runner beendet, neuer Worker nach 13 s; Stopp während Neustart-Serie → Port frei, keine Waisen. Der 17:15-Fall war keiner: Änderung in `evaluation/`, das nicht überwacht wird | 14.10. | `docs/operations.md` |
| 2 | Bilder im Chat ohne Bildbeschreibung (Ansatz des Nutzers): Der Markdown-Parser behält `![[bild.png]]` / `![[bild.png\|300]]` je Abschnitt; **Titelbilder** (vor der ersten Unterüberschrift) hängen an jedem Abschnitt der Notiz — nur je Abschnitt bekam live keine Frage zu Malek/Arkenfeld ein Bild. Der Orchestrator löst die Namen **innerhalb derselben Quelle** zu Pfaden auf (`images`), `sources` liefern sie, der Chat zeigt „🖼️ Bilder (n)“ (zugeklappt, max. 6, nur existierende Dateien). Spieler-Sicherheitstest mit Köder-Bild in einer Spielleiter-Quelle, mutationsgeprüft (quellübergreifende Auflösung → 7 Fehlschläge). Zwei volle Re-Indizes des Vaults (je ~8 min); Golden Set unverändert Datei 272/274, Abschnitt 143/149. Live: Malek → Porträt, Arkenfeld-Karte → `arkenfeld.png`, Isolde Vaer → 2 Bilder. Suche danach 273/274 exakt (m039 weicht ab, Treffer unverändert) — nach weiteren In-place-Re-Indizes neu prüfen. Rest offen (Dateiname-Rückfall, PDF-Bilder) | 14.10. | `docs/parsing.md`, `docs/ui-ux.md`, `ARCHITECTURE.md`, `docs/embedding-strategy.md` |
| 3.4 | Kommentar zur Bewertung: nach 👍/👎 ein optionales Feld (Enter speichert, ≤ 1.000 Zeichen, API antwortet sonst 422); jede Zeile in `logs/feedback.jsonl` trägt Bewertung und Kommentar zusammen, Zurücknehmen löscht ihn. Tab „Feedback“ mit Spalte „Kommentar“, beim Übernehmen ins Golden Set landet er in der Notiz. UI-, API- und Grenztests; kein zusätzlicher Backend-Aufruf beim Anzeigen | 14.10. | `docs/ui-ux.md`, `docs/operations.md`, `ARCHITECTURE.md` |
| 13.1 | Ultrareview 14.10. (Änderungen seit 1ba1f49): 10 Befunde zu Bildern und `stop.ps1`, 8 korrigiert (Einbettung mit Pfad vor Name, Ersatz-Chunk mit Bildern, Titelbilder nur mit echter H1, Frontmatter-Schlüssel `images`/`image_embeds` reserviert, Einbettungen aus Überschriften, Bild-Suche im Umfang der Indizierung — **Spieler-Invariante**, Sicherheitstest mit Köder-Bild, mutationsgeprüft —, `stop.ps1 -List`, ein Helfer für BM25 + Truecaser); #7 und #9 mit Grund verworfen (kein Fall für einen Handler, kein Start ohne Konsole). Voller Re-Index, Golden Set unverändert 272/274 · 143/149; live 15.10.: `stop.ps1 -List` und Bilder im Chat | 14./15.10. | `docs/parsing.md`, `docs/testing.md`, `docs/operations.md` |
| 13.2 | Leere Chunks: 251 von 4414 waren nur `---` (dazu Block-IDs, Sim-Kommentare, Tag-Zeilen, leere Notizen, abgeschnittene Beschriftungen) — Parser entfernt sie, Chunker hängt Stücke unter `min_chunk_size` an das nächste und verwirft Stücke ohne Buchstaben/Ziffern. Index 4414 → 3933 Chunks, 0 ohne Inhalt, Golden Set unverändert, doppelt belegte Plätze 738 → 556 | 14.10. | `docs/parsing.md`, `docs/embedding-strategy.md`, `ARCHITECTURE.md`, `docs/configuration.md`, `docs/data-flow.md` |
| 13.3 | Regressionsprüfung nach 13.1/13.2: E2E 245/274 (0,894, vorher 248/247 — Rauschen), Datei 0,993, Ø 7,1 s; 6 der 19 Dauer-Fehler jetzt richtig, 7 neue gesichtet (5 nur Stichwort, i002 frühere Schein-Treffer-Absage, p035 schwankt) — keine Regression. Holdout 9/10, 50/53; live keine leeren Chunks mehr. Stichwörter erledigt 17.10. (3.15 Teil) | 15.10. | `docs/evaluation.md` |
| 13.4 | Zwei Chat-Bugs vom 15.10.: 👎-Kommentar wurde bei jedem Verlassen des Felds halb gespeichert (`st.text_input` speichert beim Fokusverlust) → Feld in `st.form`, nur Enter/Knopf speichert; fremdes Porträt unter der Antwort (Liora Veylan als schwache Quelle Nr. 6) → Bilder nur von Notizen, deren Name in Frage oder Antwort steht (Entscheidung des Nutzers). Beide UI-Tests gegen den alten Code rot. Rest: live ansehen, Vor-/Spitzname (2 Rest) | 17.10. | `docs/ui-ux.md` |
| 3.15 (Teil) | Golden-Set-Stichwörter aus 13.3 (k015, a014, a005, m015, p026): Antworten richtig, ohne den Namen aus der Frage — je eine Tatsache, die alle drei Läufe nennen und die in der Quelle steht. Dritter Lauf neu ausgewertet 245 → 250/274 (0,912), nur diese fünf kippen; die Läufe vom 14.10. speicherten nur 200 Zeichen je Antwort und lassen sich nicht neu auswerten | 17.10. | `docs/evaluation.md` |
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
