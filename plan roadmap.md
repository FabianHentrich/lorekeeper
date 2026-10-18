# LoreKeeper — Roadmap

Stand: 18.10.2026

Hier steht **offene Arbeit**. Erledigtes ist unten eine Zeile mit Verweis; Entscheidungen, Messwerte
und Begründungen stehen in der zuständigen Doku (Tabelle in `CLAUDE.md` → „Documentation“), die
Umsetzung im `git log`. Die Nummern bleiben stabil — die Doku verweist auf sie.

**Stand 18.10.**: Qualitätsphase (13.–18.10.) abgeschlossen, **keine offene Code-Arbeit**. Retrieval
Datei 272/274, Abschnitt 144/149; Antworten 261/274 im End-to-End-Lauf (Ø 7,2 s je Frage); Holdout-Listen
9/10 · 51/53; Suche 274/274 exakt. Die naheliegenden Retrieval-Ideen sind gemessen — drei davon am
18.10. verworfen (Deckel je PDF-Kapitel, Kopfzeilen entfernen, Bilder per Dateiname). Danach am 18.10.
der Vault-Inhalt nachgezogen (Schattenorden „vier Pfade“, Fundort des Heimkehrpfennigs beim
Kultpriester, 19 Bilder eingebettet) und indiziert; Golden Set danach 272 · 144, Platz 1 134, Suche
274/274 exakt; Regressionsprüfung (12) ohne Rückschritt. Handprüfungen im Browser und System-Punkt vom
Nutzer erledigt. UI-Review 13.6 am 18.10. umgesetzt und live geprüft; Maskenball-Alias eingetragen.
Offen nur Nutzer-Inhalt (k001).

**Regressionsprüfung** nach größeren Inhaltsänderungen (Regelbuch, Vault): Golden Set + Suchgenauigkeit
+ End-to-End-Lauf + Holdout ([evaluation.md](docs/evaluation.md)), kein Tuning. Weicht die
Suchgenauigkeit nach In-place-Re-Indizes ab, den Vault einmal komplett neu aufbauen (3.10).

---

## Priorisierung

**Last** = Rechenlast beim Umsetzen/Messen: 🟢 nur Code + Unit-/UI-Tests (geht parallel zur
Simulation) · 🟡 braucht laufendes Backend oder LLM · 🔴 Indizierung oder Vision-Modell (CPU/GPU,
mehrere Minuten, nicht parallel zur Simulation). Ein Golden-Set-Lauf dauert ~1 min, ein
End-to-End-Lauf ~35 min (🟡).

| # | Punkt | Impact | Aufwand | Last | Begründung |
|---|---|---|---|---|---|
| — | keine offenen Punkte (18.10.) | | | | |

**Pausiert** (Begründung je Abschnitt unten):

| # | Punkt | Impact | Aufwand | Wieder aufnehmen, wenn … |
|---|---|---|---|---|
| 3.14 | Rechenfragen zum Regelwerk | Mittel | Mittel–Hoch | ein größeres Modell passt oder „Denken nur für Rechenfragen“ angegangen wird — größte bekannte Qualitätslücke |
| 4.2 | Internet-Zugriff (Tailscale) | Mittel | Gering | es einen Login für die API gibt |
| 3.5 | Wikilink-Expansion | Gering | Mittel | Fragen über einen Link (NPC → Gegenstand) nachweislich scheitern |
| 7c | Auto-Reload/Stopp hing bei laufendem E2E-Lauf | Gering | — | es wieder passiert (Prüfliste unten) |
| 11.0 P8 | Dienste-/Reranker-Modul | Gering | Mittel | Feature-Arbeit diese Stellen ohnehin anfasst |

**Entschieden 14.10.**: 3.2 Session-History gestrichen (Nutzer) — Sessions bleiben im RAM; bei einem
Wissenstool steht die Antwort im Regelbuch/Vault, nicht im Verlauf. Der Fähigkeitsbericht bleibt für
Spieler freigegeben (Entscheidungshilfe). `Spielleiter-Tools/potenzielle_anpassungen.md` wird nicht
indiziert — Vorschläge sind keine geltenden Regeln. `HF_HUB_OFFLINE=1` (in `src/__init__.py`): Modelle
nur aus dem lokalen Cache, Downloads mit `HF_HUB_OFFLINE=0`
([operations.md](docs/operations.md#modelle-fehlen-beim-ersten-start)).

---

## Offen beim Nutzer

Erledigtes (Vault-Inhalt, System, Handprüfungen im Browser) steht unter „Erledigt“ → „Nutzer 18.10.“.

- [ ] **Regelbuch 7.4, k001** „Wie viel kostet ein Langschwert?“ — der Normalpreis steht nirgends direkt
  (Langschwert = schwere Waffe in 7.1.1, schwere Waffe = 15 G in 7.4); der einzige Satz mit „Langschwert“
  ist das Beispiel „meisterlich 75 Gold“, die Antwort nennt 75. Vorschlag: „Ein Langschwert (schwere
  Waffe) kostet 15 Gold, meisterlich 75.“ Danach das Regelbuch neu indizieren, Golden Set + k001 3×
  (Regressionsprüfung wie in 12).
- [ ] **Simulationsberichte** (z001, z003, z006, z009, z012) — **später, eigenes Thema** (18.10.): die
  Berichte erzeugt der Generator im Repo PnP_Simulation, Überschriften also dort ändern. Die
  Berichts-Chunks sagen weder „Schwierigkeit“ noch „Simulation“; fragenahe Überschriften und „Kosten je
  Effekt“ statt „Preis“ (3.15).

---
## Offene Punkte

### Feedback durchgehen für verbesserung des Outputs

### Bug: Anfrage ans LLM und gleichzeitig die Antwort davor bewerten mit einem kommentar bricht die Anfrage ans LLM ab


## Pausierte Punkte

### 3.14 Rechenfragen zum Regelwerk ⏸

Fragen, bei denen das Modell Regeln **anwenden und rechnen** muss (höchste Wurfzahl mit Rang 1,
Dolch-Schaden bei Wurfzahl 12, Schattenklinge auf Rang 6), beantwortet qwen3:14b ohne Denken oft
falsch, auch wenn Regel und Werte im Prompt stehen (18.10. live: Dolch STÄ 4, Wurfzahl 12 → „18“).
Mit Qwen3-Denken richtiger, aber 14–41 s pro Antwort und teils leere Antworten (Token-Budget) — als
Standard ungeeignet. Messungen: [prompts.md](docs/prompts.md#regelkern). **Entschieden 14.10.: vorerst
so lassen.** Später: Denken nur für Rechenfragen (braucht eine Erkennung und mehr `max_tokens`) oder ein
größeres Modell (VRAM: qwen3:14b belegt 11,4 von 16 GB); vorher eine feste Liste von Rechenfragen mit
Sollwerten anlegen.

### 4.2 Internet-Zugriff ⏸

**Tailscale** (privates Mesh-VPN) auf Gaming-PC und Laptop → `http://gaming-pc:8501` von überall,
ohne Portfreigabe. Alternative: Cloudflare Tunnel + Access. Offen: Der Gaming-PC muss laufen
(Wake-on-LAN braucht ein immer laufendes Gerät im Heimnetz). **Vorher**: Die API vertraut der
Rolle des Aufrufers und hat keinen eigenen Login ([ARCHITECTURE.md](ARCHITECTURE.md#8-architektur-entscheidungen)
→ „Bewusste Grenze“) — solange nur Streamlit sie erreicht, reicht das; Tailscale ändert daran
nichts, eine direkt erreichbare API schon.

### 3.5 Wikilink-Expansion ⏸

Gemessen am 14.10. (Index-Kopie, Golden Set + Holdout, Messung und Begründung in
[embedding-strategy.md](docs/embedding-strategy.md#warum-keine-wikilink-expansion)): Die verlinkten
Notizen der besten Treffer im Rerank-Pool bringen nichts — Golden Set Datei 272 → 270/271, Abschnitt
unverändert 143/149, Holdout unverändert. Alle sechs Fehlschläge liegen in Regelbuch und
Simulationsberichten ohne Links. **Nicht umgesetzt.** Neu prüfen nur, wenn Fragen über einen Link
(NPC → Gegenstand) nachweislich scheitern — dann mit Links je Abschnitt statt je Notiz.

### 7c Auto-Reload/Stopp hing bei laufendem E2E-Lauf ⏸

18.10. 00:00: eine Änderung unter `src/` löste keinen Neustart aus, `stop.ps1` musste nach 20 s hart
beenden. In beiden Fällen fehlt „Shutting down“ im Log, der Worker beantwortete noch 19 s lang Fragen —
er hat das Strg+C nicht verarbeitet (Prozessbaum intakt, `GenerateConsoleCtrlEvent` erfolgreich); offene
Verbindungen erklären es nicht, die werden erst nach „Shutting down“ abgewartet. Der Rechner war in der
Nacht überlastet (Modell im geteilten GPU-Speicher, wenig RAM). Nachgestellt 18.10. 11:33 auf freiem
Rechner: Reload während E2E nach 39 s fertig (wartet die laufende Frage ab), `stop.ps1` während E2E
sauber nach 8 s — 12.3 hält. **Falls es wieder passiert**: im Log nach „Shutting down“ suchen, im
Backend-Fenster nach „WatchFiles detected changes“ (steht nur dort, nicht in der Datei), Fenster-Titel
auf „Auswählen“ prüfen (Markiermodus hält Konsolen-Ausgaben an).

### 11.0 P8 — optional

Dienste-Modul `src/services.py` (Globale, `install_llm`, Health-Loop; beseitigt die Lazy-Imports
`import src.main as main_module` und vereinfacht die Test-Fixtures) und Reranker + GPU-Rückfall als
`src/retrieval/reranker.py`. Nur zusammen mit Feature-Arbeit, die diese Stellen ohnehin anfasst.

---

## Bekannte Restfehler (ohne Handlungsbedarf)

Golden Set und End-to-End-Lauf haben Fehler mit je eigener Ursache; zwischen zwei gleichen E2E-Läufen
kippen ~15 Fragen (Rauschen, [evaluation.md](docs/evaluation.md#evaluatepy--end-to-end)). Feintuning ist
seit 14.10. beendet. Bekannt und bewusst offen:
- **p009** Zauberpunkte — Abschnitt-Miss im Golden Set, die Antwort ist seit dem Regelkern richtig.
- **„Wer ist Arkenfeld?“** — holt vor allem „Der Armdrücker von Arkenfeld“; Antwort trotzdem richtig.
- **a030** (Beweis-Tabelle im Vault) und **p049** (Regel in 4.11, Zauber in 5.5.6) — Aufzählungen über
  zwei Stellen, Einzelfälle.
- **Rechenfragen** wie „Dolch, STÄ 4, Wurfzahl 12“ → 3.14.
- **z001, z003, z006, z009, z012** (Simulationsberichte) → Überschriften beim Nutzer (später).
- **k001** → Satz im Regelbuch beim Nutzer. **x008** → Fundort seit 18.10. als Loot-Zeile in
  `Gegner/Kultpriester.md`, Retrieval holt jetzt Kultpriester + Regelbuch 7.2.4; Antwort seither richtig
  (E2E und 3/3, Punkt 12).

**Verworfene Ansätze für Bilder** (Roadmap 2/9): Vision-Beschreibungen beim Ingest, CLIP-Embeddings,
manuelle Captions, Dateiname als Rückfall; ein Endpunkt `/files/{path}` ist unnötig, weil `st.image` die
Datei auf dem Streamlit-Server liest (auch im Heimnetz).

## Eingang (neue Bugs und Ideen, noch nicht bewertet)
- **Vault, Maskenball (18.10.)**: die Überschrift „## Gegner: Eisscherben-Wächter (Traumkonstrukt)“ steht als H2 mitten in Szene 4; alle folgenden Szenen (4.3, 5, 6, 7) hängen dadurch im Überschriften-Pfad unter dem Gegner. Die Suche ordnet sie so dem Gegner zu (x008 holte vor der Kultpriester-Zeile Maskenball-Szenen als „Gegner“). Wäre als „####“ richtig eingeordnet — Inhalt, Entscheidung beim Nutzer.

---

## Erledigt

| Nr | Was | Wann | Wo dokumentiert |
|---|---|---|---|
| 13.6 | UI-Review (Subagent, Fokus „full“, Scope `ui/` + aufgerufene Endpunkte) und Durchgang durch alle Seiten/Tabs (AppTest gegen das laufende Backend, keine Exceptions); Korrekturen A–F nach Freigabe des Nutzers. **A — Eval-Jobs** (Meldung des Nutzers): E2E zeigte den ganzen Lauf „0 / 274“ (`evaluate()` ohne Fortschritts-Callback; der alte Test prüfte nur den Endstand) → Callback je Frage; der Lauf hing an einer blockierenden Schleife im Seitenskript und war nach Seitenwechsel weg (409 ohne Wiederanhängen, Tabs dahinter nie gezeichnet) → Job in `session_state` + Fragment wie bei der Indizierung, `GET /eval/active` (409 und neue Browser-Session hängen sich an), Ergebnis bleibt im Tab, gespeicherte Läufe per 🔍 wieder öffnen, E2E-Fehler je Frage sichtbar. **B** — Ergebnisse/Vergleich mit Answer Contains und gekippten Antworten (261 gegen 260 hieß vorher „Keine Unterschiede“), zwei alte `eval_202604*.json` gelöscht (Nutzer). **C — Eingaben gingen still verloren**: Prompts nach „Aktivieren“ (Textfelder mit `key` behielten den alten Text, Speichern setzte die Variante zurück), Settings/Evaluation-Regler sprangen bei geändertem „Kandidaten“ zurück (feste Maxima + Prüfung beim Speichern/Start), 👎-Kommentar nach Seitenwechsel überschrieben, Sidebar-Filter nach Seitenwechsel auf „alles“. **D** — Hybrid-Checkbox startet beim Default und sendet immer `bool` (`hybrid_default` in `/sidebar-state`); Stream ohne `done` und unterbrochene Frage bekommen „Erneut senden“. **E** — Texte: E2E-Streuung in der Anleitung, Indizier-Richtwerte (Vault ~8 min), Recategorize-Meldung, E2E-Expander, 404 vs. andere Fehler beim Abfragen; `ui-ux.md` (Fallback 20/8/3, Spalten, Session-State). **F** — Settings-Werte außerhalb des Reglerbereichs begrenzt mit Warnung statt Absturz; E2E-Wartezeit je Frage 2 × LLM-Timeout + 60 s statt 120 s; Wipe leert alle UI-Caches. Suite 900 → 921 grün; jeder neue Test gegen den alten Code rot (mutationsgeprüft). Nicht geändert: `category_map`-Kurztext bricht bei Ordnernamen mit `,`/`→` (Gering, kein Fall im Vault). **Live 18.10.** nach `stop.ps1 -Restart`: Retrieval-Lauf über die echte Seite (AppTest gegen das Backend) — neue Session hängt sich beim ersten Aufruf an, Fortschritt 89 → 272/274 auch über Seitenwechsel, Ergebnis 272 · 144 unverändert; 🔍 und E2E-Vergleich (17.10. gegen 18.10.: 94,9 → 95,3 %, 7 gekippte Antworten — wie von Hand gezählt); E2E-Fortschritt zählt (erste ~40 s 0 — Modell lädt nach dem Neustart). Dabei Beschriftung übernommener Jobs nach Typ statt „Laufende Evaluation“. Siebter E2E-Lauf `e2e_20261018_173927.json`: 261/274 wie der sechste (p027, a013 gewonnen, p043, p091 verloren — Streuung) | 18.10. | `docs/ui-ux.md`, `docs/evaluation.md`, `ARCHITECTURE.md` |
| 11 | Suchgenauigkeit nach den Re-Indizes vom 13.–18.10. (In-place, Regelbuch 3× neu, Vault-Änderungen 13:16): 18.10. nach Punkt 12 erneut **274/274 exakt** (3.830 Chunks, `ef_search` 2000, 15 s), auch m039 wieder — kein Komplett-Neuaufbau nötig. Kein eigener Punkt mehr: gehört zur Regressionsprüfung (oben) | 18.10. | `docs/evaluation.md` |
| 12 | Regressionsprüfung nach den Inhaltsänderungen vom 18.10.: Die Vault-Änderungen (13:05–13:08) kamen mit der UI-Indizierung des Nutzers um 13:16 in den Index, samt neuer Simulations-Dokumente (Begegnungsbaukasten). `Heimkehrpfennig.md` wieder gelöscht (Nutzer-Entscheid: die Notiz war früher bewusst entfernt, Regeltext nur im Regelbuch 7.2.4) — stattdessen Loot-Zeile in `Gegner/Kultpriester.md`; x008 holt jetzt Kultpriester (Platz 1) + Regelbuch 7.2.4 (Platz 2), im Golden Set als zulässige Quelle ergänzt. **Golden Set**: Datei 272, Abschnitt 144 unverändert, Platz 1 136 → 134 (m005, x008 — Kultpriester vorn, Regelbuch weiter dabei), 37 Fragen mit anderen Chunks (meist Einzeltausch durch die Simulations-Dokumente); Code-Review 13.5 neutral. Suche 274/274 exakt (3.830 Chunks). **E2E** (`e2e_20261018_152757.json`, vom Nutzer gestartet) gegen `e2e_20261017_213213.json`: **261/274** statt 260, Datei 273, Ø 7,2 s (p50 6,3, p95 13,8 s). Gewonnen x008, p055, p096, p107; verloren p035 (Beispielliste weggelassen), m056 (nur einer der zwei Meister), z004 („Trefferquote“ statt „Verlust“) — Quellen gleich, LLM-Streuung. **Holdout** 9/10 · 51/53 unverändert. x008 und „Wie viele Magiepfade hat der Schattenorden?“ je **3/3** („vier Magiepfade“, Kultpriester). **Maskenball-Bilder**: Bilder hängen an den Chunks, erscheinen aber nur mit vollem Notiznamen in der Frage → „Offen beim Nutzer“ (Alias). k001 folgt mit dem Regelbuch-Satz | 18.10. | `docs/evaluation.md` |
| 13.5 | Code-Review seit 80bde6b (25 Commits, ~500 Zeilen in 14 Dateien; Schwerpunkt Fehler, Spieler-Filter, Prozesse, Tests) + Korrekturen: **(1, Mittel)** „Erneut senden“ stürzte nach zwei Fehlern in Folge ab (doppelter Widget-Schlüssel; beim Provider-Fehler landete der Absturz zusätzlich als dritte Fehlermeldung im Verlauf) — eine Bedingung im Verlauf, 2 AppTests, mutationsgeprüft. **(2)** Verschachtelte Sektionen: die äußere wurde geloggt und aufs Budget angerechnet, kam aber nie an — jetzt gewinnt die äußerste, Unit-Test, mutationsgeprüft; Golden Set im selben Modus unverändert 272 · 144 (nur 4 Fragen andere Reihenfolge). **(3)** Verdacht „Tippfehler-Korrektur zeigt Spielern Spielleiter-Wörter“ **widerlegt** — die korrigierte Frage bleibt im Retriever; neuer Spieler-Sicherheitstest („invictux“ → intern „Invictus“, `search_query` und Prompt unverändert), mutationsgeprüft. Rest-Orakel (wer ein Wort errät, kann es bestätigen) bewusst hingenommen (Nutzer-Entscheid B), dokumentiert. **(5)** PDF: „5.2. Effekte:“ wurde als Beschriftung behandelt — Regex angeglichen, Test. **(4)** `**5.4.5** Titel` (nur Nummer fett) nicht geändert: 0 Fälle im Regelbuch. **(6)** einmal `WinError 10055` (Socket-Puffer) in der Suite, danach nicht wieder — beobachten, bei Wiederholung `-W error::ResourceWarning`. Vom Reviewer bestätigt: `stop.ps1 -Restart` (Strg+C-Wiedereinschalten experimentell), GPU-Zähler auf deutschem Windows, Tippfehler-Index, `aliases`/Bilder. Suite 900 grün | 18.10. | `docs/embedding-strategy.md`, Code |
| 9 | Bilder — Rest **geschlossen**: Bilder aus PDFs entfallen (das Regelbuch enthält keine, 0 auf 115 Seiten). Dateiname als Rückfall nicht gebaut: von 20 nicht eingebetteten Bildern hat keines eine gleichnamige Notiz, 15 sind Räume des Weihnachts-One-Shots — Einbetten unter der Raum-Überschrift wirkt genauer und ohne Code (→ „Offen beim Nutzer“); damit entfällt auch „neues Bild ohne Notiz-Änderung“ | 18.10. | — |
| 10 | Übrige Einzelfälle 3.10/3.15 — **geschlossen**: AP-Pool 3/3 richtig (Antwort erklärt den gemeinsamen AP/MP-Pool), p009 und „Wer ist Arkenfeld?“ waren schon richtig. k001 (Langschwert-Preis, zwei Schritte Kategorie → Preistabelle) und x008 (Heimkehrpfennig nur in einer Loot-Tabellenspalte; LLM erfindet den Gegner) brauchen Inhalt statt Code → „Offen beim Nutzer“. Tabellenzeilen-Suche für Obsidian-Tabellen war 14.10. schon verworfen | 18.10. | — |
| 9 (Teil) | Bilder auch bei Vor-/Spitznamen: `sources` liefern die Frontmatter-`aliases` der Notiz, der Chat zeigt ihre Bilder, wenn Frage oder Antwort den Namen **oder** einen Alias nennt („Wer ist Lilith?“ → Lyssara Vaydris). 7 der 11 Notizen mit Bildern haben Aliase (Vornamen, Spitznamen). Fremde Porträts bleiben draußen (UI-Test mit Liora, mutationsgeprüft); live: Aliase kommen mit den Quellen an. Kein Re-Index nötig | 18.10. | `docs/ui-ux.md`, `ARCHITECTURE.md` |
| 3.19 | Simulationsbericht in Regelfragen — **ohne Änderung erledigt**: 42 von 129 Regelfragen bekommen Chunks aus `llm_report.md`, zusammen 93 von 1.576 (5,9 %), fast immer 1–3 aus der normalen Rangfolge; meist „Kontext > Metriken-Erklärung“ (21 Fragen), die die Regeln für die LLM-Analyse nacherzählt. Section Expansion nur bei p011 (12 Chunks) — Antwort trotzdem 3/3 richtig. Den Abschnitt auszuschließen kostete z002 (Simulationsfrage, braucht ihn); gezieltes Tuning für eine richtige Antwort lohnt nicht. Neu prüfen, wenn Regelantworten Werte aus dem Bericht statt aus dem Regelbuch nennen | 18.10. | — |
| 3.18 | Kopfzeilen im Regelbuch-Text entfernen **verworfen**: Kopfzeile (Kapitel + Seitenzahl) steht an jedem Seitenumbruch, in 104 von 458 Chunks. Entfernt → Abschnitt 144 → 143, p024 (Krieger-Fähigkeiten) 0/3 — der Kapitelname hatte das Regelbuch vor NSC-Notizen gehalten; Variante „Kapitel einmal vorn im Embed-Text“ → 142, Platz 1 136 → 134, Alchemist/Dunkle Magie/Zeitmagie verlieren die Section Expansion. Zurückgebaut, Regelbuch neu indiziert: wieder genau 272 · 144 · 136, Suche 274/274 exakt | 18.10. | `docs/parsing.md` |
| 3.17 | PDF-Parser: Tabellenkopf `**1W8 Ergebnis**` galt als Kapitel 1 — die Regex für nummerierte Überschriften verlangte kein Leerzeichen nach der Nummer; Patzer-Tabelle und „Wurfgemisch“ hingen dadurch nicht unter 5.4.6 Alchemist. Fix + Unit-Test (schlägt mit der alten Regex fehl), Regelbuch neu indiziert (80 s): Golden Set unverändert 272 · 144, Rang 1 136, 8 Fragen mit anderen Chunks, keine kippt; p091 und p092 je 3/3 inhaltlich richtig (p092 morgens 0/3); Suche 274/274 exakt. `Talente`/`Eigenheiten` als oberste Überschrift bleiben (Kapitel-Einleitungen ohne Nummer im PDF-Text, richtig benannt) | 18.10. | `docs/parsing.md` |
| 8 | Monolith-Sektion in fremden Regelfragen — **ohne Änderung erledigt**: Monolithon (7.3, 21 Chunks) ist sauber gegliedert, kein Parser-Fehler. p011 hat es nur noch auf Platz 14 von 17; p101 (Zauberspeicher) auf 4–6 — dort passend, die Würfel sind „beliebte Träger für Zauberspeicher“. Das eigentliche Rauschen in p011 ist der Simulationsbericht (3.19, ebenfalls ohne Änderung erledigt) | 18.10. | — |
| 16 | Neustart-Knopf (Idee des Nutzers): **🔄 LoreKeeper neu starten** unter „⏻ beenden“ (gleiche Rückfrage, gesperrt während einer Indizierung) startet `stop.ps1 -Restart` — beendet sauber und startet `start.ps1` im Modus, der lief; räumt den VRAM (Ollama wird mit beendet), die 7b-Warnung verweist darauf. Stolperstein: die eigene Prozessgruppe des Buttons schaltet Strg+C ab, Kinder erben das — das Skript schaltet es vor dem Start wieder ein (mutationsgeprüft live: ohne wurde das neue Backend beim nächsten Stopp nach 20 s hart beendet). Live: Server- und Entwicklungsmodus je ~25 s, nächster Stopp sauber ~2 s; UI-Tests mutationsgeprüft | 18.10. | `docs/ui-ux.md`, `docs/operations.md` |
| 15 | Desktop-Verknüpfungen (Idee des Nutzers): `scripts/create-shortcuts.ps1` legt **LoreKeeper starten** (`start.ps1 -Server -Browser`) und **LoreKeeper beenden** (`stop.ps1`) auf den Desktop; `start.ps1 -Browser` öffnet die UI auch im Server-Modus und bei schon laufendem LoreKeeper. Fenster bleibt 15/8 s für LAN-URLs und Warnungen offen. Per Doppelklick geprüft: Start nach 14 s, zweiter Klick startet nichts doppelt, Beenden sauber nach 2 s | 18.10. | `docs/operations.md`, `README.md` |
| 7 | Deckel je PDF-Kapitel **verworfen**: Golden Set 272 · 144 unverändert, E2E 260/274 wie die Grundlinie, aber p092 (Tränke je Rang) reproduzierbar verloren — Rang-Tabelle ist der 4. Chunk ihres Kapitels; Deckel bleibt je Datei. Nebenbei: Antwortzeit Ø 7,3 s auf freier GPU | 18.10. | `docs/embedding-strategy.md` |
| 7b | VRAM-Warnung: der Health-Loop misst alle 5 min, wie viel des lokalen LLM im geteilten GPU-Speicher liegt (Windows-Leistungsindikator `Shared Usage` der Ollama-Prozesse, ein PowerShell-Aufruf, keine neue Abhängigkeit, `src/gpu_memory.py`); über 1 GB WARNING im Log (nur beim Wechsel, mutationsgeprüft), `llm_shared_gb` in `/health` und eine Warnung in der Sidebar. Anlass 17./18.10.: 2,65 GB im geteilten Speicher, 16,9 statt ~54 tok/s bei „100% GPU“. Unit-, Route-, Log- und UI-Tests. Live 18.10.: 0,0 GB bei freiem VRAM, Tempo normal; der Warnfall selbst nicht nachgestellt | 18.10. | `docs/operations.md`, `docs/ui-ux.md`, `ARCHITECTURE.md` |
| Nutzer | Inhalt und Prüfungen 18.10.: **Vault** — `Schattenorden.md` auf „vier Magiepfade“ (Nutzer-Entscheid; passt zu `Meister der Pfade.md`), Fundort des Heimkehrpfennigs (vorher erfand das LLM einen Gegner) — erst als neue Notiz `Gegenstände/Heimkehrpfennig.md`, am selben Tag wieder gelöscht (die Notiz war früher bewusst entfernt worden, Regeltext steht nur im Regelbuch 7.2.4) und durch eine Loot-Zeile in `Gegner/Kultpriester.md` ersetzt, 19 von 20 Bildern eingebettet (15 Maskenball-Räume unter ihren Szenen, Lagerhaus, Artefaktraum, Lysander-Anwesen, Morgenstern-Karte; Zuordnung vom Nutzer bestätigt, kein Bild ohne Einbettung, keine Einbettung ins Leere). **System**: alte Autostart-Aufgabe und Firewall-Regeln geprüft. **Handprüfungen im Browser**: Beenden/Neustarten, Desktop-Verknüpfung, Indizierungs-Fortschritt, Tabs Feedback/Chunks, Bilder per Alias. Indiziert und geprüft in 12. **Maskenball-Alias** (aus 12, mit Freigabe des Nutzers): `aliases: [Maskenball, Maskenball des Winterkönigs]` im Frontmatter von `Der Maskenball des Winterkönigs.md`, inkrementell indiziert (nur diese Notiz, 45 Chunks); Golden Set 272 · 144 unverändert. Offline-Probe (Retrieval + Bildregel des Chats): „Ballsaal im Maskenball des Winterkönigs“ 7 Bilder (vorher 0), „Thronsaal beim Maskenball“ 9, „Küche beim Maskenball“ 5 | 18.10. | `docs/testing.md` |
| 13.4 | Zwei Chat-Bugs vom 15.10.: 👎-Kommentar wurde bei jedem Verlassen des Felds halb gespeichert (`st.text_input` speichert beim Fokusverlust) → Feld in `st.form`, nur Enter/Knopf speichert; fremdes Porträt unter der Antwort (Liora Veylan als schwache Quelle Nr. 6) → Bilder nur von Notizen, deren Name in Frage oder Antwort steht (Entscheidung des Nutzers). Beide UI-Tests gegen den alten Code rot. Live und Vor-/Spitzname erledigt 18.10. (9, Handprüfung des Nutzers) | 17.10. | `docs/ui-ux.md` |
| 3.15 (Teil) | Golden-Set-Stichwörter aus 13.3 (k015, a014, a005, m015, p026): Antworten richtig, ohne den Namen aus der Frage — je eine Tatsache, die alle drei Läufe nennen und die in der Quelle steht. Dritter Lauf neu ausgewertet 245 → 250/274 (0,912), nur diese fünf kippen; die Läufe vom 14.10. speicherten nur 200 Zeichen je Antwort und lassen sich nicht neu auswerten | 17.10. | `docs/evaluation.md` |
| 3.10 (Teil) | Rerank-Paar mit Notizname: Markdown-Chunks gehen mit Identitätszeile (Dateiname \| Aliase) in den Cross-Encoder, PDF-Paare unverändert (Regelbuch-Name ist Rauschen, verlor k006). Golden Set 272/274 · 143/149, Platz-1-Abschnitt 118 → 120; Holdout 9/10, 51/53; E2E 250 → **253/274** (8 gewonnen, darunter m048 „Deckname: Solmar“, 5 verloren — alle Formulierung/Rauschen, gesichtet); Regelbuch-Anteil an den Quellen steigt leicht, `min_score` −4 bleibt. Kein Re-Index nötig. a005-Stichwort „Reaktor“ → „Carath“ (zu eng gewählt am selben Tag) | 17.10. | `docs/embedding-strategy.md`, `docs/evaluation.md` |
| 3.10 (Teil) | Schattenorden-Kontext: `Schattenorden.md` hatte 27 Chunks (> `section_expansion.max_chunks` 25), m003/m055 bekamen die Datei nicht ganz. Erledigt ohne Code durch 13.2 (leere Chunks entfernt): jetzt 18 Chunks, beide Fragen bekommen alle 18 (Vorschau 17.10.), beide im E2E-Lauf vom 17.10. richtig | 17.10. | — |
| 14.1 | „🔁 Erneut senden“ unter Fehlern (Idee des Nutzers): Provider-, Backend- und Verbindungsfehler bleiben als „⚠️ …“ im Verlauf (der Verbindungsfehler verschwand vorher beim nächsten Rerun), die letzte trägt den Knopf; ein Klick ersetzt Frage + Fehler und schickt die Frage erneut. Kein Knopf unter Antworten oder älteren Fehlern. 4 UI-Tests, alle gegen den alten Code rot | 17.10. | `docs/ui-ux.md` |
| 3.15 (Teil) | Aufzählungen über große Kapitel: (1) PDF-Beschriftungen („Nachteil:“, „Legendär (Rang 6):“) sind keine Überschriften mehr — 109 von 543 Regelbuch-Chunks betroffen, „Manaschwäche“ wieder auffindbar, Regelbuch 543 → 458 Chunks; (2) der Einleitungs-Chunk zählt in der Section Expansion als Treffer seiner Sektion. Golden Set 272/274 · 144/149 (+z006), Holdout 9/10 · 51/53, E2E 253 → **260/274**; p036, p037, p114, p014 je 0 → 3/3. a030, p049 → „Bekannte Restfehler“; Antwortzeit 18.10. gemessen (Ø 7,3 s, kein Aufpreis) | 17.10. | `docs/parsing.md`, `docs/embedding-strategy.md`, `docs/evaluation.md` |
| 3.15 (Teil) | Tippfehler-Toleranz: ein Wort, das in den Quellen nirgends vorkommt, wird vor der Suche durch das großgeschriebene Quellwort im Abstand 1 (ab 10 Buchstaben 2) ersetzt — gleicher Anfangsbuchstabe, keine Beugungen, nur bei klarem Kandidaten; nur intern wie das Truecasing. „was ist üebrschaden?“ findet jetzt dieselben Abschnitte wie richtig geschrieben (vorher Krieger). Golden Set mit einem Tippfehler je Frage (270): 7 → 0 abweichende Treffer; sauber unverändert 272 · 144, nur 4 gleichbedeutende Umschreibungen. Zwei lockerere Fassungen verworfen (27 bzw. 11 veränderte saubere Fragen, verloren p095/z001 bzw. p065). Grenze: Tippfehler im ersten Buchstaben. 1,4 s Aufbau, 0,13 ms pro Frage | 17.10. | `docs/embedding-strategy.md`, `ARCHITECTURE.md` |
| 3.10 (Teil) | Tabellen im Rerank auf die genannten Zeilen fokussiert: Tabellen-Chunks, deren Zeile die Frage nennt, bekommen einen zweiten Score für Kopfzeile + genannte Zeilen, der bessere zählt (der fokussierte allein verlor p086 und den Glücksstein). Golden Set unverändert 272 · 144, keine Frage kippt; „Unterschied Dolch und Kurzschwert“ Waffentabelle — → Platz 2, „Hammer“ 2 → 1, k001 bekommt die Waffentabelle. Ø +15 ms. Gefunden: Deckel pro Datei begrenzt das Regelbuch auf 3 Chunks (→ 7, am 18.10. verworfen); Langbogen nennt 18.10. die Waffentabelle | 17.10. | `docs/embedding-strategy.md` |
| 13.1 | Ultrareview 14.10. (Änderungen seit 1ba1f49): 10 Befunde zu Bildern und `stop.ps1`, 8 korrigiert (Einbettung mit Pfad vor Name, Ersatz-Chunk mit Bildern, Titelbilder nur mit echter H1, Frontmatter-Schlüssel `images`/`image_embeds` reserviert, Einbettungen aus Überschriften, Bild-Suche im Umfang der Indizierung — **Spieler-Invariante**, Sicherheitstest mit Köder-Bild, mutationsgeprüft —, `stop.ps1 -List`, ein Helfer für BM25 + Truecaser); #7 und #9 mit Grund verworfen (kein Fall für einen Handler, kein Start ohne Konsole). Voller Re-Index, Golden Set unverändert 272/274 · 143/149; live 15.10.: `stop.ps1 -List` und Bilder im Chat | 14./15.10. | `docs/parsing.md`, `docs/testing.md`, `docs/operations.md` |
| 13.3 | Regressionsprüfung nach 13.1/13.2: E2E 245/274 (0,894, vorher 248/247 — Rauschen), Datei 0,993, Ø 7,1 s; 6 der 19 Dauer-Fehler jetzt richtig, 7 neue gesichtet (5 nur Stichwort, i002 frühere Schein-Treffer-Absage, p035 schwankt) — keine Regression. Holdout 9/10, 50/53; live keine leeren Chunks mehr. Stichwörter erledigt 17.10. (3.15 Teil) | 15.10. | `docs/evaluation.md` |
| 3.10 (Teil) | Reranker `jina-reranker-v2-base-multilingual` (fp16 auf der GPU, fp32-ONNX als CPU-Rückfall statt torch): Abschnitt 139 → 142/149, Platz-1-Treffer 110 → 121, Datei unverändert 272/274; bge-m3, MiniLM en-de, gte verworfen; VRAM neben qwen3:14b 14,4 statt 14,6 GB, LLM-Tempo gleich | 14.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 3.10 (Teil) | Eintrags-Slot: nennt die Frage eine Tabellenzeile („Dolch-Angriff“, „Hammer“, „Glücksstein“), kommt die Tabelle per Namenssuche in den Rerank-Pool (Filter der Anfrage, max. 3, kein reservierter Platz). Golden Set Abschnitt 142 → 143/149 (+x004), nichts verloren; Tabellenfragen 5 → 8/11; ~75 ms. Verworfen vorher: `top_k` 25–40 (verliert m062/k006/z001), Zeilen-Chunks (142 → 141, Listenfragen 9/9 → 2/9). Spy-Test + Köder-Tabelle, mutationsgeprüft | 14.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 13.0 | Code-Review seit 194be1b (jina, Regelkern, Batches, Eintrags-Slot) + Korrekturen: **kritisch** — eine lange Frage machte den Eintrags-Slot zu einer `$contains`-Abfrage ohne Grenze (4.000 Zeichen → 500, 12.000 → Backend-Prozess stirbt ohne Log, im Heimnetz für jeden Spieler auslösbar); jetzt höchstens 60 Namen (Golden Set braucht ≤ 48, nichts ändert sich) und `question` ≤ 4.000 Zeichen an `/query`, `/query/stream`, `/eval/preview` (422) sowie im Chat-Feld — beides mutationsgeprüft, 50.000 Zeichen laufen durch. Dazu: Kostenmodell Namen × Chunks dokumentiert, Schwellen-Umgehung des Slots dokumentiert und per Test festgehalten, Docstrings/Kommentare nachgezogen, Rerank-Zeiten mit Kontext, Regelkern nur mit spielerfreigegebenem Inhalt. Aus dem Performance-Review: `start.ps1` hielt `TimeWait`-Reste für einen laufenden Server und startete das Backend nicht (jetzt `-State Listen`). Performance 14.10.: Retrieval p50 278 ms / p95 364 ms (Baseline 784/1.156), erster Token ~2 s (Regelkern +0,56 s), Antwort p50 7,6 s je nach Länge | 14.10. | `docs/embedding-strategy.md`, `docs/prompts.md`, `docs/provider-strategy.md`, `ARCHITECTURE.md` |
| 3.15 (Teil) | End-to-End-Basis: zwei Läufe (Answer Contains 0,905 / 0,901, ~15 Fragen kippen zwischen Läufen), `evaluate.py` speichert ganze Antwort, `id` und Stichwörter (Evaluation-Seite zeigt sie), Holdout `evaluation/holdout_lists.yaml` (10 Listenfragen, ungetunt 9/10), acht Golden-Set-Stichwörter angepasst (k001, p095 an die Regelbuch-Änderung, sechs zu enge), Hybrid mit jina neu gemessen (bleibt aus), Eintrags-Slot für Obsidian-Tabellen gemessen und verworfen; Feintuning danach beendet | 14.10. | `docs/evaluation.md`, `docs/embedding-strategy.md`, `docs/ui-ux.md` |
| 3.16 | Chunks unter der Quelle einsehbar (Idee des Nutzers): `sources` liefern zusätzlich `content` (ganzer Chunk-Text), der Chat zeigt je Chunk einen Ausklapper „↳ Überschrift · Score“ mit dem Text (Überschriftenzeile gekürzt, `<br>` → Umbruch, ältere Nachrichten: Vorschau). Spieler-Sicherheitstest prüft auch `content` auf Geheimnisse; UI-, API- und Langtext-Test. Keine zusätzliche Backend-Anfrage | 14.10. | `docs/ui-ux.md`, `ARCHITECTURE.md` |
| 3.15 (Teil) | 👎 aus dem Chat ausgewertet (5 Bewertungen, 4 Fälle, alle live nachgestellt): **Folgefragen** wie „bitte mehr infos“ wurden nicht umgeschrieben — „Info/Infos/Informationen/erzähl/hast“ ergänzt, Tests in beide Richtungen, keine Golden-Frage neu als Folgefrage; **Kleinschreibung** kostete systematisch (Golden Set klein 143 → 135/149) — klein getippte Wörter vor der Suche in die Schreibweise der Quellen (`src/retrieval/truecase.py`, nur intern): klein 135 → 142, wie geschrieben unverändert 143, Holdout klein 6 → 7/8; **Klassenübersicht** vom Nutzer im Regelbuch ergänzt (5.4/5.5), neu indiziert: „welche klassen gibt es?“ nennt alle 12. Tippfehler erledigt 17.10. (3.15 Teil) | 14.10. | `docs/embedding-strategy.md`, `docs/prompts.md`, `ARCHITECTURE.md` |
| 12.4 | `stop.ps1` beendete den Browser: Windows vergibt PIDs neu, der Browser, den `start.ps1` geöffnet hatte, galt als Kind des UI-Prozesses und wurde nach 20 s hart beendet. Kinder zählen nur noch, wenn sie nach dem Elternprozess gestartet wurden; neu `.\stop.ps1 -List` zeigt die Prozesse, ohne zu stoppen | 14.10. | `docs/operations.md` |
| 12.3 | Auto-Reload hing (dreimal am 14.10.), nachgestellt mit drei Ursachen: **(1)** Uvicorn startet den Worker unter Windows per Strg+C an die ganze Konsole neu — das traf auch das vom Backend gestartete Ollama (laufende Antwort → 503). Jetzt eigene Prozessgruppe, Beenden per Strg+Pause samt `llama-server` (1 s, Modell nicht mehr im VRAM). **(2)** Ein laufender End-to-End-Lauf (`asyncio.to_thread`) hielt den alten Worker am Leben — Python wartet beim Beenden auf Pool-Threads, der Lauf fragt den eigenen, nicht mehr antwortenden Server mit 120 s je Frage, und der Reloader wartet ohne Grenze: kein neuer Worker. Jetzt Daemon-Thread (`run_in_daemon_thread`, mutationsgeprüft); Indizierung bleibt bewusst ein Pool-Thread (wird fertig geschrieben). **(3)** `stop.ps1` während eines Neustarts: der Reloader startete noch einen Worker, der kein Strg+C bekam und als Waise Port 8000 hielt — `stop.ps1` sucht vor dem harten Beenden erneut nach Kindprozessen. Live geprüft: Neustart während E2E-Lauf mit geladenem Modell → Antwort fertig, Ollama + Runner beendet, neuer Worker nach 13 s; Stopp während Neustart-Serie → Port frei, keine Waisen. Der 17:15-Fall war keiner: Änderung in `evaluation/`, das nicht überwacht wird | 14.10. | `docs/operations.md` |
| 2 | Bilder im Chat ohne Bildbeschreibung (Ansatz des Nutzers): Der Markdown-Parser behält `![[bild.png]]` / `![[bild.png\|300]]` je Abschnitt; **Titelbilder** (vor der ersten Unterüberschrift) hängen an jedem Abschnitt der Notiz — nur je Abschnitt bekam live keine Frage zu Malek/Arkenfeld ein Bild. Der Orchestrator löst die Namen **innerhalb derselben Quelle** zu Pfaden auf (`images`), `sources` liefern sie, der Chat zeigt „🖼️ Bilder (n)“ (zugeklappt, max. 6, nur existierende Dateien). Spieler-Sicherheitstest mit Köder-Bild in einer Spielleiter-Quelle, mutationsgeprüft (quellübergreifende Auflösung → 7 Fehlschläge). Zwei volle Re-Indizes des Vaults (je ~8 min); Golden Set unverändert Datei 272/274, Abschnitt 143/149. Live: Malek → Porträt, Arkenfeld-Karte → `arkenfeld.png`, Isolde Vaer → 2 Bilder. Suche danach 273/274 exakt (m039 weicht ab, Treffer unverändert) — nach weiteren In-place-Re-Indizes neu prüfen. Rest am 18.10. geschlossen (9) | 14.10. | `docs/parsing.md`, `docs/ui-ux.md`, `ARCHITECTURE.md`, `docs/embedding-strategy.md` |
| 3.4 | Kommentar zur Bewertung: nach 👍/👎 ein optionales Feld (Enter speichert, ≤ 1.000 Zeichen, API antwortet sonst 422); jede Zeile in `logs/feedback.jsonl` trägt Bewertung und Kommentar zusammen, Zurücknehmen löscht ihn. Tab „Feedback“ mit Spalte „Kommentar“, beim Übernehmen ins Golden Set landet er in der Notiz. UI-, API- und Grenztests; kein zusätzlicher Backend-Aufruf beim Anzeigen | 14.10. | `docs/ui-ux.md`, `docs/operations.md`, `ARCHITECTURE.md` |
| 13.2 | Leere Chunks: 251 von 4414 waren nur `---` (dazu Block-IDs, Sim-Kommentare, Tag-Zeilen, leere Notizen, abgeschnittene Beschriftungen) — Parser entfernt sie, Chunker hängt Stücke unter `min_chunk_size` an das nächste und verwirft Stücke ohne Buchstaben/Ziffern. Index 4414 → 3933 Chunks, 0 ohne Inhalt, Golden Set unverändert, doppelt belegte Plätze 738 → 556 | 14.10. | `docs/parsing.md`, `docs/embedding-strategy.md`, `ARCHITECTURE.md`, `docs/configuration.md`, `docs/data-flow.md` |
| 3.13 | Regelkern im `qa`-Prompt (15 Stichpunkte zum Regelwerk, direkt vor der Frage; „Ressourcen vs. Punkte“ ergänzt: p009 Zauberpunkte 0/3 → 3/3): mit mMiniLMv2 Rang-1-Wurfzahl und Rüstung je 0/3 → 3/3, Lore-Fragen unverändert; im Systemprompt wirkungslos. Live mit jina: Rüstung 3/3, Rang-1-Wurfzahl wieder 0/3 (Lead „höchstens 13“). Rang 1 mit jina, Dolch, Schattenklinge → 3.14 (pausiert) | 14.10. | `docs/prompts.md` |
| 12.2 | Reranker rechnet in Batches zu 32 Paaren: 200 Kandidaten füllten neben qwen3:14b die GPU (Rerank 3–8 s, einmal DirectML-Fehler → bis zum Neustart CPU); jetzt ~0,8 s bei +0,5 GB, `top_k` 20 unverändert ein Batch (~90 ms) | 14.10. | `docs/embedding-strategy.md` |
| 3.10 (Teil) | Regelwerk-Slot (eigene Suche in `rules`, Mindest-Score −4) | 13.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 3.10 (Teil) | `top_k` 20 (+2 Abschnitte, Hybrid verworfen), `ingestion.skip_sections` gegen „Alte Vault-Werte“ (Full-Reindex, 4419 Chunks), `ef_search` 1000 → 2000 (nach den Re-Index-Läufen wieder exakt), verwaiste Chunks entfernt, Golden Set 274 Fragen — Datei 272/274, Abschnitt 137/149 | 13.10. | `docs/embedding-strategy.md`, `docs/parsing.md`, `docs/configuration.md`, `docs/evaluation.md` |
| 3.10 (Teil) | Tabellen-Chunks: Zeilennamen im Embed-Text („Einträge: Dolch, …“) — Langschwert/Armbrust gelöst, Abschnitt 137 → 139/149, nichts verloren (Vektor-Rang 22 → 14, 47 → 2); Fließtext-Variante verworfen; Full-Reindex, Suche exakt | 13.10. | `docs/embedding-strategy.md` |
| 3.12 | Folgefragen ohne Inhaltswort („Fass das kürzer zusammen“, „Gib mir ein Beispiel“) gehen durch Condense — Suche und Quellenliste passen zum Thema; Wortliste nur ergänzend (unbekanntes Wort = Inhaltswort), Unit-Tests in beide Richtungen, mutationsgeprüft. Live 13.10.: „Fass das kürzer zusammen“ → „Fasse die Beschreibung von Arkenfeld kurz zusammen.“ (1,2 s), Quellen `Orte/Arkenfeld.md` statt Venthir/Regelbuch | 13.10. | `docs/prompts.md` |
| 4.3 | Sauberes Beenden: `stop.ps1` (Strg+C an beide Fenster, nach 20 s hart, samt Ollama; Sperre während Ingest) und **⏻ LoreKeeper beenden** in der Sidebar (Admin). Gemessen: Strg+C ist in allen Lagen sauber (Backend 1,3–1,6 s, UI 0,2 s, auch bei Stream/Eval) — 7 von 10 Starts endeten durch Fenster-Schließen ohne Shutdown; die verzögerte Strg+C-Reaktion war nicht reproduzierbar (Verdacht: markierter Text im Konsolenfenster hält den Prozess an) | 13.10. | `docs/operations.md`, `docs/ui-ux.md` |
| 5.5 | Vektorsuche exakt und reproduzierbar: `hnsw_ef_search` 100 → 1000 (später 2000, siehe 3.10) (0 statt 11 abweichende Golden-Fragen zwischen Prozessen, +5 ms/Frage, p105 gewonnen); Stolperfalle l2 beim Anlegen behoben; Prüfung „🎯 Suchgenauigkeit“ in der UI und Anleitung auf der Evaluation-Seite | 13.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 5.6 | Gesamtbudget der Section Expansion `max_total_chunks: 30`: max. 49 → 30 Chunks pro Frage, Treffer unverändert (276/278, 137/153); trifft nur die 3 Fragen mit gestapelten Sektionen und streicht dort die fremde. 25 und kleiner schnitten Klassen-/Magie-Sektionen (Sweep 0–12 in einem Prozess) | 13.10. | `docs/embedding-strategy.md`, `docs/configuration.md` |
| 6.0 | Simulation im Index: Quellen-Gruppe `simulation` (12.10.); Neu-Indizierung nach SIM 52 und Begegnungsbericht 13.10. (81 Dateien, 2957 Chunks, Index 4733). Eval: Datei 276/278, Abschnitt 136/153 (z010 verloren), z001–z016 Datei 14/16, Abschnitt 9/16; 55 von 278 Fragen mit Simulations-Chunks. Spieler bekommen bei 47 von 124 reinen Regelfragen Ø 2,3 Chunks aus dem Fähigkeitsbericht (6,7 % aller Chunks, 3× in den Top 3) | 12./13.10. | `docs/configuration.md`, `docs/evaluation.md` |
| 7.2 | Code-Review: Filter-Vertrag (nur `$eq`/`$ne`/`$in`/`$and`, sonst 422), GPU-Rückfall bei gleichzeitigen Fragen | 13.10. | `docs/configuration.md`, `docs/testing.md` |
| 7.3 | Inline-Doku: Docstrings für alle Module, Klassen, Funktionen; Config-Felder kommentiert | 13.10. | Code |
| 7.4 | Clean-Code-Review, Pakete A–D (Fehler, toter Code, Vereinfachung, Clean Code) | 13.10. | `git log` |
| 8.0 | Gleichzeitige Fragen gemessen, `OLLAMA_NUM_PARALLEL` bleibt 1, Warte-Hinweis im Chat | 13.10. | `docs/provider-strategy.md`, `docs/ui-ux.md` |
| 9.0 | Testabdeckung nach Risiko 86 → 93 % | 13.10. | `docs/testing.md` |
| 10.0 | Logging: Fehler mit Traceback in der Datei, wichtige Ereignisse, weniger Rauschen, eine Datei pro Tag, `logs/ollama.log` | 13.10. | `docs/operations.md` |
| 11.0 | Architektur-Review + Umbau P1–P7 (Recategorize, Fallback-Schalter, Index-Konsistenz, Filter erzwungen, UI-Fehlerpfad, atomares Speichern, Router aufgeteilt) | 13.10. | `ARCHITECTURE.md` 8 |
| 12.0 | Code-Review seit a3b5d77 + Korrekturen: `stop.ps1` nimmt sich selbst aus dem Prozessbaum (Button wartete 20 s und beendete sich selbst) und lässt fremde Server auf 8000/8501 in Ruhe; `hnsw_ef_search` ≥ 1, `max_total_chunks` ≥ 0; Collection wird beim Start geöffnet (ef_search wirklich beim Start); p50/p95 der Evaluation-Seite wie in `pytest -m perf`; Exaktheitsprüfung zählt Gleichstände an der Top-K-Grenze nicht; Tests für verschachtelte Sektionen und Duplikate (je mutationsgeprüft) | 13.10. | `docs/operations.md`, `docs/evaluation.md`, `docs/testing.md` |
| 12.1 | Ingest-Lücke: Ein Lauf über alle Quellen löscht Chunks von Quellen, die nicht mehr in `sources.yaml` stehen (vorher blieben sie durchsuchbar, 13.10.: 36 Chunks); nie bei leerer Quellenliste, nicht bei Einzel-Reindex; beide Schutzbedingungen mutationsgeprüft | 13.10. | `ARCHITECTURE.md` 4, `docs/data-flow.md`, `docs/configuration.md` |
| 3.3 | Chunk-Inspector (Tab „Chunks“) | 12.10. | `docs/ui-ux.md` |
| 3.4 | 👍/👎 unter jeder Antwort, `logs/feedback.jsonl`, Übernahme ins Golden Set | 12.10. | `docs/ui-ux.md`, `docs/operations.md` |
| 3.6 | Letztes Frage/Antwort-Paar im Antwort-Prompt | 12.10. | `docs/prompts.md`, `docs/configuration.md` |
| 3.8/3.9 | Ingest-Fortschritt ohne Blockieren, Anhängen an laufende Jobs | 12.10. | `docs/ui-ux.md` |
| 3.11 | Spieler-/Spielleiter-Modus inkl. Sicherheitstests (deterministisch, live, Spy-Test) | 11./12.10. | `ARCHITECTURE.md` 4.8, `docs/testing.md` |
| 4.2 (Teil) | Spieler-Filter im Backend erzwungen (`role`) | 12.10. | `ARCHITECTURE.md` 4.8 |
| 5.2 | Performance-Tests: Anfrage-Budgets, `pytest -m perf`, Startup-Baseline 17,7 s | 11./12.10. | `docs/testing.md` |
| 5.3 | Antwortzeit: Qwen3-Denken aus, Kontext 16k, Keep-alive 1 h, `top_rank` 3 — erster Token 8,7 → 1,1 s | 11./12.10. | `docs/provider-strategy.md`, `docs/configuration.md` |
| 5.4 | Reranker auf der GPU (DirectML): 840 → 50 ms, unter Last 14 s → 70 ms | 12.10. | `docs/embedding-strategy.md` |
| 7.0/7.1 | Testlücken Gemini/Prompts/Eval auf 100 %, Eval-CLI in der Windows-Konsole | 12.10. | `docs/testing.md` |
| 3.7 | Admin-Passwort für die UI, Bruteforce-Bremse | 11.10. | `docs/ui-ux.md`, `ARCHITECTURE.md` 4.8 |
| 4.1 | Heimnetz-Betrieb (`start.ps1 -Server`, Firewall-Skript, kein Autostart) | 11.10. | `docs/operations.md` |
| 5.1 | Stage-Timings je Anfrage (Log, `done`-Event, Chat-Zeile) | 11.10. | `docs/operations.md` |
| 1 | PDF-Parsing: OCR (RapidOCR), Bildextraktion, Überschriften aus dem Inhaltsverzeichnis | 14.04. | `docs/parsing.md` |
| 3.1 | Hybrid-Suche (BM25 + Vektor, RRF), Schwelle vor der Fusion | 14.04. | `docs/embedding-strategy.md` |

**Live-Prüfungen (13.10., echtes LLM und echter Index)**: Ollama-Autostart findet die Modelle
(`OLLAMA_MODELS`); Folgefrage „fass das kürzer zusammen“ liefert eine kürzere Antwort zum selben
Thema (Verlauf wirkt; seit 3.12 passt auch die Quellenliste — live geprüft); Regelwerk-Slot im Chat bei x002 und x005 —
Regelbuch unter den Quellen, Antworten nennen die Regel; Rerank-Zeiten in der Chat-Zeile, warm
~70–250 ms auf der GPU; Chunk-Inspector und Feedback-Liste über die API; Warte-Hinweis bei zwei
gleichzeitigen Fragen; `pytest -m live` 283 grün (2:21 min).
