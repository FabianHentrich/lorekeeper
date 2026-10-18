# Tests

Wie die Testsuite aufgebaut ist, wie man sie ausführt und was die Sicherheits- und
Performance-Prüfungen absichern. Regeln für Code-Änderungen (Tests sind Pflicht, `ruff`/`mypy`
sauber, keine echten LLM-Aufrufe) gelten wie in den übrigen Docs beschrieben.

## Inhaltsverzeichnis
1. [Aufbau](#aufbau)
2. [Tests ausführen](#tests-ausführen)
3. [Sicherheitsprüfungen (Spieler-Modus, Admin-Login)](#sicherheitsprüfungen-spieler-modus-admin-login)
4. [Log-Pfade](#log-pfade)
5. [Performance-Prüfungen](#performance-prüfungen)
6. [Abdeckung](#abdeckung)

---

## Aufbau

| Ordner | Inhalt | Läuft mit |
|--------|--------|-----------|
| `tests/unit/` | Ein Modul pro Datei, externe Dienste gemockt | `pytest` |
| `tests/integration/` | API-Schicht über `TestClient`, RAG-Pipeline, End-to-End Ingest → Query, Spieler-Sicherheit | `pytest` |
| `tests/ui/` | Streamlit-Seiten mit `AppTest` gegen ein Fake-Backend (`tests/ui/conftest.py`) | `pytest` |
| `tests/test_code_quality.py` | `ruff check .` und `mypy` (Konfiguration in `pyproject.toml`) | `pytest` |
| `tests/perf/` | Zeitbasierte Prüfungen gegen `tests/perf/baseline.json` | `pytest -m perf` |
| `tests/live/` | Prüfungen gegen das laufende Backend und den echten Index | `pytest -m live` |

Grundregeln: Unit-Tests mocken externe Dienste (Ollama, Gemini, ChromaDB). Integrationstests nutzen
`chromadb.EphemeralClient` (im Speicher, kein Plattenzugriff). **Kein echter LLM- oder API-Aufruf in
der Testsuite.** Neue Logik in `src/` (außer dünner Verdrahtung in den Routen-Modulen `src/api/*_routes.py`) bekommt einen
Unit-Test; Routenverhalten gehört in `tests/integration/`.

`tests/integration/test_e2e_ingest_query.py` ist der End-to-End-Test: Er indiziert einen kleinen
Vault über HTTP und fragt ihn über `/query/stream` ab — echtes Parsen, echte Embeddings (kleines
MiniLM-Modell), echte ChromaDB, Fake-LLM. Er lädt ein Modell und dauert daher länger als der Rest.

`tests/integration/conftest.py` stellt das echte Embedding-Modell als session-weite Fixture
`embed_service` bereit (einmal pro Testlauf geladen, von RAG-Pipeline, End-to-End- und
Spieler-Sicherheitstest geteilt) und `parse_sse()` für die Events aus `/query/stream`.

---

## Tests ausführen

```powershell
pytest                                           # komplette Suite (~1 min)
pytest --cov=src --cov-report=term-missing       # mit Coverage
pytest tests/unit/retrieval/test_embeddings.py   # eine Datei
pytest tests/unit                                # nur Unit-Tests
ruff check .; mypy                               # Lint + Typen einzeln
pytest -m perf                                   # zeitbasierte Prüfungen (siehe unten)
pytest -m live                                   # gegen das laufende Backend (~7 min)
```

---

## Sicherheitsprüfungen (Spieler-Modus, Admin-Login)

Der Spieler-Modus darf Spielern nie Spielleiter-Inhalte geben. Drei Ebenen:

- **Retrieval, deterministisch** — `tests/integration/test_player_mode_security.py`: Freigegeben
  sind eine Regel-Quelle und ein Charakterbogen (Lore), nur für die Spielleitung eine Lore-Quelle
  und eine Regel-Quelle — mit Geheimnissen, die bewusst wie Regelfragen formuliert sind. Echte
  Embeddings, ChromaDB im Speicher, ein Fake-Cross-Encoder, der Spielleiter-Text zuerst rankt;
  Cap-Backfill, Section Expansion ohne Top-Rank-Grenze, Regelwerk-Slot ohne Mindest-Score (seine
  eigene Suche in Gruppe `rules` darf die Spielleiter-Regeln nie erreichen), Vektor- und
  Hybrid-Suche, Condense mit einer Köder-Folgefrage. Weder Chunks, Quellen noch der LLM-Prompt
  dürfen ein Geheimnis enthalten. **Bilder**: Ein Köder-Porträt liegt nur in der Spielleiter-Lore;
  Charakterbogen und eine freigegebene **Datei-Quelle** (`Aushang.md`, ihr Ordner enthält alle anderen
  Quellen) betten denselben Namen ein — ihre Chunks dürfen kein `images` bekommen, und kein Bild einer
  Spieler-Antwort darf außerhalb der freigegebenen Quellen liegen (Ultrareview 13.1, mutationsgeprüft).

  Dazu ein **Spy-Test** (`test_every_store_access_of_a_player_request_carries_the_restriction`):
  Er zeichnet jeden Aufruf von Vektorsuche, BM25-Suche, Eintrags-Slot (`get_table_entries`) und
  Section-Expansion-Nachladen während
  Spieler-Anfragen auf und verlangt in jedem die Einschränkung `source_id $in [freigegeben]`. Er prüft
  die Struktur statt der Daten: Die Expansion ohne Filter war mit diesen Testdaten dicht (sie bleibt in
  der Quelle des Treffers), der Spy-Test schlägt trotzdem an. Mutationsgeprüft für alle Wege; für den
  Eintrags-Slot liegt eine Spielleiter-Tabelle mit dem Köder-Eintrag „Fluchklinge“ in den Testdaten
  (ohne Filter im Retriever oder im Store: je 5 Sicherheitstests rot, Code-Review 14.10.).

  Positiv geprüft wird außerdem, dass der Slot für Spieler wirklich greift: Belegt der
  Charakterbogen die oberen Plätze, holt er einen Chunk aus der freigegebenen Regel-Quelle — nie aus
  der Spielleiter-Regel-Quelle, die der Köder-Reranker am höchsten bewertet.

  Die Durchsetzung im Backend (`role: "player"`, `src/api/access.py`) wird eigens geprüft, je mit
  Vektor- und Hybrid-Suche: die Rolle allein schränkt ein; mitgeschickte Filter (`source_id`,
  `group`, verschachteltes `$and`, die Spielleiter-Regel-Quelle) können nur einengen; eine
  Spielleiter-Session-ID ergibt eine neue Session; `/eval/preview` ist ebenfalls eingeschränkt; ohne
  freigegebene Quelle kommt `403`.
- **Filter-Vertrag** — `tests/integration/test_filter_contract.py`: Jeder Filter, den
  `validate_filters` durchlässt, liefert in ChromaDB und in der BM25-Suche dieselben Chunks — sonst
  könnte die Hybrid-Suche eine Bedingung still ignorieren.
- **UI und Session, deterministisch** — `tests/ui/test_player_mode_security.py`: manipulierter
  oder übriggebliebener Session-State, jede Folgefrage gefiltert, Fail-closed bei Fehlern von
  `/sidebar-state`, Entzug einer Freigabe wirkt bei der nächsten Frage, Bruteforce-Schutz des
  Admin-Logins.
- **Live, `pytest -m live`** (standardmäßig ausgeschlossen) — `tests/live/test_player_mode_live.py`
  gegen das laufende Backend und den echten Index: jede Golden-Set-Frage zu Vault-Wissen (Vektor und
  Hybrid) darf im Spieler-Modus nur freigegebene Quellen liefern; fünf Spielleiter-Geheimnisse über
  den vollen Antwortweg (LLM) dürfen nicht auftauchen. Dauert ~7 Minuten.

Die deterministischen Tests sind **mutationsgeprüft**: Absichtlich eingebaute Lücken lassen sie
fehlschlagen — BM25-Filter kaputt, Section Expansion nur nach Dateiname, Vektorfilter weg,
Spielleiter-Filter in den Spieler-Filter gemischt, Fail-open, Backend-Durchsetzung entfernt,
Anfrage-Filter in der Suche des Regelwerk-Slots weggelassen (5 Tests schlagen fehl). Ebenso
mutationsgeprüft: der GPU→CPU-Rückfall bei gleichzeitigen Fragen und die Fehleranzeige im
Chunk-Inspector.

---

## Log-Pfade

Fehler und wichtige Ereignisse müssen im Log landen — nach einem Neustart ist es die einzige Spur
(Roadmap 10.0). `tests/integration/test_logging_paths.py` prüft mit `caplog`: unbehandelte
Exceptions (Middleware), fehlgeschlagene Ingest-Jobs mit Traceback, abgebrochene Streams, nur
Health-*Wechsel*, `session`/`role`/`model` an den Zeilen einer Frage, die Frage genau einmal auf
INFO, den Gemini-Key nie im Log, Log-Level je Request-Art, `logs/ollama.log` und den Warte-Hinweis
(SSE `status`, Zähler auch nach Abbruch). `tests/unit/test_logging_setup.py` prüft, dass
`uvicorn.error` in die Datei kommt, eine Datei pro Tag mit Wechsel um 6 Uhr und dass die
`suppress`-Liste in Code und `settings.yaml` gleich ist. Alle mutationsgeprüft.

---

## Performance-Prüfungen

Zwei Arten, damit eine Änderung, die LoreKeeper spürbar langsamer macht, sofort auffällt:

- **Deterministisch — Teil des normalen `pytest`.** Sie zählen Arbeit statt Zeit, sind also
  hardwareunabhängig:
  - `tests/ui/test_backend_calls.py` — Backend-Anfragen pro Seitenaufbau und Rerun (jeder Klick
    führt das Seitenskript neu aus), Anfragen pro Chat-Nachricht, und dass lokale Dienste über
    `127.0.0.1` statt `localhost` angesprochen werden (unter Windows kostet `localhost` ~2 s pro
    neuer Verbindung). Budgets stehen in `BUDGETS`; ist eine zusätzliche Anfrage gewollt, im selben
    Commit erhöhen.
  - `tests/integration/test_api_routes.py::test_llm_calls_per_request` — ein LLM-Aufruf pro
    Antwort, ein zweiter nur bei einer echten Folgefrage (Condense).
- **Zeitbasiert — `pytest -m perf`, standardmäßig ausgeschlossen.** Vergleich gegen
  `tests/perf/baseline.json` mit Toleranz (×1,5 plus kleiner fester Puffer):
  - `tests/perf/test_ui_render.py` — Renderzeit je Seite (Fake-Backend).
  - `tests/perf/test_golden_retrieval.py` — Golden Set über das **laufende** Backend
    (`127.0.0.1:8000`, sonst übersprungen): p50/p95 der Retrieval-Latenz, Chunks pro Frage
    (Ø/max — fängt aufgeblähten Kontext), Abschnitt-Trefferquote darf nicht sinken. Baseline erneuert
    14.10. nach jina und Eintrags-Slot: Abschnitt 0,96, p50/p95 272/347 ms, max. 30 Chunks (vorher
    0,891, 784/1.156 ms — der alte Wert hätte erst einen großen Rückgang bemerkt).
  - `tests/perf/test_startup.py` — startet ein zweites Backend auf freiem Port gegen die echte
    Config und `chroma_data/` und misst bis zur ersten `/health`-Antwort. **Übersprungen, solange
    LoreKeeper läuft**; auf einem ruhigen Rechner bei gestopptem LoreKeeper ausführen.
    Baseline (12.10.): 17,7 s — ~9,6 s Python-Imports, ~6,7 s Lifespan.
  - `tests/perf/test_reranker_gpu.py` — echtes Modell auf der GPU (fp16) vs. CPU-Rückfall (fp32):
    gleiches Ranking, Scores ±0,02, GPU mindestens 3× schneller. Übersprungen ohne `onnxruntime-directml`.

```powershell
pytest -m perf                                # gegen die Baseline prüfen
$env:PERF_UPDATE_BASELINE=1; pytest -m perf   # aktuelle Werte übernehmen (neue Hardware, gewollte Änderung)
pytest -m perf tests/perf/test_startup.py     # nur Start — LoreKeeper muss gestoppt sein
```

LLM-Antwortzeiten werden nicht automatisch getestet (modell- und hardwareabhängig); dafür gibt es
die `timings` pro Anfrage (Log-Zeile `Query timings:` / Zeile unter jeder Antwort im Chat).

---

## Abdeckung

`pytest --cov=src --cov=ui --cov=evaluation` — Stand 13.10.: **93 %**. Ergänzt wird nach
Fehlerrisiko, nicht nach Prozent: zuerst Pfade, die Daten schreiben oder Zustand umschalten
(Speichern, Aktivieren, Provider-Wechsel samt Rückfall, Golden Set). Bewusst ungetestet: der
Ollama-Autostart in `src/main.py` und `stop.ps1` (brauchen echte Prozesse; `stop.ps1` am 13.10.
von Hand geprüft: Dev- und Server-Modus, Ingest-Sperre, `-Force`, fremder Server auf 8000 bleibt
unberührt, ein Strg+C ignorierender Prozess wird nach 20 s beendet; Button-Startweg mit dem Skript
als Kind des UI-Prozesses — `logs/stop.log` endet nach Sekunden mit „stopped … (clean)“, nicht mit
„Still running … powershell“; Desktop-Verknüpfungen am 18.10. per Doppelklick: Start im Server-Modus
nach 14 s, zweiter Klick startet nichts doppelt, Beenden sauber nach 2 s; `stop.ps1 -Restart` am 18.10.
über den Startweg des Buttons: Server- und Entwicklungsmodus bleiben erhalten, ~25 s, Modell aus dem VRAM,
der nächste Stopp sauber nach ~2 s — mutationsgeprüft: ohne das Wiedereinschalten von Strg+C wurde das
neue Backend nach 20 s hart beendet; vom Nutzer am 18.10. im Browser: beide Knöpfe, Browserstart der
Verknüpfung, Indizierungs-Fortschritt beim Seitenwechsel, Tabs „Feedback“/„Chunks“, Bilder per Alias), das Übernehmen von Feedback per Häkchen im Data-Editor (AppTest kann keine Zellen anklicken; die Logik in `ui/feedback.py` ist getestet) und
die CLI-Einstiegspunkte (`main()`).
