# Mitarbeit an LoreKeeper

Dieses Dokument beschreibt die Entwicklungsumgebung, die Regeln für Code-Änderungen, die Tests
und den Ablauf eines Pull Requests.

---

## Entwicklungsumgebung

Einrichtung (venv, Abhängigkeiten inkl. DirectML-Tausch, Modelle, `.env`):
[README.md](README.md) → „Schnellstart“. LoreKeeper läuft nur unter Windows.

Backend und UI für die Entwicklung getrennt starten:

```powershell
uvicorn src.main:app --reload --port 8000   # Backend
streamlit run ui/LoreKeeper.py              # UI (zweites Terminal)
```

oder beides zusammen mit `.\start.ps1`.

---

## Projektaufbau

Gesamtüberblick: [ARCHITECTURE.md](ARCHITECTURE.md). Die wichtigsten Ordner:

- `src/api/` — FastAPI-Routen, Pydantic-Schemas, Zugriff nach Rolle (`access.py`)
- `src/ingestion/` — Parser, Chunking, Ingestion-Orchestrator
- `src/retrieval/` — Embeddings, ChromaDB, BM25, Retriever, GPU-Reranker
- `src/generation/` — LLM-Provider und Generator
- `src/conversation/` — Sessions
- `ui/` — Streamlit-Oberfläche
- `tests/unit/` — schnelle Unit-Tests (externe Dienste gemockt)
- `tests/integration/` — API-Schicht und RAG-Pipeline
- `tests/ui/` — Streamlit-Seiten mit `AppTest` gegen ein Fake-Backend
- `tests/perf/` — zeitbasierte Prüfungen (`pytest -m perf`)
- `tests/live/` — Prüfungen gegen das laufende Backend (`pytest -m live`)

---

## Regeln für Code-Änderungen

- **Erst lesen, dann schreiben.** Jede Datei, die du änderst, vorher vollständig lesen.
- **So wenig wie nötig.** Nur ändern, was die Aufgabe verlangt — kein Refactoring nebenbei, keine
  Aufräum-Commits, keine Kommentare oder Typannotationen in unverändertem Code.
- **Keine Über-Abstraktion.** Keine Abstraktionen für Einmal-Logik; drei ähnliche Zeilen sind
  besser als eine verfrühte Abstraktion.
- **Provider-Interface einhalten.** Neue LLM-Backends implementieren
  [`BaseLLMProvider`](src/generation/providers/base.py) (`generate`, `generate_stream` als
  `async def` mit `yield`, `health_check`), setzen `self.model` im `__init__` (es gibt keinen
  Default), schreiben die Token-Zählung in `stream_result.usage` und werden in der
  `ProviderFactory` registriert.
- **Async-Grenzen beachten.** Jeder CPU-lastige Aufruf (vor allem `encode()` / `predict()`) läuft
  über `fastapi.concurrency.run_in_threadpool`. Den Event-Loop zu blockieren ist ein Fehler.
- **ChromaDB embedded.** Die Ingestion-CLI nie starten, während das Backend `chroma_data/` hält —
  das führt zu Dateisperren. Nie direkt in `chroma_data/` schreiben, solange das Backend läuft.
- **Tests sind Pflicht.** Neue Logik in `src/` (außer dünner Verdrahtung in `src/api/routes.py`)
  bekommt einen Unit-Test. Routenverhalten gehört in `tests/integration/` (`test_api_routes.py`
  für Query/Session/Config, `test_sources_routes.py` für `/sources/*`, Ingest-Jobs und
  `/admin/wipe`).
- **Fehlerbehandlung nur an Systemgrenzen** (Nutzereingaben, externe APIs, Dateisystem) — keine
  Handler für Fälle, die intern nicht auftreten können.
- **Lint und Typen.** `ruff check .` und `mypy` müssen sauber sein; beide laufen als Teil von
  `pytest` (`tests/test_code_quality.py`), Konfiguration in `pyproject.toml`.
- **Keine Geheimnisse in der Config.** Schlüssel kommen aus `.env` oder Umgebungsvariablen, nie aus
  `config/settings.yaml`.
- **Neue Abhängigkeiten** nur nach Absprache und immer in `requirements.txt`.
- **Score-Schwelle 0,5** nicht senken — bei 0,3 kommen fast orthogonale Chunks in den Kontext.

---

## Tests ausführen

```powershell
pytest                                        # komplette Suite (~1 min)
pytest --cov=src --cov-report=term-missing    # mit Coverage
pytest tests/unit/retrieval/test_embeddings.py   # eine Datei
pytest tests/unit                             # nur Unit-Tests
ruff check .; mypy                            # Lint + Typen einzeln
```

Alle Tests mocken externe Dienste (Ollama, Gemini; ChromaDB in Unit-Tests). Integrationstests
nutzen `chromadb.EphemeralClient` (im Speicher). **Kein echter LLM-Aufruf in der Testsuite.**

`tests/integration/test_e2e_ingest_query.py` ist der End-to-End-Test: Er indiziert einen kleinen
Vault über HTTP und fragt ihn über `/query/stream` ab — echtes Parsen, echte Embeddings (kleines
MiniLM-Modell), echte ChromaDB, Fake-LLM. Er lädt ein Modell und dauert daher länger als der Rest.

### Sicherheitsprüfungen (Spieler-Modus, Admin-Login)

Der Spieler-Modus darf Spielern nie Spielleiter-Inhalte geben. Drei Ebenen:

- **Retrieval, deterministisch** — `tests/integration/test_player_mode_security.py`: eine
  freigegebene Regel-Quelle und eine Spielleiter-Quelle mit Geheimnissen, die bewusst wie
  Regelfragen formuliert sind; echte Embeddings, ChromaDB im Speicher, ein Fake-Cross-Encoder, der
  Spielleiter-Text zuerst rankt; Cap-Backfill, Section Expansion ohne Top-Rank-Grenze, Vektor- und
  Hybrid-Suche, Condense mit einer Köder-Folgefrage. Weder Chunks, Quellen noch der LLM-Prompt
  dürfen ein Geheimnis enthalten. Die Durchsetzung im Backend (`role: "player"`,
  `src/api/access.py`) wird eigens geprüft: die Rolle allein schränkt ein, mitgeschickte Filter
  (`source_id`, `group`, verschachteltes `$and` auf Spielleiter-Quellen) können nur einengen, eine
  Spielleiter-Session-ID ergibt eine neue Session, `/eval/preview` ist ebenfalls eingeschränkt, ohne
  freigegebene Quelle kommt `403`.
- **UI und Session, deterministisch** — `tests/ui/test_player_mode_security.py`: manipulierter
  oder übriggebliebener Session-State, jede Folgefrage gefiltert, Fail-closed bei Fehlern von
  `/sidebar-state`, Entzug einer Freigabe wirkt bei der nächsten Frage, Bruteforce-Schutz des
  Admin-Logins.
- **Live, `pytest -m live`** (standardmäßig ausgeschlossen) — `tests/live/test_player_mode_live.py`
  gegen das laufende Backend und den echten Index: jede Golden-Set-Frage zu Vault-Wissen (Vektor und
  Hybrid) darf im Spieler-Modus nur freigegebene Quellen liefern; fünf Spielleiter-Geheimnisse über
  den vollen Antwortweg (LLM) dürfen nicht auftauchen. Dauert ~7 Minuten.

Die deterministischen Tests sind mutationsgeprüft: Absichtlich eingebaute Lücken (BM25-Filter
kaputt, Section Expansion nur nach Dateiname, Vektorfilter weg, Spielleiter-Filter in den
Spieler-Filter gemischt, Fail-open, Backend-Durchsetzung entfernt, …) lassen sie fehlschlagen.

### Performance-Prüfungen

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
    (Ø/max — fängt aufgeblähten Kontext), Abschnitt-Trefferquote darf nicht sinken.
  - `tests/perf/test_startup.py` — startet ein zweites Backend auf freiem Port gegen die echte
    Config und `chroma_data/` und misst bis zur ersten `/health`-Antwort. **Übersprungen, solange
    LoreKeeper läuft**; auf einem ruhigen Rechner bei gestopptem LoreKeeper ausführen.
    Baseline (12.10.): 17,7 s — ~9,6 s Python-Imports, ~6,7 s Lifespan.
  - `tests/perf/test_reranker_gpu.py` — echtes Modell auf GPU vs. CPU: identisches Ranking, GPU
    mindestens 3× schneller. Übersprungen ohne `onnxruntime-directml`.

```powershell
pytest -m perf                                # gegen die Baseline prüfen
$env:PERF_UPDATE_BASELINE=1; pytest -m perf   # aktuelle Werte übernehmen (neue Hardware, gewollte Änderung)
pytest -m perf tests/perf/test_startup.py     # nur Start — LoreKeeper muss gestoppt sein
```

LLM-Antwortzeiten werden nicht automatisch getestet (modell- und hardwareabhängig); dafür gibt es
die `timings` pro Anfrage (Log-Zeile `Query timings:` / Zeile unter jeder Antwort im Chat).

---

## Pull Requests

1. Feature-Branch anlegen (`feat/kurzer-name` oder `fix/kurzer-name`).
2. Änderung mit Tests umsetzen.
3. Komplette Testsuite lokal laufen lassen — alles muss grün sein.
4. Geändertes Verhalten in `README.md`, `ARCHITECTURE.md` oder `docs/` nachziehen — jede
   Information steht an genau einer Stelle (siehe Dokumenttabelle in ARCHITECTURE.md).
5. Erledigte Roadmap-Punkte in `plan roadmap.md` abhaken (was umgesetzt wurde, Messwerte, offene
   Reste). Ein Changelog gibt es nicht — die Historie steht in Roadmap und `git log`.
6. PR mit klarer Beschreibung öffnen: **was** sich ändert und **warum**.

Geprüft wird: Korrektheit und Tests, die Async-Regeln, ob die Doku stimmt, und ob der Diff beim
Thema bleibt.

---

## Fehler melden

Issue mit: erwartetem Verhalten, tatsächlichem Verhalten (Stacktrace, Log-Ausgabe aus
`logs/lorekeeper.log`), Schritten zum Nachstellen und Umgebung (Windows-Version, Python-Version,
Provider Ollama/Gemini).

---

## Lizenz

Beiträge stehen unter der [MIT-Lizenz](LICENSE).
