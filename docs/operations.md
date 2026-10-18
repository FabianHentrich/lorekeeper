# Betrieb

Alles für den laufenden Betrieb: Indizieren, Betrieb im Heimnetz, Fehlersuche und Logs. Die
Ersteinrichtung steht im [README](../README.md#schnellstart-lokal-windows), alle Einstellungen
und Umgebungsvariablen in [configuration.md](configuration.md).

## Inhaltsverzeichnis
1. [Indizieren](#indizieren)
   - [Inkrementell](#inkrementell)
   - [Alles neu](#alles-neu)
   - [Über die API](#über-die-api)
2. [Server im Heimnetz (LAN)](#server-im-heimnetz-lan)
3. [Häufige Probleme](#häufige-probleme)
   - [Ollama antwortet nicht](#ollama-antwortet-nicht)
   - [ChromaDB-Dateisperre](#chromadb-dateisperre)
   - [Gemini 429 (Rate-Limit)](#gemini-429-rate-limit)
   - [Modelle fehlen beim ersten Start](#modelle-fehlen-beim-ersten-start)
   - [Reranker läuft auf der CPU (DirectML-Warnung)](#reranker-läuft-auf-der-cpu-directml-warnung)
   - [PDF-OCR funktioniert nicht](#pdf-ocr-funktioniert-nicht)
   - [Suche liefert das falsche Dokument](#suche-liefert-das-falsche-dokument)
4. [Logs](#logs)

---

## Indizieren

### Inkrementell

In der UI: **⚙ Sources → ⚡ Nur Änderungen indizieren**. Dateien mit unverändertem
`content_hash` (SHA-256 des Inhalts) werden übersprungen, neue und geänderte neu embedded,
gelöschte aus ChromaDB entfernt. Die Ergebniszeile zeigt indizierte vs. übersprungene Dokumente
und erzeugte/aktualisierte/gelöschte Chunks. Der Fortschritt blockiert die Seite nicht und
übersteht Klicks und Seitenwechsel.

Per CLI dasselbe — nur bei **gestopptem** Backend:

```powershell
python -m src.ingestion.orchestrator
```

### Alles neu

Nötig nach Änderungen an Chunking, Parsern oder Embed-Text — die ändern den Hash einer Datei
nicht, der inkrementelle Lauf würde alles überspringen:

- UI: **⚙ Sources → 🔄 Alle Dokumente neu indizieren**
- API: `POST /ingest?full=true`

Nach einem **anderen Embedding-Modell** (andere Vektordimension) muss die Collection erst geleert
werden: **⚙ Sources → Danger Zone** (`POST /admin/wipe`), dann normal indizieren. Oder bei
gestopptem Backend:

```powershell
Remove-Item -Recurse -Force .\chroma_data
python -m src.ingestion.orchestrator
```

Nur `category_map` / `group` geändert? Dann **Recategorize** — schreibt die Metadaten neu, ohne
neu zu embedden. Welche Aktion wann: [configuration.md](configuration.md#quellen--configsourcesyaml).

### Über die API

```powershell
curl.exe -X POST http://127.0.0.1:8000/ingest               # inkrementell
curl.exe -X POST "http://127.0.0.1:8000/ingest?full=true"   # alles
# → {"job_id": "abc123", "status": "queued"}   (409, solange ein anderer Job läuft)

curl.exe http://127.0.0.1:8000/ingest/active
# → laufender/wartender Job (Felder wie /ingest/status) oder null

curl.exe http://127.0.0.1:8000/ingest/status/abc123
# → {"status": "done", "documents_processed": 42, "documents_total": 42,
#    "documents_skipped": 39, "chunks_created": 5, "chunks_updated": 12, "chunks_deleted": 3, ...}
```

`documents_processed` zählt übersprungene Dateien mit (treibt die Fortschrittsanzeige);
`chunks_deleted` zählt die Chunks nicht mehr vorhandener Dateien. Immer `127.0.0.1` statt
`localhost` — unter Windows kostet `localhost` ~2 s pro neuer Verbindung.

---

## Server im Heimnetz (LAN)

LoreKeeper läuft auf dem Gaming-PC (Ollama + Backend + UI) und wird von Laptop oder Handy im selben
Netz bedient. Erreichbar ist nur die Streamlit-UI (Port 8501); das Backend bleibt auf
`127.0.0.1:8000`, weil die UI es serverseitig aufruft.

**Einmalige Einrichtung am Gaming-PC:**

1. Passwort in `.env` setzen — schützt die Admin-Seiten (Sources, Evaluation, Prompts, Settings)
   und den Provider-/Gemini-Key-Wechsel in der Chat-Sidebar; der Chat selbst bleibt offen:
   ```
   LOREKEEPER_ADMIN_PASSWORD=<dein Passwort>
   ```
2. Heimnetz als **Privat** einstufen (Einstellungen → Netzwerk und Internet →
   Netzwerkeigenschaften). Die Firewall-Regel gilt nur im Profil „Privat“.
3. Das Einrichtungsskript einmal **als Administrator** ausführen — es legt die Firewall-Regel an
   (eingehend TCP 8501, nur Privat) und warnt, wenn das Netz öffentlich ist oder eine
   Python-Blockregel die Freigabe überstimmen würde. Bewusst **kein Autostart**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup-lan.ps1
   powershell -ExecutionPolicy Bypass -File scripts\setup-lan.ps1 -Uninstall   # rückgängig
   ```
4. Die IP des PCs im Router reservieren (DHCP-Reservierung), damit die URL gleich bleibt.
   Ollama-Variablen für den Server: [configuration.md](configuration.md#llm) → `llm.ollama`.

**Startarten (`start.ps1`):**

| Befehl | Backend | UI | Zweck |
|---|---|---|---|
| `.\start.ps1` | `127.0.0.1`, Auto-Reload | nur `127.0.0.1`, öffnet den Browser | Entwicklung auf diesem PC |
| `.\start.ps1 -Server` | `127.0.0.1`, kein Reload | `0.0.0.0` (LAN), minimierte Fenster | Heimnetz-Server; gibt die LAN-URLs aus |
| `.\start.ps1 -Server -Browser` | wie `-Server` | wie `-Server`, öffnet zusätzlich den Browser — auch wenn LoreKeeper schon läuft | Desktop-Verknüpfung |

Dann am Laptop oder Handy `http://<ip-des-gaming-pcs>:8501` öffnen. Fragt Windows, ob Python
Verbindungen annehmen darf, nur **Private Netzwerke** erlauben — „Abbrechen“ legt eine Blockregel
an, die die Port-Freigabe überstimmt.

**Im Alltag:**

- **Starten** (kein Autostart — Absicht): `.\start.ps1 -Server` im Projektordner. Backend und UI
  öffnen sich als zwei minimierte Fenster, die Konsole zeigt die LAN-URLs.
- **Desktop-Verknüpfungen** (18.10., Roadmap 15): einmalig
  `powershell -ExecutionPolicy Bypass -File scripts\create-shortcuts.ps1` legt **LoreKeeper starten**
  (`start.ps1 -Server -Browser`) und **LoreKeeper beenden** (`stop.ps1`) auf den Desktop. Doppelklick
  genügt — keine Konsole, keine venv. Das Fenster der Verknüpfung bleibt 15 s (Beenden: 8 s) offen, damit
  LAN-URLs und Warnungen (kein Admin-Passwort, Indizierung läuft) lesbar sind. Läuft LoreKeeper schon,
  öffnet „starten“ nur die Seite. Die Verknüpfungen enthalten absolute Pfade — nach einem Umzug des
  Projektordners das Skript erneut ausführen; `-Uninstall` entfernt sie.
- **Stoppen**: `.\stop.ps1` im Projektordner oder in der UI **⏻ LoreKeeper beenden** (Sidebar,
  nur Admin). Beides schickt Strg+C an das Backend- und das UI-Fenster (wie ein Tastendruck dort),
  wartet bis zu 20 s (`-Timeout`) und beendet dann hart, was noch läuft — samt einem Ollama, das
  das Backend gestartet hat. Erkannt werden nur LoreKeepers eigene Prozesse (`src.main:app`,
  `ui/LoreKeeper.py`); ein anderer Server auf 8000/8501 bleibt unberührt. Während einer Indizierung verweigert beides (ein harter Abbruch kann
  einen halb geschriebenen Index hinterlassen); `.\stop.ps1 -Force` bricht sie trotzdem ab. Die
  Ausgabe des Buttons steht in `logs/stop.log`. `.\stop.ps1 -Restart` bzw. **🔄 LoreKeeper neu starten**
  (gleiche Stelle, 18.10., Roadmap 16) beendet genauso und startet danach `start.ps1` im Modus, der lief
  (`--reload` → Entwicklung, sonst `-Server`); scheitert das Beenden (Indizierung, Port belegt), wird nicht
  gestartet. Dauer ~25 s, Ollama samt Modell ist danach aus dem VRAM. Der Button startet das Skript in
  einer eigenen Prozessgruppe, die Strg+C abschaltet — und Kindprozesse erben das. Das Skript schaltet es
  vor dem Start wieder ein; ohne das ignorierte das neue Backend das nächste Strg+C und wurde nach 20 s
  hart beendet, samt Ollama (live geprüft 18.10.). `.\stop.ps1 -List` zeigt nur, welche Prozesse
  gestoppt würden — auch während einer Indizierung. Kindprozesse zählen nur, wenn sie **nach** ihrem Elternprozess gestartet wurden:
  Windows vergibt PIDs neu, und am 14.10. hielt das Skript so den Browser, den `start.ps1` geöffnet
  hatte, für ein Kind des UI-Prozesses und beendete ihn hart.
- **Nicht die Fenster schließen**: Das X beendet den Prozess, bevor sein Shutdown läuft — ein von
  LoreKeeper gestartetes Ollama läuft dann weiter, und im Log fehlt „Shutting down“ (13.10.: 7 von
  10 Starts endeten so). Strg+C im Fenster ist sauber; gemessen am 13.10.: Backend 1,3–1,6 s,
  UI 0,2 s, auch während eines Streams oder einer Evaluation.
- **Am Gaming-PC entwickeln**: Server stoppen (`.\stop.ps1`), dann `.\start.ps1`. Der Server-Modus lädt Code-
  Änderungen **nicht** nach: nach einem Update stoppen und neu starten.
- **Auto-Reload (`.\start.ps1`)**: Nur Änderungen unter `src/` lösen ihn aus (`--reload-dir src`);
  Code in `evaluation/` lädt das Backend erst beim nächsten Start. Uvicorn startet den Worker unter
  Windows per Strg+C an die ganze Konsole neu. Deshalb läuft ein vom Backend gestartetes Ollama in
  einer **eigenen Prozessgruppe** (sonst starb es mitten in einer Antwort, 503) und wird beim
  Herunterfahren gezielt per Strg+Pause beendet — samt `llama-server`, das Modell bleibt nicht im
  VRAM. Eine laufende Antwort wird vor dem Neustart zu Ende geschickt. Ein laufender
  End-to-End-Lauf bricht beim Neustart ab (Daemon-Thread; vorher hielt er den alten Prozess am
  Leben, und der Neustart hing ohne Ende), eine laufende Indizierung wird dagegen fertig geschrieben
  — der Neustart wartet so lange (Roadmap 12.3, 14.10.).
- **`stop.ps1` während eines Neustarts**: Startet der Reloader in den 20 s noch einen neuen Worker,
  bekommt der kein Strg+C mehr. `stop.ps1` sucht deshalb vor dem harten Beenden noch einmal nach
  Kindprozessen und beendet auch diesen samt seinem Ollama (vorher blieb er ohne Fenster übrig und
  hielt Port 8000). Ausgabe dann „some processes killed“ nach ~20 s.
- `start.ps1` überspringt einen Teil, dessen Port schon belegt ist („already running“) — ein
  übriger Teil aus dem anderen Modus erst beenden (`.\stop.ps1`).

**Alte Autostart-Aufgabe entfernen:** Die erste Version von `setup-lan.ps1` legte eine
Anmelde-Aufgabe „LoreKeeper Server“ an. Löschen braucht Admin-Rechte — `setup-lan.ps1` erneut als
Administrator ausführen (behält die Firewall-Regel, entfernt die Aufgabe) oder in einer
Admin-PowerShell:

```powershell
Unregister-ScheduledTask -TaskName "LoreKeeper Server" -Confirm:$false
```

**Warum HTTP und nicht HTTPS:** Im Heimnetz verlässt der Verkehr nie das eigene WLAN/LAN, und
Streamlit liefert standardmäßig HTTP. HTTPS bräuchte ein Zertifikat, dem die Browser trauen — ein
selbstsigniertes warnt auf jedem Gerät, ein „echtes“ braucht eine Domain. Das praktische Risiko:
Jemand **im eigenen WLAN** könnte mitlesen, auch das Admin-Passwort beim Login. Für den Zugriff
von außen (Roadmap 4.2) ist Tailscale geplant, das alles Ende-zu-Ende verschlüsselt.

**Spieler am Tisch:** Am Handy wählen Spieler **🎲 Spieler** — sie durchsuchen nur freigegebene
Quellen (Sources-Seite → Spalte „🎲 Spieler“, z.B. Regelwerk und Charakterbogen). Die
Spielleitung wählt **📜 Spielleiter**. Das ist ein Spoiler-Schutz, keine Sperre: jeder kann jede
Rolle wählen. Das Backend erzwingt die Spieler-Einschränkung selbst
([ARCHITECTURE.md](../ARCHITECTURE.md) → 4.8).

---

## Häufige Probleme

### Ollama antwortet nicht

```powershell
ollama serve            # Ollama von Hand starten
ollama list             # vorhandene Modelle
ollama pull qwen3:14b   # Modell holen, falls es fehlt
ollama ps               # geladenes Modell, Kontextlänge, GPU-Anteil
```

**„Ollama lists no models at all“ im Log**, obwohl die Modelle da sind: LoreKeeper hat Ollama selbst
gestartet (`Starting Ollama` im Log), und dieses `ollama serve` sucht die Modelle in
`%USERPROFILE%\.ollama\models`. Die Ollama-App kann einen anderen Ordner nutzen (Einstellung der
App — steht in `%LOCALAPPDATA%\Ollama\server.log` als `OLLAMA_MODELS`). Abhilfe: `OLLAMA_MODELS`
als Benutzer-Umgebungsvariable auf denselben Ordner setzen —
`[Environment]::SetEnvironmentVariable('OLLAMA_MODELS', '<Ordner der App>', 'User')`, wirkt für neu
gestartete Programme (Terminal neu öffnen); die App ändert sich dadurch nicht, es wird nichts
verschoben. Oder die Ollama-App vor LoreKeeper starten. Die Ausgabe des selbst gestarteten Ollama
steht in `logs/ollama.log`.

**Antworten 3–4× langsamer als sonst** (`Query timings` zeigt ~15–20 statt ~54 tok/s bei gleicher
Prompt-Größe, Retrieval normal): Das Modell liegt teilweise im **geteilten Speicher** (System-RAM über
PCIe). `ollama ps` sagt trotzdem „100% GPU“ — das heißt nur, dass keine Schicht auf der CPU rechnet.
Prüfen mit `Get-Counter '\GPU Process Memory(*)\Shared Usage'`: Ein `llama-server` mit mehreren GB
„shared usage“ ist der Fall. Am 17./18.10. hielt `dwm.exe` (Windows-Desktop) 7,85 GB VRAM, qwen3:14b
lag mit 2,65 GB im geteilten Speicher, 16,9 statt ~54 tok/s; ein Neuladen des Modells half nicht, weil
der Speicher nicht frei war. Abhilfe: das Programm schließen, für das `dwm` den Speicher hält (meist ein
Spiel im Vollbild/randlos), sonst ab- und wieder anmelden oder neu starten; danach das Modell neu laden
(**🔄 LoreKeeper neu starten** in der Sidebar beendet Ollama mit, oder `ollama stop qwen3:14b`; die nächste
Frage lädt es). Seit 18.10. prüft der Health-Loop das alle 5 Minuten (`src/gpu_memory.py`, ein
PowerShell-Aufruf von ~1–2 s): über 1 GB steht `LLM partly in shared GPU memory: … GB` als WARNING im Log
und eine Warnung in der Sidebar, beim Zurückfallen `LLM back in VRAM` (INFO).

**Mehrere Fragen gleichzeitig**: Ollama beantwortet sie nacheinander (`OLLAMA_NUM_PARALLEL=1`,
bewusst so gelassen). `OLLAMA_NUM_PARALLEL=2` wurde gemessen: Das erste Token kommt bei Gleichzeitigen
schneller, aber jede Antwort läuft langsamer — auch allein (−27 %), bei ~2,6 GB mehr VRAM (Messung: [provider-strategy.md](provider-strategy.md)).

### ChromaDB-Dateisperre

Entsteht, wenn Backend und Ingestion-CLI gleichzeitig laufen (embedded): Backend stoppen,
`python -m src.ingestion.orchestrator`, Backend wieder starten — oder gleich über die UI
indizieren. Scheitert der Abgleich der Datei-Hashes an der Datenbank, endet der Indizier-Job mit
Fehler, statt still alle Dateien für neu zu halten und alte Chunks liegen zu lassen.

### Gemini 429 (Rate-Limit)

- Kontingent erschöpft (`limit: 0`) → Abrechnung aktivieren oder warten. Achtung: Budgets in Google
  Cloud warnen nur, sie stoppen nicht.
- LoreKeeper wiederholt automatisch (bis zu 3×, exponentielles Backoff).

### Modelle fehlen beim ersten Start

LoreKeeper lädt Modelle seit 14.10. nur aus dem lokalen Cache (`HF_HUB_OFFLINE=1`, gesetzt in
`src/__init__.py`; [configuration.md](configuration.md#umgebungsvariablen)). Gebraucht werden
`intfloat/multilingual-e5-base` (~1,1 GB) und vom Reranker `jina-reranker-v2-base-multilingual`
`tokenizer.json`, `onnx/model_fp16.onnx` (~560 MB, GPU) und `onnx/model.onnx` (~1,1 GB, CPU-Rückfall).
Fehlt das Embedding-Modell, bricht der Start mit `OSError: We couldn't connect to
'https://huggingface.co' … couldn't find them in the cached files` ab; fehlt `onnx/model.onnx` beim
CPU-Rückfall, mit `LocalEntryNotFoundError` („… cannot find the requested files in the local cache …“);
fehlt nur die GPU-Datei, läuft der Reranker mit Warnung auf der CPU.

**Beheben**: einmal den Vorab-Download aus README → Schnellstart, Schritt 4 ausführen (der Befehl
importiert `src` nicht und darf deshalb herunterladen), oder das Backend einmal mit erlaubtem Download
starten:

```powershell
$env:HF_HUB_OFFLINE = "0"; .\start.ps1     # lädt fehlende Modelle; danach wieder normal starten
```

Das gilt auch nach einem Wechsel von `embeddings.model` oder `reranking.model`.

### Reranker läuft auf der CPU (DirectML-Warnung)

Sichtbar als Warnung in der Chat-Sidebar („Reranker läuft auf der CPU statt der GPU“) und als
Log-Zeile `DirectML reranker unavailable (…) — falling back to the CPU`: Die Antworten
bleiben praktisch gleich (fp32 statt fp16), nur die Suche ist langsamer (Sekunden statt ~0,1 s,
unter CPU-Last mehr). Ohne Internet und ohne `onnx/model.onnx` im Cache scheitert der Rückfall.
Übliche Ursache: ein `pip install -r requirements.txt` hat das normale `onnxruntime` wieder über
`onnxruntime-directml` gelegt (beide teilen das Modul `onnxruntime/`). Prüfen und beheben:

```powershell
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"   # braucht DmlExecutionProvider
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-directml
```

`pip check` meldet danach, dass chromadb, pymupdf-layout und rapidocr `onnxruntime` vermissen — das
ist erwartet, das Modul ist da. Bei Erfolg steht im Log `Reranker loaded on the GPU (DirectML)`.

Steht stattdessen `DirectML reranker failed (…) — switching to the CPU for this run`, ist
die GPU **im Betrieb** ausgefallen (Treiber-Reset, VRAM voll — z.B. ein Spiel neben Ollama). Die
Antworten laufen auf der CPU weiter; nach einem Neustart des Backends ist der Reranker wieder auf
der GPU.

### PDF-OCR funktioniert nicht

- Installiert? `pip show rapidocr_onnxruntime`
- Log nach `pymupdf4llm failed for ... retrying without OCR` durchsuchen — der Parser fällt
  automatisch auf „ohne OCR“ zurück.
- Andere Sprache als Deutsch: `ingestion.pdf.ocr_language` (ISO 639-3: `eng`, `fra`, `deu`, …).
- OCR ist CPU-lastig; große gescannte PDFs dauern deutlich länger als Text-PDFs.

### Suche liefert das falsche Dokument

1. Evaluation → **Frage testen**: Welche Chunks liefert die Frage, mit welchem Score?
2. Evaluation → **Chunks**: Wie ist die erwartete Datei zerlegt — gibt es den Inhalt als eigenen
   Chunk, stimmt die Überschrift?
3. Nach Chunking-Änderungen alles neu indizieren („🔄 Alle Dokumente neu indizieren“).
4. Filter in der Chat-Sidebar prüfen (Gruppe abgewählt? Spieler-Modus?).
5. `top_k` erhöhen (mehr Kandidaten für den Reranker); `score_threshold` nicht unter 0,5 senken.

---

## Logs

```powershell
Get-Content logs\lorekeeper.log -Tail 50                                  # laufender Tag
Select-String -Path logs\lorekeeper*.log -Pattern 'ERROR' | Select-Object -Last 20   # alle Tage
```

Eine Datei pro Tag: Der laufende Tag steht in `lorekeeper.log`, vergangene Tage in
`lorekeeper.<JJJJ-MM-TT>.log`. Der Tag wechselt um 6 Uhr (`logging.rotate_hour`), damit ein
Spielabend über Mitternacht in einer Datei bleibt; behalten werden 30 Tage (`logging.backup_count`).

Pro Frage stehen auf INFO: `Question (session=… role=…): '…'` (bei Folgefragen mit `→ condensed
'…'`, die Fassung, mit der gesucht wurde), die Rerank-Zusammenfassung, ggf. `Section expansion: …`
und `Query timings: …` (Modell, Condense, Retrieval-Stufen, erster Token, gesamt, Chunks,
Prompt-Größe, seit 18.10. auch `tokens_in=… tokens_out=… <n>tok/s`, sofern der Provider Tokens meldet —
die Rate rechnet bei `/query` über den ganzen LLM-Aufruf inkl. Prefill, beim Stream ab dem ersten Token;
Richtwert für qwen3:14b ganz im VRAM ~54 tok/s). Die Frage steht nur in der ersten dieser Zeilen; die anderen ordnet man über
`session=` bzw. die Reihenfolge zu. Erfolgreiche GET-Anfragen (Seitenaufrufe, Job-Polling) stehen
nur auf DEBUG; die Zeile von `/query/stream` misst nur den Start des Streams. Den Log-Level setzt
`logging.level` (`DEBUG` zeigt u.a. alle Rerank-Scores, jede expandierte oder verworfene Sektion
und den Grund fehlgeschlagener Health-Checks).

Ein von LoreKeeper gestartetes Ollama schreibt seine eigene Ausgabe nach `logs/ollama.log`
(angehängt, ab 10 MB nach `ollama.log.1` verschoben) — dort steht der Grund, wenn Ollama abstürzt.
Läuft Ollama schon vorher (Tray-App), hat es sein eigenes Log unter
`%LOCALAPPDATA%\Ollama\server.log`.

**Fehler suchen** — `Select-String -Path logs\lorekeeper.log -Pattern 'ERROR|Traceback' -Context 0,20`.
Was im Log steht:
- **Unbehandelte Fehler** einer Anfrage: eine Zeile `<METHODE> <pfad> → unhandled exception`, danach
  der Traceback von uvicorn (`uvicorn.error` wird in die Datei geleitet; vorher stand er nur in der
  Konsole). Auch ein Absturz beim Start steht mit Traceback in der Datei.
- **Ingest-Jobs**: `Ingest job <id> started (<art>)`, dann `done (…): N file errors` oder
  `failed` mit Traceback. Die Job-Liste lebt nur im Speicher — nach einem Neustart ist das Log die
  einzige Spur. Fehler einzelner Dateien stehen als `Error processing <pfad>` mit Traceback.
- **Eval-Jobs** (`Retrieval eval job <id> failed`), **LLM-Fehler** (`LLM generation failed`,
  `Streaming generation failed`) und der **Fallback** (`Primary provider failed …, trying
  fallback`, WARNING) mit Traceback — der Fallback antwortet, ein Fehler im Hauptprovider fällt
  sonst nicht auf.
- **Abgebrochene Antworten** (Seite geschlossen, Lese-Timeout der UI): `Stream aborted by the client
  after <ms> (<n> chars sent, …)` — es fehlt dann die `Query timings`-Zeile.
- **Zuordnung**: `Query timings`, LLM-Fehler und abgebrochene Streams tragen `session=<8 Zeichen>
  role=<rolle>` (Timings auch `model=` — nach einem Fallback das Modell, das wirklich geantwortet
  hat). Fragen mehrere Leute gleichzeitig, nach `session=` filtern.
- **Health**: nur Wechsel — `Health changed: chroma=… llm=… (was …)`, WARNING wenn etwas ausfällt,
  INFO bei Erholung. Fehlt ein Ollama-Modell, kommt der Hinweis `ollama pull …` einmal pro Ausfall.
  Den Grund eines fehlgeschlagenen Checks zeigt `logging.level: DEBUG`.
- **Änderungen**: `Sources saved: … player_visible=[…]` (welche Quellen Spieler durchsuchen
  können), `Source removed`, `Settings updated: {…}` mit den Werten, `Gemini runtime key set/cleared`
  (nie der Key selbst).
- **Abgelehnte Spieler-Anfragen**: `Refused player query/preview: …` mit Grund (keine Quelle
  freigegeben, ungültiger Filter).
- **Admin-Login** (UI-Prozess, keine Logdatei → Streamlit-Konsolenfenster): jeder Fehlversuch und
  die Sperre nach 5 Fehlversuchen als WARNING.
- Die CLIs `python -m src.ingestion.orchestrator` und `…recategorize` schreiben mit derselben
  Konfiguration in dieselbe Datei.

Bewertungen (👍/👎 im Chat) landen in `logs/feedback.jsonl`, ein JSON-Objekt pro Zeile. Eine
geänderte oder zurückgenommene Bewertung ist eine neue Zeile mit derselben `message_id` — die letzte
Zeile je `message_id` gilt (`rating: null` = zurückgenommen); `comment` enthält den optionalen
Freitext zur Bewertung. Auswerten geht am einfachsten über
Evaluation → **Feedback**, oder:

```powershell
Get-Content logs\feedback.jsonl | ConvertFrom-Json | Where-Object rating -eq 'down' |
  Select-Object ts, role, question, comment, sources
```
