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

Dann am Laptop oder Handy `http://<ip-des-gaming-pcs>:8501` öffnen. Fragt Windows, ob Python
Verbindungen annehmen darf, nur **Private Netzwerke** erlauben — „Abbrechen“ legt eine Blockregel
an, die die Port-Freigabe überstimmt.

**Im Alltag:**

- **Starten** (kein Autostart — Absicht): `.\start.ps1 -Server` im Projektordner. Backend und UI
  öffnen sich als zwei minimierte Fenster, die Konsole zeigt die LAN-URLs.
- **Stoppen**: beide LoreKeeper-Fenster schließen.
- **Am Gaming-PC entwickeln**: Server stoppen, dann `.\start.ps1`. Der Server-Modus lädt Code-
  Änderungen **nicht** nach: nach einem Update stoppen und neu starten.
- `start.ps1` überspringt einen Teil, dessen Port schon belegt ist („already running“) — ein
  übriges Fenster aus dem anderen Modus erst schließen.

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

Beim ersten Start lädt LoreKeeper `intfloat/multilingual-e5-base` (~1,1 GB) und
`mmarco-mMiniLMv2-L12-H384-v1` (~120 MB) von Hugging Face, mit `reranking.backend: directml` auch
den ONNX-Export des Rerankers (`onnx/model.onnx`, ~450 MB). Danach liegt alles in
`~/.cache/huggingface/`. Vorab laden: README → Schnellstart, Schritt 4.

### Reranker läuft auf der CPU (DirectML-Warnung)

Log-Zeile `DirectML reranker unavailable (…) — falling back to the CPU (torch)`: Die Antworten
bleiben gleich, nur die Suche ist langsamer (~0,8 s statt ~0,05 s, unter CPU-Last viel mehr).
Übliche Ursache: ein `pip install -r requirements.txt` hat das normale `onnxruntime` wieder über
`onnxruntime-directml` gelegt (beide teilen das Modul `onnxruntime/`). Prüfen und beheben:

```powershell
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"   # braucht DmlExecutionProvider
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-directml
```

`pip check` meldet danach, dass chromadb, pymupdf-layout und rapidocr `onnxruntime` vermissen — das
ist erwartet, das Modul ist da. Bei Erfolg steht im Log `Reranker loaded on the GPU (DirectML)`.

Steht stattdessen `DirectML reranker failed (…) — switching to the CPU (torch) for this run`, ist
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
Get-Content logs\lorekeeper.log -Tail 50
```

Pro Frage stehen auf INFO: `Question (session=… role=…): '…'` (bei Folgefragen mit `→ condensed
'…'`, die Fassung, mit der gesucht wurde), die Rerank-Zusammenfassung, ggf. `Section expansion: …`
und `Query timings: …` (Modell, Condense, Retrieval-Stufen, erster Token, gesamt, Chunks,
Prompt-Größe). Die Frage steht nur in der ersten dieser Zeilen; die anderen ordnet man über
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
Zeile je `message_id` gilt (`rating: null` = zurückgenommen). Auswerten geht am einfachsten über
Evaluation → **Feedback**, oder:

```powershell
Get-Content logs\feedback.jsonl | ConvertFrom-Json | Where-Object rating -eq 'down' |
  Select-Object ts, role, question, sources
```
