# Operations

This document covers operational instructions for LoreKeeper: initial setup workflows, running ingestion manually or via API, and troubleshooting common issues.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Ingestion](#ingestion)
   - [Normal Ingest (incremental)](#normal-ingest-incremental)
   - [Full Re-ingest](#full-re-ingest)
   - [Ingest via API (asynchronous)](#ingest-via-api-asynchronous)
   - [Exclusion Patterns](#exclusion-patterns)
3. [Home-network server (LAN)](#home-network-server-lan)
4. [Configuration](#configuration)
5. [Common Issues](#common-issues)
   - [Ollama not responding](#ollama-not-responding)
   - [ChromaDB file lock](#chromadb-file-lock)
   - [Gemini 429 (rate limit)](#gemini-429-rate-limit)
   - [Reranker model not found](#reranker-model-not-found)
   - [PDF OCR not working](#pdf-ocr-not-working)
   - [Retrieval returns wrong document](#retrieval-returns-wrong-document)
6. [Logs](#logs)

---

## Initial Setup

```powershell
# 1. Create environment file
copy .env.example .env
# Enter GEMINI_API_KEY (optional, only for Gemini provider)

# 2. Configure sources
# config/sources.yaml — see docs/configuration.md "Sources sidecar"

# 3. Pull Ollama model
ollama pull qwen3:14b

# 4. Start backend (loads embedding model + ChromaDB on first start)
uvicorn src.main:app --reload --port 8000

# 5. Index documents (first time)
python -m src.ingestion.orchestrator
# or in the UI: ⚙ Sources page → "⚡ Nur Änderungen indizieren"
```

---

## Ingestion

### Normal Ingest (incremental)
```powershell
python -m src.ingestion.orchestrator
```
Files with an unchanged `content_hash` (SHA-256 of file content) are skipped.
New and modified files are re-embedded. Deleted files are removed from ChromaDB.

In the UI: **⚙ Sources → ⚡ Nur Änderungen indizieren**. The result line shows
how many documents were indexed vs. skipped as unchanged, plus chunks
created / updated / deleted.

### Full Re-ingest
Required after changes to chunking settings or parser / embed-text logic — these
don't change a file's hash, so the incremental run would skip everything:

- UI: **⚙ Sources → 🔄 Alle Dokumente neu indizieren**
- API: `POST /ingest?full=true`

After changing the **embedding model** (different vector dimension) the collection
must be emptied first: **⚙ Sources → Danger Zone** (`POST /admin/wipe`), then a
normal ingest. Alternatively, with the backend stopped:

```powershell
Remove-Item -Recurse -Force .\chroma_data
python -m src.ingestion.orchestrator
```

Changed `category_map` / `group` only? Use **Recategorize** instead — it rewrites
the metadata without re-embedding.

### Ingest via API (asynchronous)
```bash
curl -X POST http://localhost:8000/ingest              # incremental
curl -X POST "http://localhost:8000/ingest?full=true"  # full re-ingest
# → {"job_id": "abc123", "status": "queued"}
# → 409 if another ingest/reindex job is still running

curl http://localhost:8000/ingest/status/abc123
# → {"status": "done", "documents_processed": 42, "documents_total": 42,
#    "documents_skipped": 39, "chunks_created": 5, "chunks_updated": 12, "chunks_deleted": 3, ...}
```

`documents_processed` includes skipped files (it drives the progress display);
`chunks_deleted` counts the deleted chunks of files that no longer exist.

### Exclusion Patterns
Configured in `settings.yaml` under `ingestion.exclude_patterns`:
```yaml
exclude_patterns:
  - ".obsidian/*"    # Obsidian configuration
  - ".trash/*"       # Obsidian trash
  - "*alt.md"        # Backup versions
  - "*(1).md"        # Duplicates
  - "*.draft.*"      # Drafts
```

---

## Home-network server (LAN)

Run LoreKeeper on the gaming PC (Ollama + backend + UI) and use it from a laptop
in the same network. Only the Streamlit UI (port 8501) is reachable; the backend
stays on `127.0.0.1:8000` because the UI calls it server-side.

**One-time setup on the gaming PC:**

1. Set a password in `.env` — it protects the admin pages (Sources, Evaluation,
   Prompts, Settings) and the provider/Gemini-key controls in the chat sidebar;
   the chat itself stays open:
   ```
   LOREKEEPER_ADMIN_PASSWORD=<your password>
   ```
2. Classify the home network as **Private** (Settings → Network & Internet →
   network properties). The firewall rule only applies to the Private profile.
3. Run the setup script once **as administrator** — it adds the firewall rule
   (inbound TCP 8501, Private only) and warns if the network is Public or a Python
   block rule would override the allow rule. There is deliberately no autostart —
   start the server yourself with `.\start.ps1 -Server` when you need it:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup-lan.ps1
   powershell -ExecutionPolicy Bypass -File scripts\setup-lan.ps1 -Uninstall   # undo
   ```
4. Reserve the PC's IP address in the router (DHCP reservation) so the URL
   stays the same. Ollama settings for the server: `OLLAMA_CONTEXT_LENGTH=16384`,
   `OLLAMA_KEEP_ALIVE=1h` (see `docs/provider-strategy.md`).

**Start modes (`start.ps1`):**

| Command | Backend | UI | Use |
|---|---|---|---|
| `.\start.ps1` | `127.0.0.1`, auto-reload | `127.0.0.1` only, opens browser | Development on this PC |
| `.\start.ps1 -Server` | `127.0.0.1`, no reload | `0.0.0.0` (LAN), minimized windows | Home-network server; prints the LAN URLs |

Then open `http://<gaming-pc-ip>:8501` on the laptop or phone (tested on a phone
in the home Wi-Fi). If Windows asks whether Python may accept connections, allow
**Private networks** only — "Cancel" creates a block rule that overrides the port rule.

**Day-to-day use:**

- **Start the server** (no autostart — on purpose): `.\start.ps1 -Server` in the
  project folder. Backend and UI open as two minimized windows in the taskbar;
  the console prints the LAN URLs.
- **Stop**: close both LoreKeeper windows.
- **Develop on the gaming PC**: stop the server, then `.\start.ps1` — auto-reload,
  UI only on this PC. Server mode does **not** reload on code changes: after an
  update, stop and start `.\start.ps1 -Server` again.
- `start.ps1` skips a part whose port is already in use ("already running"), so a
  leftover window from the other mode has to be closed first.

**Removing the old autostart task:** the first version of `setup-lan.ps1` registered
a logon task "LoreKeeper Server". It needs admin rights to delete — either run
`setup-lan.ps1` again as administrator (keeps the firewall rule, removes the task)
or in an admin PowerShell:

```powershell
Unregister-ScheduledTask -TaskName "LoreKeeper Server" -Confirm:$false
```

**Why HTTP and not HTTPS:** inside the home network the traffic never leaves your
own Wi-Fi/LAN, and Streamlit serves plain HTTP by default. HTTPS would need a
certificate the browsers trust — a self-signed one triggers a warning on every
device, a "real" one needs a domain name. The practical risk of HTTP at home is
that someone **in your Wi-Fi** could read the traffic, including the admin
password at login. For access from outside (4.2) the plan is Tailscale, which
encrypts everything end-to-end, so plain HTTP inside the tunnel is fine there too.

**Players at the table:** on their phones, players pick **🎲 Spieler** — they can only
search the sources released for players (Sources page → column "🎲 Spieler"; default:
rulebook and Charakterbogen). The game master picks **📜 Spielleiter**. This is a
spoiler guard, not a lock: anyone can choose either role (see `docs/ui-ux.md`).

Access from outside the home network (Tailscale) is planned as roadmap 4.2.

---

## Configuration

### Override Order
```
Env variables  >  .env file  >  config/settings.local.yaml  >  config/settings.yaml  >  Pydantic defaults
```

### Key Env Variables
| Variable | Effect |
|----------|--------|
| `GEMINI_API_KEY` | Enable Gemini provider |
| `LLM__PROVIDER` | Provider at startup (`ollama` or `gemini`) |
| `LLM__OLLAMA__MODEL` | Override Ollama model |
| `CHROMA_MODE` | `embedded` (default) or `client` (external ChromaDB server, unused) |
| `LOREKEEPER_ADMIN_PASSWORD` | UI: password for admin pages and provider controls (empty = open) |
| `LOREKEEPER_API_URL` | UI: backend URL (default `http://127.0.0.1:8000`) |

---

## Common Issues

### Ollama not responding
```powershell
ollama serve          # start Ollama manually
ollama list           # check available models
ollama pull qwen3:14b  # pull model if missing
```

### ChromaDB file lock
Occurs when the backend and ingestion script run simultaneously (embedded mode):
→ Stop backend, run `python -m src.ingestion.orchestrator`, then restart backend.

### Gemini 429 (rate limit)
- Free Tier: 15 RPM; if `limit: 0`, quota is exhausted → enable billing
- LoreKeeper retries automatically (max 3×, exponential backoff)

### Reranker model not found
On first start, LoreKeeper downloads `intfloat/multilingual-e5-base` (~1.1 GB) and `mmarco-mMiniLMv2-L12-H384-v1` (~120 MB) from HuggingFace.
Afterwards it is cached at `~/.cache/huggingface/`.

### PDF OCR not working
- Verify `rapidocr_onnxruntime` is installed: `pip show rapidocr_onnxruntime`
- Check logs for `pymupdf4llm failed for ... retrying without OCR` — the parser falls back automatically
- For non-German PDFs, change `ingestion.pdf.ocr_language` in `settings.yaml` (uses ISO 639-3 codes: `eng`, `fra`, `deu`, etc.)
- OCR is CPU-intensive; large scanned PDFs may take significantly longer than text PDFs

### Retrieval returns wrong document
1. Check if a full re-ingest is needed (chunking changed → "🔄 Alle Dokumente neu indizieren")
2. Check `score_threshold` in `settings.yaml` (default: 0.5)
3. Increase `top_k` (more candidates for the reranker)
4. Disable the category filter in the sidebar

---

## Logs

```powershell
Get-Content logs\lorekeeper.log -Tail 50
```

Log level in `settings.yaml`:
```yaml
logging:
  level: INFO   # DEBUG | INFO | WARNING | ERROR
```
