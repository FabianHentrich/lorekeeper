# Operations

This document covers operational instructions for LoreKeeper: initial setup workflows, running ingestion manually or via API, troubleshooting common issues, and Docker execution.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Ingestion](#ingestion)
   - [Normal Ingest (incremental)](#normal-ingest-incremental)
   - [Full Re-ingest](#full-re-ingest)
   - [Ingest via API (asynchronous)](#ingest-via-api-asynchronous)
   - [Exclusion Patterns](#exclusion-patterns)
3. [Configuration](#configuration)
4. [Common Issues](#common-issues)
   - [Ollama not responding](#ollama-not-responding)
   - [ChromaDB file lock](#chromadb-file-lock)
   - [Gemini 429 (rate limit)](#gemini-429-rate-limit)
   - [Reranker model not found](#reranker-model-not-found)
   - [PDF OCR not working](#pdf-ocr-not-working)
   - [Retrieval returns wrong document](#retrieval-returns-wrong-document)
5. [Docker](#docker)
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
ollama pull qwen3:8b

# 4. Start backend (loads embedding model + ChromaDB on first start)
uvicorn src.main:app --reload --port 8000

# 5. Index documents (first time)
python -m src.ingestion.orchestrator
# or via the sidebar: "🔄 Re-index Documents"
```

---

## Ingestion

### Normal Ingest (incremental)
```powershell
python -m src.ingestion.orchestrator
```
Files with an unchanged `content_hash` (SHA-256 of file content) are skipped.
New and modified files are re-embedded. Deleted files are removed from ChromaDB.

### Full Re-ingest
Required after changes to the embedding strategy (chunking, embed-text construction):

```powershell
# Stop backend first!
Remove-Item -Recurse -Force .\chroma_data
python -m src.ingestion.orchestrator
```

### Ingest via API (asynchronous)
```bash
curl -X POST http://localhost:8000/ingest
# → {"job_id": "abc123", "status": "queued"}

curl http://localhost:8000/ingest/status/abc123
# → {"status": "done", "documents_processed": 42, "chunks_created": 287, ...}
```

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

## Configuration

### Override Order
```
Env variables  >  .env file  >  config/settings.yaml  >  Pydantic defaults
```

### Key Env Variables
| Variable | Effect |
|----------|--------|
| `GEMINI_API_KEY` | Enable Gemini provider |
| `LLM__PROVIDER` | Provider at startup (`ollama` or `gemini`) |
| `LLM__OLLAMA__MODEL` | Override Ollama model |
| `CHROMA_MODE` | `embedded` (local) or `client` (Docker) |

---

## Common Issues

### Ollama not responding
```powershell
ollama serve          # start Ollama manually
ollama list           # check available models
ollama pull qwen3:8b  # pull model if missing
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
1. Check if re-ingest is needed (delete `chroma_data` + re-ingest)
2. Check `score_threshold` in `settings.yaml` (default: 0.5)
3. Increase `top_k` (more candidates for the reranker)
4. Disable the category filter in the sidebar

---

## Docker

```bash
# First start
docker compose up --build -d
docker compose exec ollama ollama pull qwen3:8b

# Index documents
docker compose exec api python -m src.ingestion.orchestrator

# Normal start / stop
docker compose up -d
docker compose down
```

In the Docker setup, ChromaDB runs in `client` mode — the API container accesses the
`chromadb` container via HTTP. `chroma_data` resides only in the `chromadb` container.

---

## Logs

```powershell
# Local logs
Get-Content logs\lorekeeper.log -Tail 50

# Docker logs
docker compose logs -f api
```

Log level in `settings.yaml`:
```yaml
logging:
  level: INFO   # DEBUG | INFO | WARNING | ERROR
```
