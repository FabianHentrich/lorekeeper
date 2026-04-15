# Configuration Reference

This document provides a comprehensive reference for configuring LoreKeeper. It details the setup for ingestion, chunking, embeddings, LLM providers, conversation settings, and how these values can be overridden using environment variables.

## Table of Contents
1. [`ingestion`](#ingestion)
   - [Sources sidecar — `config/sources.yaml`](#sources-sidecar--configsourcesyaml)
   - [PDF parsing — `ingestion.pdf`](#pdf-parsing--ingestionpdf)
2. [`chunking`](#chunking)
3. [`embeddings`](#embeddings)
4. [`vectorstore`](#vectorstore)
5. [`retrieval`](#retrieval)
6. [`llm`](#llm)
   - [`llm.ollama`](#llmollama)
   - [`llm.gemini`](#llmgemini)
7. [`conversation`](#conversation)
8. [`logging`](#logging)
9. [`api`](#api)
10. [Full Env Variable Reference](#full-env-variable-reference)

---

All settings live in `config/settings.yaml`. Env variables and the `.env` file
override YAML values (double underscore for nesting: `LLM__OLLAMA__MODEL`).

**Override order** (highest priority first):
```
Env variables  >  .env file  >  config/settings.yaml  >  Pydantic defaults
```

### Editing via the UI

A subset of these settings is editable live via the 🛠 **Settings** page
(`ui/pages/4_Settings.py`) and the `GET /config` / `PUT /config` endpoints.
Writes go to `config/settings.yaml` **and** mutate the running services in
place — no restart required.

| Section | Editable via UI | Live effect |
|---|---|---|
| `retrieval.*` (incl. `hybrid.*`, `reranking.*`) | ✅ all fields | next query |
| `llm.ollama.{temperature,top_p,max_tokens,timeout}` | ✅ | next query |
| `llm.gemini.{temperature,top_p,max_tokens,timeout}` | ✅ | next query |
| `llm.fallback_enabled` | ✅ | next query |
| `conversation.{window_size,condense_question,session_timeout_minutes}` | ✅ | next query / next GC wake |
| `chunking.*` | ✅ | **next reindex only** (UI shows warning) |
| `llm.provider` | via sidebar provider switch | hot-swap |
| `ingestion.sources` | via Sources page | requires reindex |
| prompts | via Prompts page | hot-reload |
| `embeddings.*`, `vectorstore.*`, `api.*`, `logging.*`, secrets | ❌ read-only or `.env` only | restart |

The allow-list lives in `src/config/manager.py:_EDITABLE_KEYS`. Keys outside
it are silently dropped by `ConfigManager.save_settings()`.

---

## `ingestion`

Controls which files are read.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `supported_formats` | `list[str]` | `[".md", ".pdf", ".png", ".jpg", ".webp"]` | File extensions to process |
| `exclude_patterns` | `list[str]` | see below | Glob patterns for files/folders to exclude |
| `watch_for_changes` | `bool` | `false` | Not implemented (reserved for auto-reindex) |
| `document_paths` | `list[str]` | `[]` | **Deprecated.** Replaced by `config/sources.yaml`. If set without a `sources.yaml`, the loader migrates them on the fly to sources with `group=lore` and warns. |

### Sources sidecar — `config/sources.yaml`

Sources are configured in a separate sidecar file (gitignored, user-specific).
A source is either a folder or a single file and belongs to one of the three
filter groups `lore`, `adventure`, `rules`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | yes | Stable identifier (used to scope reindex/delete and as `source_id` metadata on every chunk). Renaming = new source. |
| `path` | `str` | yes | Absolute or relative path. May point at a folder OR a single file. |
| `group` | `lore`\|`adventure`\|`rules` | yes | Default filter group for the three sidebar buttons. Can be overridden per folder in `category_map`. |
| `default_category` | `str` | no (yes for file sources) | Fallback `content_category` when no entry of `category_map` matches. |
| `category_map` | `dict[str, str\|dict]` | no | Top-level folder name → category (and optionally group). Values can be a plain string (`npc`) or a dict (`{category: story, group: adventure}`). String values inherit the source-level `group`. Case-insensitive lookup. Folder sources only. |
| `exclude_patterns` | `list[str]` | no | Additive to global `ingestion.exclude_patterns`. |

**Example:**
```yaml
# config/sources.yaml
sources:
  - id: pnp-welt
    path: C:/Users/you/Obsidian/PnP-Welt
    group: lore                          # fallback for unmapped folders
    default_category: misc
    category_map:
      NPCs: npc                          # string shorthand → inherits group: lore
      Orte: location
      Gegner: enemy
      Geschichte:                        # dict form → overrides group
        category: story
        group: adventure
      Regelwerk:
        category: rules
        group: rules
```

**Managing sources:**
- Edit `config/sources.yaml` directly, or
- Use the Streamlit page **⚙ Sources** (`ui/pages/1_Sources.py`) to add / edit / reindex / recategorize / remove sources, or
- Use the REST endpoints: `GET/PUT /sources`, `POST /sources/{id}/reindex`, `DELETE /sources/{id}`, `POST /sources/recategorize`, `POST /admin/wipe`.

**Recategorize vs. Reindex:**
- *Recategorize* — `python -m src.ingestion.recategorize` rewrites only the `group` / `content_category` / `source_id` metadata of existing chunks against the current `sources.yaml`. No re-embedding, runs in seconds.
- *Reindex* — full delete + parse + chunk + embed for one source. Required when content or `path` changes.

**Default excludes:**
```yaml
exclude_patterns:
  - ".obsidian/*"   # Obsidian configuration
  - ".trash/*"      # Obsidian trash
  - "*alt.md"       # Backup versions
  - "*(1).md"       # Duplicates
  - "*.draft.*"     # Drafts
```

### PDF parsing — `ingestion.pdf`

Controls OCR and image extraction for PDF files. Requires `rapidocr_onnxruntime`
(installed via `requirements.txt`).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ocr_enabled` | `bool` | `true` | Run OCR on text-less regions in PDFs |
| `ocr_language` | `str` | `deu` | Language code for OCR engine (ISO 639-3) |
| `ocr_dpi` | `int` | `300` | DPI for OCR rendering (higher = better quality, slower) |
| `extract_images` | `bool` | `true` | Extract embedded images as separate index entries |
| `image_format` | `str` | `png` | Image format for extraction (`png`, `jpg`) |

**Example:**
```yaml
ingestion:
  pdf:
    ocr_enabled: true
    ocr_language: "deu"
    ocr_dpi: 300
    extract_images: true
    image_format: "png"
```

**How it works:**

1. pymupdf4llm converts each PDF page to Markdown with `page_chunks=True`
2. If `ocr_enabled`, RapidOCR runs on regions without extractable text
3. Heading hierarchy is corrected using TOC data from the PDF (most reliable), with a numbered-heading regex as fallback
4. `ignore_code=True` prevents monospace text in rulebooks from being misidentified as code blocks
5. Extracted images become separate `ParsedDocument` entries with `document_type="image"`
6. If OCR fails (engine not installed, corrupt page), the parser retries without OCR automatically

> **Note:** Changes to `ingestion.pdf` require a re-ingest of affected PDFs.
> Incremental ingest handles this automatically (content hash changes).

---

## `chunking`

Controls how documents are split into chunks.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | `str` | `heading_aware` | `heading_aware` \| `recursive` \| `fixed_size` |
| `max_chunk_size` | `int` | `256` | Max chunk size in estimated tokens (`len(text) / 3.5`); applies to prose only — tables are kept atomic |
| `chunk_overlap` | `int` | `30` | Overlap in recursive prose splitting; tables never receive overlap |
| `min_chunk_size` | `int` | `20` | Chunks below this size are merged with the previous chunk **only if both belong to the same `heading_hierarchy`** — merging across heading boundaries would make the chunk's heading metadata lie about half its content |

**Strategies:**

| Strategy | Behavior | Best for |
|----------|----------|---------|
| `heading_aware` | Each Markdown/PDF heading = one chunk; Markdown tables kept atomic; oversized prose → recursive split | Obsidian vaults, structured docs, PDFs |
| `recursive` | Recursive splitting on `\n\n`, `\n`, `. ` | Prose without clear structure |
| `fixed_size` | Fixed word windows with overlap | Uniform chunks, unstructured text |

> **Table-aware chunking (heading_aware only):** Lines starting with `|` are detected as
> Markdown tables and treated as atomic units — no split mid-row, no overlap. Very large
> tables are split at row boundaries with the header repeated in each part.

> **Note:** Changes to `chunking` require a full re-ingest
> (delete `chroma_data`). See [operations.md](operations.md).

---

## `embeddings`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | `str` | `intfloat/multilingual-e5-base` | HuggingFace model ID. e5-models get automatic query/passage prefixes. |
| `device` | `str` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `batch_size` | `int` | `64` | Batch size during ingest embedding |
| `normalize` | `bool` | `true` | L2 normalization of vectors (required for cosine similarity) |

The model is preloaded at server start (`warmup` in `lifespan`), not on the first query.

---

## `vectorstore`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | `str` | `embedded` | `embedded` (local) \| `client` (Docker/remote) |
| `persist_directory` | `str` | `./chroma_data` | Storage location in embedded mode |
| `chroma_host` | `str` | `chromadb` | Host in client mode |
| `chroma_port` | `int` | `8000` | Port in client mode |
| `collection_name` | `str` | `lorekeeper` | ChromaDB collection |
| `distance_metric` | `str` | `cosine` | `cosine` \| `l2` \| `ip` |

**Env override for Docker:**
```bash
CHROMA_MODE=client
```

> **Warning:** `persist_directory` and `chroma_host/port` are mutually exclusive.
> Switch exclusively via `CHROMA_MODE`.

---

## `retrieval`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `top_k` | `int` | `15` | Number of candidates from ChromaDB (before reranking) |
| `score_threshold` | `float` | `0.5` | Minimum cosine similarity. 0.5 ≈ "similar"; 0.3 returns near-orthogonal noise |
| `reranking.enabled` | `bool` | `true` | Enable cross-encoder reranking |
| `reranking.model` | `str` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker model (multilingual) |
| `reranking.top_k_rerank` | `int` | `8` | Final chunk count after reranking → LLM context |
| `reranking.max_per_source` | `int` | `3` | Soft cap on chunks coming from a single source file (0 = unlimited). Prevents one dense document from filling all `top_k_rerank` slots and crowding out other relevant sources. Two-pass: first fills with diversity preference, then backfills cap-blocked chunks if `top_k_rerank` would otherwise not be reached — so the cap never silently returns fewer chunks than requested. |
| `hybrid.enabled` | `bool` | `false` | Enable hybrid retrieval: vector + BM25 run in parallel, results fused via Reciprocal Rank Fusion before reranking. Per-request override via the `hybrid_search` field on `QueryRequest` |
| `hybrid.bm25_weight` | `float` | `0.3` | BM25 share in RRF (0.0 = pure vector, 1.0 = pure BM25) |
| `hybrid.bm25_top_k` | `int` | `15` | How many BM25 candidates to fetch before fusion |

**Rules of thumb:**
- `top_k` should be at least twice `top_k_rerank`
- Only set `score_threshold > 0` when reranking is disabled; otherwise the reranker handles quality control
- `score_threshold` is always evaluated on the **cosine** scores before RRF fusion — RRF scores max out around `1/(60+1) ≈ 0.016` and would be entirely wiped by a cosine-scale threshold

---

## `llm`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | `str` | `ollama` | Active provider at startup: `ollama` \| `gemini` |
| `fallback_provider` | `str\|null` | `gemini` | Fallback on provider failure |
| `fallback_enabled` | `bool` | `false` | Enable fallback |

### `llm.ollama`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_url` | `str` | `http://localhost:11434` | Ollama server URL |
| `model` | `str` | `qwen3:8b` | Model name (must be available via `ollama pull`) |
| `temperature` | `float` | `0.3` | Creativity (0 = deterministic, 1 = creative) |
| `top_p` | `float` | `0.9` | Nucleus sampling |
| `max_tokens` | `int` | `2048` | Maximum output tokens |
| `timeout` | `int` | `300` | Request timeout in seconds |

**Env override:**
```bash
LLM__OLLAMA__MODEL=qwen3:14b
LLM__PROVIDER=ollama
```

### `llm.gemini`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | `str` | `gemini-2.5-flash` | Gemini model ID |
| `api_key_env` | `str` | `GEMINI_API_KEY` | Name of the env variable containing the API key |
| `temperature` | `float` | `0.3` | Creativity |
| `top_p` | `float` | `0.9` | Nucleus sampling |
| `max_tokens` | `int` | `1024` | Maximum output tokens |
| `timeout` | `int` | `30` | Request timeout in seconds |

**Env override:**
```bash
GEMINI_API_KEY=your-key-here   # in .env
LLM__PROVIDER=gemini
```

**Runtime API key (no .env required):**

The Gemini API key can also be supplied at runtime from the Streamlit sidebar
("LLM Provider" section). It is held in process memory only — never written to
disk — and is lost on backend restart. Resolution order at provider creation:

1. Runtime override (set via `POST /provider/gemini/key`)
2. `os.environ[api_key_env]`
3. `.env` file

Endpoints:
- `GET /provider/gemini/status` → `{has_key, source: env|runtime|none}`. Never returns the key itself.
- `POST /provider/gemini/key` body `{"api_key": "..."}` — stores the key. If Gemini is the active provider, the provider + generator are rebuilt immediately so the new key applies. Pass `null` or an empty string to clear the override.

This makes "start program → enter key in UI → use Gemini" possible without touching `.env`.

---

## `conversation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `window_size` | `int` | `8` | Maximum number of messages in the context window (user + assistant) |
| `max_context_tokens` | `int` | `4096` | Maximum tokens in conversation context |
| `condense_question` | `bool` | `true` | Reformulate follow-up questions into standalone questions |
| `condense_model` | `str\|null` | `null` | Separate Ollama model for condensing (null = use primary) |
| `session_timeout_minutes` | `int` | `60` | Delete inactive sessions after N minutes |
| `session_gc_interval_seconds` | `int` | `300` | How often the GC loop runs (5 minutes) |

---

## `logging`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | `str` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `file` | `str\|null` | `logs/lorekeeper.log` | Log file; `null` = console only |
| `max_bytes` | `int` | `10485760` | Max file size (10 MB) before rotation |
| `backup_count` | `int` | `5` | Number of rotated log files |
| `suppress` | `list[str]` | see YAML | Set third-party loggers to WARNING |

---

## `api`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | `str` | `0.0.0.0` | Bind address |
| `port` | `int` | `8000` | Port (overridden by uvicorn if started differently) |
| `reload` | `bool` | `true` | Hot reload (development only) |

---

## Full Env Variable Reference

| Env Variable | YAML Equivalent | Example |
|-------------|----------------|---------|
| `GEMINI_API_KEY` | – (secret, env only) | `AIza...` |
| `LLM__PROVIDER` | `llm.provider` | `gemini` |
| `LLM__OLLAMA__MODEL` | `llm.ollama.model` | `qwen3:14b` |
| `LLM__OLLAMA__BASE_URL` | `llm.ollama.base_url` | `http://remote:11434` |
| `LLM__GEMINI__MODEL` | `llm.gemini.model` | `gemini-2.0-flash` |
| `CHROMA_MODE` | `vectorstore.mode` | `client` |
| `VECTORSTORE__CHROMA_HOST` | `vectorstore.chroma_host` | `chromadb` |
| `RETRIEVAL__TOP_K` | `retrieval.top_k` | `15` |
| `CONVERSATION__WINDOW_SIZE` | `conversation.window_size` | `12` |
| `LOGGING__LEVEL` | `logging.level` | `DEBUG` |
