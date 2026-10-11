# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Per-query stage timings: `/query` and the `/query/stream` `done` event return `timings` (condense, retrieval with embed/search/bm25/rerank/expand stages, time to first token, total, chunk count, prompt size); one `Query timings:` log line per request; the chat shows them under each answer
- **Settings page** (`ui/pages/4_Settings.py`): central tab-based UI for editing retrieval, LLM, conversation and chunking defaults. Saves diff-only to `settings.yaml` and mutates the running services in place
- `GET /config` and `PUT /config` endpoints backed by `ConfigManager.save_settings()` with an allow-list (`src/config/manager.py:_EDITABLE_KEYS`) that keeps secrets and infrastructure knobs off the UI surface
- Read-only display of `embeddings.*` and `vectorstore.*` in the Settings page (changing these still requires `.env` + restart)
- **Sources page** (`ui/pages/1_Sources.py`): full source CRUD, per-source reindex, folder scanning, recategorization, and danger-zone wipe — replaces manual `sources.yaml` editing
- **Evaluation page** (`ui/pages/2_Evaluation.py`): 5-tab UI for Golden Set management, single-question retrieval preview, retrieval eval, end-to-end eval, and side-by-side result comparison
- Evaluation API routes (`/eval/*`): QA-pair CRUD, retrieval preview, async eval jobs with progress, result storage with auto-cleanup (max 3 per type)
- Source management API routes (`/sources/*`): CRUD, folder discovery, per-source ingestion, recategorization
- `config/sources.yaml` sidecar for source configuration (auto-migrated from `settings.yaml` on first startup)
- Multiple document path support for ingestion with per-source group and category mapping
- Structured logging for all evaluation endpoints and the retrieval eval core loop
- **"⚡ Nur Änderungen indizieren"** button on the Sources page: incremental ingest that skips files with an unchanged content hash (next to the full reindex)
- `documents_skipped` in the ingest job status; the Sources page reports indexed vs. skipped documents
- Integration tests for all `/sources/*` endpoints and the ingest job lifecycle, plus an end-to-end test (ingest a vault over HTTP → `/query/stream` → incremental re-ingest) with real embeddings and in-memory ChromaDB
- ruff and mypy configuration in `pyproject.toml`, run as part of `pytest` via `tests/test_code_quality.py`

### Changed
- Qwen3 thinking is switched off via `reasoning_effort: "none"` instead of the `/no_think` prompt suffix, which current Ollama ignores — the model was silently reasoning for ~7 s before every answer. Reasoning that still arrives in Ollama's `reasoning` field is counted as `tokens_thinking`
- PDF parser skips table-of-contents sections (dot leaders or an "Inhaltsverzeichnis" heading) — their chapter lists matched almost any question and crowded out the real rule text. Requires a re-index of PDF sources
- PDF parser rejoins words hyphenated across line breaks inside table cells ("Wir-<br>kungsdauer" → "Wirkungsdauer"); body text was already dehyphenated by pymupdf4llm. Requires a re-index of PDF sources
- Retrieval eval: optional `expected_section` per golden question checks that a retrieved chunk lies in the expected numbered rulebook section; report gains `section_hit_rate` / `section_misses`, the Evaluation page shows it and the Golden-Set editor has an "Abschnitt" column
- Follow-up rewriting (condense) only runs when the question depends on the chat history (`needs_condense()` heuristic); standalone questions are retrieved verbatim instead of being rewritten from the previous answer
- `/query` and the `/query/stream` `done` event return `search_query` (the question actually used for retrieval); the chat UI shows it as „🔎 Gesucht nach: …“ when it was rewritten, and the condensed question is logged in full
- `OllamaProvider` reads `temperature`/`top_p`/`max_tokens` from `self.config` on each call instead of caching them at init, so runtime edits via the Settings page propagate without rebuilding the provider
- Retrieval-tuning sliders moved from chat sidebar to Evaluation page (per-run overrides) and Settings page (defaults)
- `/health` endpoint now returns instantly from a background-refreshed cache (60s interval) instead of blocking on LLM/ChromaDB checks per request
- Sidebar HTTP calls (`/provider`, `/provider/gemini/status`) cached with TTL to eliminate redundant requests on every Streamlit rerun
- Evaluation page data (`/eval/qa-pairs`, `/eval/results`) cached with TTL and explicit cache invalidation on mutations
- `evaluate_retrieval.run_evaluation()` refactored: core logic extracted into `run_evaluation_with_retriever()` for reuse by both CLI and API
- Per-source routes (`/sources/{id}/folders`, `/sources/{id}/reindex`, `DELETE /sources/{id}`) accept ids containing `/`; the UI URL-encodes ids
- `PUT /sources` rejects new ids that are empty, `.`, `..` or have surrounding whitespace (already configured ids are never rejected)
- Sources page shows the backend's error detail instead of a generic HTTP status
- `BaseLLMProvider.generate_stream` is declared as a plain `def` returning an `AsyncGenerator`, with `system_prompt` matching the Ollama/Gemini signatures
- `BaseLLMProvider.model` no longer defaults to `"unknown"` — provider implementations must set `self.model` in `__init__`

### Fixed
- Documentation: corrected stale values and inconsistencies across docs
- Ingest stats: `chunks_deleted` counted orphaned files instead of deleted chunks
- Sources page: chunk total stayed stale after an ingest; status polling looped forever when the backend was unreachable
- Source ids with `#`, `&`, `?` or spaces broke reindex, delete and folder lookup
- Failed scans and missing source paths were shown as "no subfolders"
- `/query/stream` reported the primary provider's model in `model_used` even when the fallback provider answered
- Sources page: chunk total stayed stale after deleting a source or wiping the collection; success messages for save/delete/add were wiped by the immediate page reload
- Console logging raised a logging error for every line containing `→` when stdout was redirected on Windows (cp1252)

### Removed
- Docker setup (`Dockerfile`, `Dockerfile.ui`, `docker-compose.yaml`, `.dockerignore`, `requirements-ui.txt`) — LoreKeeper runs Windows-only with embedded ChromaDB

---

## [0.2.0] — 2026-04-10

### Added
- Per-source soft diversity cap (`max_per_source`) with two-pass backfill in the retriever
- Token usage tracking (per-message and per-session) for both Ollama and Gemini
- Session-tokens metric in the UI header

## [0.1.0] — 2026-04-09

### Added
- Retrieval-tuning sliders in the Streamlit sidebar (Top-K, Final Chunks, Soft-Cap)

## [0.0.1] — 2026-04-08

### Added
- Initial project structure: FastAPI backend, Streamlit UI, ChromaDB vectorstore
- Obsidian-native Markdown parser (wikilinks, callouts, tags, aliases)
- Heading-aware chunking with atomic tables
- Multilingual E5 bi-encoder + mMiniLMv2 cross-encoder reranking
- Ollama and Gemini LLM providers with runtime switching
- SSE streaming with source citations
- Multi-turn conversation with session GC
- Source management UI page
- Incremental ingestion with SHA-256 content hashing
- Evaluation framework with Golden Set (46 questions)
- Docker Compose setup (API + ChromaDB + Ollama + UI)
