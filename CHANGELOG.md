# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **Sources page** (`ui/pages/1_Sources.py`): full source CRUD, per-source reindex, folder scanning, recategorization, and danger-zone wipe — replaces manual `sources.yaml` editing
- **Evaluation page** (`ui/pages/2_Evaluation.py`): 5-tab UI for Golden Set management, single-question retrieval preview, retrieval eval, end-to-end eval, and side-by-side result comparison
- Evaluation API routes (`/eval/*`): QA-pair CRUD, retrieval preview, async eval jobs with progress, result storage with auto-cleanup (max 3 per type)
- Source management API routes (`/sources/*`): CRUD, folder discovery, per-source ingestion, recategorization
- `config/sources.yaml` sidecar for source configuration (auto-migrated from `settings.yaml` on first startup)
- Multiple document path support for ingestion with per-source group and category mapping
- Structured logging for all evaluation endpoints and the retrieval eval core loop

### Changed
- Retrieval-tuning sliders moved from chat sidebar to Evaluation page
- `/health` endpoint now returns instantly from a background-refreshed cache (60s interval) instead of blocking on LLM/ChromaDB checks per request
- Sidebar HTTP calls (`/provider`, `/provider/gemini/status`) cached with TTL to eliminate redundant requests on every Streamlit rerun
- Evaluation page data (`/eval/qa-pairs`, `/eval/results`) cached with TTL and explicit cache invalidation on mutations
- `evaluate_retrieval.run_evaluation()` refactored: core logic extracted into `run_evaluation_with_retriever()` for reuse by both CLI and API

### Fixed
- Documentation: corrected stale values and inconsistencies across docs

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
