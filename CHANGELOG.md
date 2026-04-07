# Changelog

All notable changes to LoreKeeper are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- API-layer integration tests (`tests/integration/test_api_routes.py`) covering
  `/query`, `/query/stream` (including SSE `done`-event shape and `session_id`
  propagation), `/health`, `/sessions`, `/stats`, `/provider`, and `/ingest`.
- DEBUG-level retrieval tracing in `Retriever` — logs query, top-k candidates
  with pre-threshold scores, score-threshold pruning, and rerank scores with
  bi-encoder/cross-encoder comparison. Helps diagnose bad answers.
- `CONTRIBUTING.md` with development setup, coding guidelines, and PR process.
- `CHANGELOG.md` (this file).
- `LICENSE` (MIT).

## [0.2.0] — 2026-04-07

Initial public release.

### Features
- FastAPI backend + Streamlit UI for RAG-based Q&A over an Obsidian vault.
- Dual LLM provider support: **Ollama** (local) and **Google Gemini** (cloud)
  with runtime switching via `/provider` endpoint.
- Two-stage retrieval: bi-encoder (`intfloat/multilingual-e5-base`) →
  cross-encoder reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
- Obsidian-native Markdown parsing: wikilinks, callouts, image embeds, tags,
  and YAML frontmatter aliases are all extracted into searchable metadata.
- Semantic source filtering in the UI (Lore / Abenteuer / Regelwerk) —
  restricts retrieval to specific content categories so the LLM cannot mix,
  for example, a rulebook class with a lore NPC of the same name.
- SSE token streaming with session_id and source citations in the terminal
  `done` event; Streamlit renders via `st.write_stream()`.
- Multi-turn conversations with sliding-window memory and background session
  garbage collection (asyncio task in FastAPI lifespan).
- Question condensing for follow-up queries via a second Ollama model.
- Content-hash-based (SHA-256) incremental re-ingestion — unchanged files are
  skipped; Windows timestamp unreliability is sidestepped.
- Heading-aware chunking with atomic table handling.
- Identity-layer embedding: filename stem + aliases prepended to each chunk
  before embedding, so entity-specific queries match even tables/lists that
  don't self-reference the entity by name.
- Two ChromaDB modes: `embedded` (local dev) and `client` (Docker/HTTP).
- Rotating file logs with third-party noise suppression.
- Docker Compose stack: API, ChromaDB, Ollama (with AMD GPU support), and UI.

### Tests
- 124 unit + integration tests covering ingestion, retrieval, generation,
  conversation, prompts, config, the full RAG pipeline, the HTTP API layer,
  and the FastAPI lifespan.

[Unreleased]: https://github.com/fabianhentrich/LoreKeeper/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/fabianhentrich/LoreKeeper/releases/tag/v0.2.0
