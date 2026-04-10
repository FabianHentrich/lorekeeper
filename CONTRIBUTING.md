# Contributing to LoreKeeper

Thanks for your interest in improving LoreKeeper! This document describes how
to set up a development environment, the expectations for code changes, and
how to submit a pull request.

---

## Development Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) (for local LLM) **or** a Gemini API key
- ~5 GB disk space for models (embedding + reranker + LLM)

### Local (venv)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY if using Gemini
```

### Running the stack

```bash
# Backend
uvicorn src.main:app --reload --port 8000

# Frontend (separate terminal)
streamlit run ui/LoreKeeper.py
```

### Docker

```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen3:8b
```

---

## Project Layout

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full system overview and
[`CLAUDE.md`](./CLAUDE.md) for the coding conventions used throughout the
project.

Key directories:

- `src/api/` — FastAPI routes and Pydantic schemas
- `src/ingestion/` — Parsers, chunking, ingestion orchestrator
- `src/retrieval/` — Embeddings, ChromaDB wrapper, retriever, reranker
- `src/generation/` — LLM providers and generator
- `src/conversation/` — Session management
- `ui/` — Streamlit frontend
- `tests/unit/` — Fast unit tests (mocked externals)
- `tests/integration/` — RAG pipeline + API layer tests

---

## Coding Guidelines

- **Read before you write.** Fully read any file you're modifying.
- **Minimal changes.** Change only what the task requires — no drive-by
  refactors, cleanup commits, or style passes in unrelated code.
- **Follow the provider interface.** New LLM backends implement
  [`BaseLLMProvider`](./src/generation/providers/base.py) (`generate`,
  `generate_stream`, `health_check`) and register in `ProviderFactory`.
- **Respect async boundaries.** Any CPU-bound call (especially
  `sentence_transformers` `encode()` / `predict()`) **must** be offloaded via
  `fastapi.concurrency.run_in_threadpool`. Blocking the event loop is a bug.
- **ChromaDB mode.** Local dev uses `embedded` (on-disk), Docker uses
  `client` (HTTP). Never access `chroma_data/` directly when the client
  container is running — it causes file locks.
- **Tests are required.** Any logic added to `src/` (except thin wiring in
  `src/api/routes.py`) needs a unit test. API route behavior should be
  covered in `tests/integration/test_api_routes.py`.
- **No secrets in config.** Keys come from `.env` / environment variables,
  never from `config/settings.yaml`.

See [`CLAUDE.md`](./CLAUDE.md) for the complete list of conventions.

---

## Running the Test Suite

```bash
# Full suite
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Single file
pytest tests/unit/retrieval/test_embeddings.py

# Only the fast unit tests
pytest tests/unit
```

All tests mock external services (Ollama, Gemini, ChromaDB for unit tests).
Integration tests use `chromadb.EphemeralClient` (in-memory). **No real LLM
calls in the test suite — ever.**

---

## Pull Request Process

1. Fork and create a feature branch (`feat/short-name` or `fix/short-name`).
2. Make your changes with accompanying tests.
3. Run the full test suite locally — everything must pass.
4. Update `CHANGELOG.md` under the `## [Unreleased]` section.
5. If you changed behavior documented in `README.md`, `ARCHITECTURE.md`, or
   `docs/`, update those too.
6. Open a PR with a clear description: **what** changed and **why**.

PRs are reviewed on:

- Correctness and test coverage
- Adherence to the async/threading rules above
- Whether documentation reflects the new behavior
- Minimalism — does the diff stay focused on the stated goal?

---

## Reporting Bugs

Open an issue with:

- What you expected to happen
- What actually happened (stack trace, log output)
- Steps to reproduce (ideally a minimal example)
- Your environment (OS, Python version, provider: Ollama/Gemini, Docker or local)

Log output lives in `logs/lorekeeper.log` by default.

---

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](./LICENSE).
