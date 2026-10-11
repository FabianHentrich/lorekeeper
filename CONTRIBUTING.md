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
- `tests/ui/` — Streamlit pages rendered with `AppTest` against a fake backend
- `tests/perf/` — timing-based performance checks (`pytest -m perf`, see below)

---

## Coding Guidelines

- **Read before you write.** Fully read any file you're modifying.
- **Minimal changes.** Change only what the task requires — no drive-by
  refactors, cleanup commits, or style passes in unrelated code.
- **Follow the provider interface.** New LLM backends implement
  [`BaseLLMProvider`](./src/generation/providers/base.py) (`generate`,
  `generate_stream`, `health_check`), set `self.model` in `__init__` (there is
  no default) and register in `ProviderFactory`.
- **Respect async boundaries.** Any CPU-bound call (especially
  `sentence_transformers` `encode()` / `predict()`) **must** be offloaded via
  `fastapi.concurrency.run_in_threadpool`. Blocking the event loop is a bug.
- **ChromaDB mode.** LoreKeeper runs Windows-only with `embedded` (on-disk)
  ChromaDB. Don't run the ingestion CLI while the backend holds `chroma_data/`
  — it causes file locks.
- **Tests are required.** Any logic added to `src/` (except thin wiring in
  `src/api/routes.py`) needs a unit test. API route behavior should be
  covered in `tests/integration/` (`test_api_routes.py` for query/session/config,
  `test_sources_routes.py` for `/sources/*`, the ingest job lifecycle and
  `/admin/wipe`).
- **Lint and types.** `ruff check .` and `mypy` must be clean — both run as
  part of `pytest` (`tests/test_code_quality.py`). Config lives in
  `pyproject.toml`.
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

# Lint + type check on their own (also part of the full suite)
ruff check .
mypy
```

All tests mock external services (Ollama, Gemini, ChromaDB for unit tests).
Integration tests use `chromadb.EphemeralClient` (in-memory). **No real LLM
calls in the test suite — ever.**

`tests/integration/test_e2e_ingest_query.py` is the end-to-end check: it ingests
a small vault over HTTP and queries it via `/query/stream`, with real parsing,
embeddings (small MiniLM model) and ChromaDB, and a fake LLM. It loads the
embedding model, so it takes noticeably longer than the rest.

### Security checks (player mode, admin login)

The player mode must never hand game-master content to players. Three levels:

- **Retrieval, deterministic** — `tests/integration/test_player_mode_security.py`:
  a rules source (released) and a game-master source with secrets deliberately
  phrased to match rule questions, real embeddings + in-memory ChromaDB, a fake
  cross-encoder that ranks game-master text first, cap backfill, section
  expansion without top-rank limit, vector and hybrid search, condense with a
  bait follow-up. Neither chunks, sources nor the LLM prompt may contain a secret.
- **UI and session, deterministic** — `tests/ui/test_player_mode_security.py`:
  tampered/leftover session state, every follow-up filtered, fail closed when
  `/sidebar-state` fails or lacks the list, revoking a source takes effect on the
  next question, and the admin-login brute-force protection.
- **Live, `pytest -m live`** (excluded by default) — `tests/live/test_player_mode_live.py`
  against the running backend and the real index: every golden-set question about
  vault knowledge, vector and hybrid, must retrieve only released sources; five
  game-master secrets asked through the full answer path (LLM) must not appear.
  Takes ~7 minutes (251 checks).

The deterministic tests were mutation-checked: breaking the BM25 filter, keying
section expansion by file name only, dropping the vector filter, merging game-master
filters into the player filter, failing open, old cache clearing and skipping the
lockout check each make them fail.

### Performance checks

Two kinds, so a change that makes LoreKeeper noticeably less efficient shows up
right away:

- **Deterministic — part of the normal `pytest` run.** They count work, not time,
  so they are hardware-independent and fail reliably:
  - `tests/ui/test_backend_calls.py` — backend requests per page render and per
    rerun (every click reruns the page script, so an uncached request costs its
    full latency on every click), requests per chat message, and that local
    service URLs use `127.0.0.1` instead of `localhost` (on Windows `localhost`
    resolves to IPv6 first and costs ~2 s per new connection). Budgets live in
    `BUDGETS`; raise one in the same change if an extra request is intended.
  - `tests/integration/test_api_routes.py::test_llm_calls_per_request` — one LLM
    call per answer, a second one only for a real follow-up (condense).
- **Timing-based — `pytest -m perf`, excluded by default.** Compared against
  `tests/perf/baseline.json` with a tolerance (×1.5 plus a small absolute slack),
  because absolute times depend on the machine:
  - `tests/perf/test_ui_render.py` — render time per page (fake backend).
  - `tests/perf/test_golden_retrieval.py` — runs the golden set through the
    **running** backend (`127.0.0.1:8000`; skipped if none) and checks p50/p95
    retrieval latency, chunks per question (avg/max — catches context bloat) and
    that the section hit rate does not drop. Takes a few minutes.

```bash
pytest -m perf                          # compare against the baseline
PERF_UPDATE_BASELINE=1 pytest -m perf   # accept the current values (new hardware, intended trade-off)
```

LLM answer times are not tested automatically (model- and hardware-dependent);
use the per-query `timings` (log line `Query timings:` / caption in the chat).

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
- Your environment (OS, Python version, provider: Ollama/Gemini)

Log output lives in `logs/lorekeeper.log` by default.

---

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](./LICENSE).
