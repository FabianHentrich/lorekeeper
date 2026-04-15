# UI / UX

Das Benutzerinterface von LoreKeeper ist vollständig in Streamlit geschrieben. Dieses Dokument beschreibt den Aufbau der Seiten, die UX-Konzepte, die Integration mit der FastAPI-Backend-Schicht und den Umgang mit dem Session-State.

## Inhaltsverzeichnis
1. [Streamlit Chat Interface](#streamlit-chat-interface)
2. [Sidebar Elements](#sidebar-elements)
3. [Header](#header)
4. [Chat Area](#chat-area)
5. [Sources Page (1_Sources.py)](#sources-page-uipages1_sourcespy)
6. [Performance Characteristics](#performance-characteristics)
7. [Prompts Page (3_Prompts.py)](#prompts-page-uipages3_promptspy)
8. [Evaluation Page (2_Evaluation.py)](#evaluation-page-uipages2_evaluationpy)
9. [Settings Page (4_Settings.py)](#settings-page-uipages4_settingspy)
10. [Session State Overview](#session-state-overview)

---

## Streamlit Chat Interface

```mermaid
flowchart LR
    subgraph SIDEBAR["📋 Sidebar"]
        direction TB
        S1["🔌 API URL<br/><i>http://localhost:8000</i>"]
        S2["🔀 LLM Provider<br/>Ollama ↔ Gemini · live switch<br/><i>Active: qwen3:8b</i>"]
        S3["💚 Status<br/>🟢 healthy · ChromaDB ✅ · LLM ✅<br/><i>cached 30s</i>"]
        S4["🎯 Quellen-Filter<br/>☑ 🗺️ Lore<br/>☑ 📖 Abenteuer<br/>☑ 📋 Regelwerk"]
        S5["🔍 Hybrid Search<br/><i>BM25 + Vektor · Per-Session-Override</i>"]
        S6["→ 🛠 Settings<br/><i>Retrieval · LLM · Conv · Chunking</i>"]
        S7["🗑️ Neue Session"]
        S1 --- S2 --- S3 --- S4 --- S5 --- S6 --- S7
    end

    subgraph CHAT["💬 Chat Area"]
        direction TB
        C1["📜 <b>LoreKeeper</b> · <i>Frag deine Welt.</i><br/>┃ Session-Tokens: <b>4.812</b>"]
        C2["[user] Was ist Arkenfeld?"]
        C3["[assistant] Arkenfeld ist eine<br/>Handelsstadt im Tiefland... ▌<br/>⬇ 1.842 in · ⬆ 312 out"]
        C4["📎 Quellen ▼<br/>📄 Locations/Arkenfeld.md · Best 0.82 · 2 Chunks<br/>↳ Overview (0.82)<br/>↳ History (0.71)"]
        C5["[ Stelle eine Frage über deine Welt... ]"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    SIDEBAR ~~~ CHAT

    classDef side fill:#ede9fe,stroke:#7c3aed,color:#4c1d95
    classDef chat fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    class S1,S2,S3,S4,S5,S6,S7 side
    class C1,C2,C3,C4,C5 chat
```

---

## Sidebar Elements

### API URL
Connection target for the backend. Default: `http://localhost:8000`.
Can be changed to a remote server without restarting.

### LLM Provider
Dropdown for switching between Ollama and Gemini at runtime.
- Switch fails → dropdown reverts + error message
- Successful switch → confirmation message

#### Gemini API Key Input

Directly below the provider selector. Calls `GET /provider/gemini/status` to
ask the backend whether a key is currently available — the endpoint never
returns the key itself, only `{has_key, source: env|runtime|none}`.

| Backend state | Sidebar shows |
|---|---|
| Key from `os.environ` / `.env` | `🔑 Gemini-Key: ✅ (Umgebungsvariable)` plus an **Override** expander |
| Key set via UI in this session | `🔑 Gemini-Key: ✅ (UI-Eingabe)` plus the same **Override** expander |
| No key | Yellow warning + password input + **Key speichern** button |

The input is `type="password"` and submits to `POST /provider/gemini/key`. The
backend stores the key in process memory only — it is **never written to disk**
and is lost on backend restart. If Gemini is the active provider, the backend
hot-rebuilds the provider + generator so the new key applies immediately;
otherwise the key just sits in memory until the user switches to Gemini.

This makes "start backend with no `.env` → paste key in UI → use Gemini" a
one-step path without touching any config files.

### Status Display
Result of the `/health` endpoint, **cached for 30 seconds** (no poll on every rerun).
- 🟢 healthy: ChromaDB + LLM reachable
- 🟡 degraded: One component unreachable
- 🔴 API unreachable: Backend down

### Hybrid Search Toggle

A single `st.checkbox("🔍 Hybrid Search (BM25 + Vektor)")` under the **Suche**
subheader. Sets the `hybrid_search` field on the next `QueryRequest` and
overrides `retrieval.hybrid.enabled` **for this session only**. Useful when the
chat uses the Settings-configured default most of the time but a given question
benefits from keyword matching (exact names, rule IDs).

Retrieval tuning sliders (top_k, top_k_rerank, max_per_source) used to live
here but moved to two dedicated places:
- **Defaults** for all queries → [Settings Page](#settings-page-uipages4_settingspy).
- **Per-run overrides** for A/B testing against the Golden Set → Evaluation Page.

The sidebar intentionally stays minimal — provider, source filter, hybrid
toggle, session reset. Everything else is one click away on the Settings page.

### Source Filter (Lore / Adventure / Rules)
Three checkboxes that restrict the vector search by the `group` metadata field
attached to every chunk during ingestion. The mapping is **direct** (no
hardcoded category lists in the UI):

| Checkbox | `group` value |
|---|---|
| 🗺️ **Lore** | `lore` |
| 📖 **Abenteuer** | `adventure` |
| 📋 **Regelwerk** | `rules` |

A chunk's `group` is determined by which **source** it was ingested from
(see `config/sources.yaml`, documented in [docs/configuration.md](configuration.md)).
This is the structural fix for an earlier bug where the filter had to map
to a hand-maintained list of `content_category` values — a single root-level
file (the rulebook PDF) silently fell through the mapping and was unreachable.

The filter solves a real disambiguation problem: a query like *"What can the
time mage do?"* could match both the rulebook class **and** an NPC named
*Arkenfeld the Time Mage*. Unchecking 🗺️ Lore restricts retrieval to the
rulebook chunks at the vectorstore level, before the LLM ever sees them.

**State semantics:**

| Selection | Filter sent to backend |
|---|---|
| All three checked | `None` (no filter) |
| Subset checked | `{"group": {"$in": [...selected groups...]}}` |
| Nothing checked | Request blocked at chat input with `st.error` (no API call made) |

When a subset is active, the sidebar shows a "🔍 Suche eingeschränkt auf: ..."
caption listing the selected groups so the filter state stays visible während
the conversation.

The retriever combines this with the hard-coded `document_type != "image"`
filter into a ChromaDB `$and` query (`src/retrieval/retriever.py`).

### Sources Link
A caption pointing the user to the **⚙ Sources** page for chunk statistics,
re-indexing (global and per-source), and source management. The chunk count
metric and the re-index button have been moved from the sidebar to the Sources
page to keep the sidebar focused on query-time settings.

### Settings Link
A caption pointing to the **🛠 Settings** page, where all runtime-tunable
parameters (retrieval, LLM, conversation, chunking) are editable in one place.
See [Settings Page](#settings-page-uipages4_settingspy) below.

### New Session
Clears `st.session_state.messages` and `session_id` — the next question starts
without conversation history.

---

## Header

The page header is split into two columns:

| Left | Right |
|---|---|
| `📜 LoreKeeper` title + `Frag deine Welt.` caption | **Session-Tokens metric** — total tokens consumed in the current session |

The metric on the right shows the cumulative `tokens_in + tokens_out + tokens_thinking`
of the active session, formatted with thousand separators. Hovering reveals
the breakdown via tooltip (`In: … · Out: … · Thinking: …`). The counter is
reset by the **🗑️ Neue Session** sidebar button.

---

## Chat Area

### Message Rendering
- Past messages are rendered from `st.session_state.messages`
- Each assistant message shows a token-usage caption directly below the
  answer text (`⬇ N in · ⬆ N out · 🧠 N think`, the thinking part only
  appears if non-zero)
- Sources are displayed as a collapsible `st.expander("📎 Quellen")` and
  **grouped by file** (see below)

### Streaming

Token-by-token via Server-Sent Events. A blinking `▌` cursor is appended to
the placeholder until the `done` event arrives, at which point the sources
expander is rendered and `session_id` is stored for follow-up questions.

```mermaid
sequenceDiagram
    participant UI as 🖥️ Streamlit UI
    participant API as ⚡ FastAPI<br/>/query/stream
    participant RAG as 🔍 Retrieval
    participant LLM as 🧠 LLM Provider

    UI->>API: POST /query/stream<br/>{question, session_id, metadata_filters}
    API->>RAG: retrieve(query, filters)
    RAG-->>API: chunks
    API->>LLM: generate_stream(prompt)

    loop für jedes Token
        LLM-->>API: token
        API-->>UI: data: {"type":"token","content":"..."}
        Note over UI: append to placeholder + ▌
    end

    API-->>UI: data: {"type":"done","session_id":"...",<br/>"sources":[...],"model_used":"...",<br/>"usage":{...},"session_usage":{...}}
    Note over UI: render sources (grouped)<br/>render token caption<br/>update session-tokens metric<br/>store session_id
```

The `done` event carries two usage payloads:

| Field | Meaning |
|---|---|
| `usage` | Tokens consumed by **this single request** (`tokens_in`, `tokens_out`, `tokens_thinking`) |
| `session_usage` | Cumulative session totals after this request, used to refresh the header metric |

### Source Display (grouped by file)

Sources are **grouped per document** in the expander, so multiple chunks from
the same file appear under one heading instead of looking like duplicates:

```
📄 Locations/Arkenfeld.md — Best Score: 0.82 · 2 Chunks
    ↳ Arkenfeld > Overview (Score: 0.82)
       Arkenfeld is a mid-sized trading city...
    ↳ Arkenfeld > History (Score: 0.71)
       Founded during the Salt Wars...
```

| Document Type | Rendering |
|---|---|
| Markdown / PDF (single chunk) | `📄 [Filename](file:///...) — Best Score: 0.82` plus chunk preview |
| Markdown / PDF (multiple chunks) | Same header with `· N Chunks` suffix; each chunk listed indented as `↳ Heading (Score)` + preview |
| Image | `st.image(source_path)` with filename as caption (rendered separately, before grouped docs) |

Links open the original file locally (e.g. in Obsidian if `.md` is associated with it).
If `source_path` does not exist, a warning is shown.

### Token Display

| Location | Source | Format |
|---|---|---|
| Below each assistant message | `usage` field of the `done` event, persisted in `messages[*].usage` | `⬇ N in · ⬆ N out · 🧠 N think` (thinking only when > 0) |
| Header metric (top right) | `session_usage` field of the `done` event, mirrored into `st.session_state.session_usage` | `Session-Tokens` metric with thousand separators and tooltip breakdown |

`tokens_thinking` is populated by Gemini 2.5 (`thoughts_token_count`) when
thinking is enabled. For Ollama with Qwen3, `/no_think` is set, so the value
is always `0` and the icon is hidden.

---

## Sources Page (`ui/pages/1_Sources.py`)

A dedicated Streamlit page (auto-discovered via the `pages/` directory) for
managing the ingestion sources defined in `config/sources.yaml` without ever
opening a YAML file. Reachable via the page selector at the top of the
sidebar.

### Sections

#### 1. Indizierte Chunks (gesamt) + Alle Dokumente neu indizieren

| Backend call | Notes |
|---|---|
| `GET /stats`, `POST /ingest` | Global chunk count and full re-ingest trigger (all sources). Moved here from the chat sidebar so all indexing concerns live on one page. |

All ingest operations (global and per-source) use **live progress polling**:
`_poll_ingest_job()` polls `GET /ingest/status/{job_id}` every 1.5 seconds and
displays the current state inside a `st.status` widget — documents processed,
chunks created/updated/deleted, and final duration. Errors from the ingest job
are shown as `st.warning` items.

#### 2. Konfigurierte Quellen

| Backend call | Notes |
|---|---|
| `GET /sources` (read), `PUT /sources` (save) | Editable `st.data_editor` table with columns *id, path, type, group, default_category, category_map*. |

- `id` and `path` are read-only in-line; `type` is auto-derived (`file` / `folder` / `missing`) from `Path.is_file/exists`.
- `category_map` is edited as compact text: `key→category` (inherits source group) or `key→category:group` (overrides group). Example: `NPCs→npc, Geschichte→story:adventure`. Parsed back into string or dict entries on save.
- **💾 Änderungen speichern** saves the edited table via `PUT /sources`. Fields the editor doesn't expose (`exclude_patterns`) are preserved from the original.
- **🔄 Neu laden** clears the cache and reruns to pick up external changes.

#### 3. Aktionen pro Quelle

Each source gets an expander (`📁 {id} — {path}`) with two sections:

**Action buttons** (three columns):

| Button | Backend call | Notes |
|---|---|---|
| 🔄 **Reindex** | `POST /sources/{id}/reindex` | Deletes + re-ingests one source. Shows live progress via `_poll_ingest_job()`. Required when content or path changed. |
| 🏷 **Recategorize (alle)** | `POST /sources/recategorize` | Rewrites only `group` / `content_category` / `source_id` metadata of existing chunks (seconds, no embedding). Applies to **all** sources, not just the current one. |
| 🗑 **Source entfernen** | `DELETE /sources/{id}` | Drops the source from the config **and** deletes its chunks. Gated behind a confirmation checkbox ("Wirklich löschen"). Shows deleted chunk count on success. |

**Ordner-Zuordnung** (folder mapping):

| Backend call | Notes |
|---|---|
| `GET /sources/{id}/folders` | Fetches the folder tree for this source and renders a second `st.data_editor` with columns *name, type, category, group*. |

The table is pre-populated from the source's current `category_map` — unmapped
folders fall back to the source-level `default_category` and `group`. The user
edits category and group per folder directly in the table, then clicks
**💾 Zuordnung speichern** which builds a new `category_map` (string entries
when group matches the source default, dict entries when it differs) and saves
all sources via `PUT /sources`. A hint reminds the user to run Recategorize
afterwards so existing chunks pick up the new values.

#### 4. Neue Source hinzufügen (scan-based workflow)

Adding a source is a multi-step process:

| Step | UI | Backend call |
|---|---|---|
| **1. Pfad eingeben und scannen** | Text input + 🔍 Scannen button | `POST /sources/scan` with `{"path": "..."}` |
| **2. Source-Einstellungen** | ID (auto-suggested from folder/file name), Default-Group dropdown, Default-Kategorie input | — |
| **3. Ordner zuordnen** (folder sources only) | `st.data_editor` with the scanned folder tree, pre-filled with the chosen defaults. User edits category/group per folder. | — |
| **4. Source hinzufügen** | ➕ button | `PUT /sources` with the appended entry |

The scan result (folder list, is-file flag) is persisted in `st.session_state`
so it survives Streamlit reruns between steps. Validation: ID must be non-empty
and unique across existing sources. On success, the scan state is cleared and
the page refreshes.

File sources skip step 3 (no subfolders to map).

#### 5. ⚠ Danger Zone

| Backend call | Notes |
|---|---|
| `POST /admin/wipe` (body `{"confirm": "DELETE"}`) | Type `DELETE` into the input as confirmation. Drops and recreates the ChromaDB collection in both embedded and client mode. |

### When to use which action

- **Pure config edit** (e.g. fix a typo in `category_map`): edit the table or folder mapping → Save → click **Recategorize**. No re-embedding.
- **Path changed / file moved / contents changed**: edit the table → Save → click **Reindex** for that source.
- **Source no longer needed**: tick the confirmation → **Source entfernen**.
- **New source**: use the scan-based workflow (step 1–4) to add and map folders, then **Reindex** the new source.
- **DB schema gone bad** (rare, e.g. after a major refactor): use the Danger Zone, then click **Alle Dokumente neu indizieren** at the top of the Sources page.

### Source identity

Every chunk carries a `source_id` metadata field equal to the source's
configured `id`. This is what makes per-source delete / reindex / recategorize
safe in a multi-source setup — collisions on `source_file` (e.g. two vaults
both containing `notes.md`) cannot cross-contaminate. Renaming the `id`
counts as creating a new source; the old chunks become orphans on the next
ingest.

---

## Performance Characteristics

| Action | Latency (typical) |
|--------|------------------|
| Sidebar rerun | <100ms (cached API calls) |
| First query after server start | +2–4s (embedding model warm, ChromaDB connected) |
| Query (Ollama qwen3:8b) | 8–30s total |
| Query (Gemini 2.5 Flash) | 3–8s total |
| Reranking (8 candidates) | ~300ms |

**Embedding model** is preloaded at server start (`warmup` in lifespan) —
the first query is therefore no slower than subsequent ones.

---

## Prompts Page (`ui/pages/3_Prompts.py`)

A dedicated Streamlit page for editing prompt templates, managing variants,
and comparing prompts side-by-side — no YAML editing required.

### Tab 1: Aktive Prompts

Four `st.text_area` widgets (height 300px) for the active templates: System,
QA, Condense, No-Context. Each field shows a help tooltip listing the
available Jinja2 variables for that template.

Below each template, a **Preview** expander renders the template with sample
data (via `POST /prompts/preview`), so you can verify the output before
saving.

**💾 Aktive Prompts speichern** writes to `config/prompts.yaml` and
hot-reloads the `PromptManager` — the next query uses the updated templates
immediately.

### Tab 2: Varianten

**Save active as variant:** A compact row (Name + Description + Button) saves
a copy of the current active prompts as a named variant in `config/prompts/`.

**Saved variants list:** Each variant renders as an `st.expander` containing:
- Four editable `st.text_area` fields (height 250px) for all templates
- An editable description field
- Three action buttons:
  - **💾 Speichern** — saves edits to the variant file
  - **Aktivieren** — overwrites `prompts.yaml` with this variant and
    hot-reloads the PromptManager
  - **🗑 Löschen** — deletes the variant file

Active variants are marked with ✅ in the list header (detected via MD5 hash
comparison with `prompts.yaml`).

### Tab 3: Vergleichen

Two selectboxes choose variants (or "(Aktiv)" for the current prompts). For
each of the four templates, a side-by-side `st.columns(2)` view shows both
texts. Templates that differ are marked ⚠ unterschiedlich, identical ones ✅.

### Caching

All fetch functions use `@st.cache_data(ttl=10)` with `.clear()` after
mutations. This means the page shows fresh data within 10 seconds of external
changes, and immediately after in-page edits.

---

## Evaluation Page (`ui/pages/2_Evaluation.py`)

A dedicated Streamlit page for managing the Golden Set, testing retrieval,
running evaluations, and comparing results. Five tabs:

### Tab 1: Golden Set

Editable `st.data_editor` table with all QA pairs from `evaluation/qa_pairs.yaml`.
Columns: ID, Question, Source Type (markdown/pdf/image), Category, Expected
Sources (comma-separated), Answer Contains (keywords for E2E), Notes.

**💾 Golden Set speichern** writes via `PUT /eval/qa-pairs`. Rows without ID
or question are silently skipped.

### Tab 2: Frage testen

Single-question retrieval preview. Enter a question, optionally adjust
retrieval parameters (Top-K, Rerank, Max-per-Source via shared sliders), and
click **🔍 Testen**. Shows the returned chunks as a table (Score, Source,
Heading, Preview) with latency.

Below the results, a convenience section lets you add the tested question
directly to the Golden Set — the Expected Sources field is pre-filled with the
top 3 retrieved files.

### Tab 3: Retrieval Eval

Runs the full Golden Set through the retriever (no LLM). Configurable
retrieval parameters via the same shared sliders. Starts as a background job
(`POST /eval/run` with `eval_type: retrieval`), polled via
`GET /eval/status/{job_id}`.

Results show: Hit Rate metric, breakdown by source type, and per-question
expandable details (expected vs. retrieved sources, scores).

### Tab 4: End-to-End

Runs the full Golden Set through the complete pipeline including LLM. Shows
Hit Rate, Answer Contains Rate, and average latency. Per-question details
include the generated answer and source comparison.

Warning: slow and consumes LLM tokens.

### Tab 5: Ergebnisse

Lists saved evaluation results (max 3 per type). Each result shows filename,
type, hit rate, question count, and config. Delete button per result.

**Ergebnisse vergleichen**: Select two results for a side-by-side comparison.
Shows hit rate difference and lists all questions where the hit/miss status
changed between runs.

### Shared Components

**`_render_tuning_sliders(prefix)`**: Three linked sliders (Kandidaten,
Finale Chunks, Max pro Quelle) reused across the Frage testen and Retrieval
Eval tabs. The sliders are bound: `top_k_rerank ≤ top_k` and
`max_per_source ≤ top_k_rerank`.

**`_poll_eval_job(job_id)`**: Polls `GET /eval/status/{job_id}` every 1s
inside a `st.status` widget until done or error.

---

## Settings Page (`ui/pages/4_Settings.py`)

Central place for editing the runtime-tunable parameters of LoreKeeper.
Reads the current state via `GET /config` and sends diff-only updates via
`PUT /config`. Every save is persisted to `config/settings.yaml` **and**
applied in place to the running services — no restart, next query uses the new
values.

### Layout

Four `st.tabs`, plus a read-only expander for infrastructure knobs:

| Tab | Backend section | Live effect |
|---|---|---|
| 🔎 **Retrieval** | `retrieval.*` | Immediate on next query |
| 🧠 **LLM** | `llm.ollama.*`, `llm.gemini.*`, `llm.fallback_enabled` | Immediate on next query |
| 💬 **Conversation** | `conversation.*` | Immediate on next query (GC interval picks up on next wake) |
| 📥 **Ingestion** | `chunking.*` | Persists, but takes effect on the **next reindex** (explicit warning banner) |
| Read-only expander | `embeddings.*`, `vectorstore.*` | View only — change via `.env` / `settings.yaml` + restart |

### Retrieval tab

| Control | Maps to | Notes |
|---|---|---|
| Slider **Kandidaten (Top-K)** | `retrieval.top_k` | 1–50, how many chunks the vectorstore returns (bi-encoder recall pool). |
| Slider **Score-Threshold (Cosine)** | `retrieval.score_threshold` | 0.0–0.9 · Under 0.3 floods the context with noise. |
| Checkbox **Reranking aktiv** | `retrieval.reranking.enabled` | Disabling skips cross-encoder reranking. |
| Slider **Finale Chunks** | `retrieval.reranking.top_k_rerank` | `max_value` bound to top_k. Disabled when reranking is off. |
| Slider **Max. Chunks pro Quelle** | `retrieval.reranking.max_per_source` | Soft-Cap. `0` = unlimited. Disabled when reranking is off. |
| Checkbox **Hybrid-Default aktiv** | `retrieval.hybrid.enabled` | Sets the server default. Sidebar checkbox still overrides per session. |
| Slider **BM25-Anteil (RRF)** | `retrieval.hybrid.bm25_weight` | 0.0–1.0, only enabled when hybrid is on. |
| Slider **BM25-Kandidaten** | `retrieval.hybrid.bm25_top_k` | Only enabled when hybrid is on. |

### LLM tab

Two sub-sections side-by-side, one per provider (Ollama / Gemini). For each:
temperature, top-p, max tokens, timeout. A single global checkbox toggles
`llm.fallback_enabled`. The active model name is displayed read-only — model
selection stays in `settings.yaml` (swapping models has non-trivial startup
costs and isn't worth a UI toggle).

### Conversation tab

Three controls: `conversation.window_size` (sliding-window size for condense
history), `conversation.condense_question` (toggle condense step entirely),
`conversation.session_timeout_minutes` (idle-session GC deadline). All three
take effect on the next query; the GC interval refresh kicks in on the next
GC wake-up.

### Ingestion tab

Chunking-only. A **warning banner** at the top makes clear that changes do
**not** rewrite existing chunks — after saving, trigger **Alle Dokumente neu
indizieren** on the Sources page. Controls: `chunking.strategy` (heading_aware
/ recursive / fixed_size), `chunking.max_chunk_size`, `chunking.chunk_overlap`,
`chunking.min_chunk_size`.

### Allow-list and persistence

Not every field from `settings.yaml` is editable here. The backend keeps a
hard-coded allow-list in `src/config/manager.py:_EDITABLE_KEYS`; unknown or
disallowed keys are silently dropped by `ConfigManager.save_settings()` before
writing. Secrets (Gemini API key), infrastructure (`vectorstore.mode`,
`embeddings.model`, `api.*`, `logging.*`), and already-elsewhere-editable
fields (`llm.provider`, sources, prompts) never reach the Settings page.

The editable snapshot returned by `GET /config` always mirrors what the
`PUT /config` endpoint accepts plus the two read-only infrastructure blocks
the UI displays for context.

Internally, updates mutate the existing nested Pydantic model instances in
place — services holding references (Retriever, Reranker, ConversationManager,
LLM providers) keep working with the same objects and pick up new values on
their next call. That's why a running query doesn't blow up mid-stream when
you save a tuning change.

### Diff-only saves

Each tab's save button computes a dict diff between the loaded snapshot and
the current widget state, then `PUT /config`s only the changed keys. If
nothing changed, the button shows **Keine Änderungen** and makes no request.
Cache invalidation clears the page's `@st.cache_data(ttl=5)` so the
post-save view reflects disk state.

---

## Session State Overview

| Key | Type | Meaning |
|-----|------|---------|
| `messages` | `list[dict]` | Full chat history including `sources` and per-message `usage` |
| `session_id` | `str \| None` | Backend session UUID for conversation context |
| `session_usage` | `dict` | Cumulative `{tokens_in, tokens_out, tokens_thinking}` for the active session, displayed in the header metric |
| `_selected_provider` | `str` | Currently selected provider (widget state) |
| `_provider_switch_ok` | `bool` | Temporary flag: switch succeeded |
| `_provider_switch_error` | `str` | Temporary flag: error message on switch failure |
| `_scan_result` | `dict \| None` | Sources page: cached response from `POST /sources/scan` (survives reruns between add-source steps) |
| `_scanned_path` | `str` | Sources page: the path that was scanned (paired with `_scan_result`) |
