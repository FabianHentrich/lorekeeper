# Provider Strategy

This document details the Language Model (LLM) providers supported by LoreKeeper, how to configure them, how fallback and condense mechanisms work, and how token usage is tracked.

## Table of Contents
1. [Overview](#overview)
2. [Provider Comparison](#provider-comparison)
3. [Configuration](#configuration)
   - [Ollama](#ollama)
   - [Gemini](#gemini)
4. [Switching Providers at Runtime](#switching-providers-at-runtime)
5. [Fallback Provider](#fallback-provider)
6. [Condense Provider](#condense-provider)
7. [Token Usage Reporting](#token-usage-reporting)
8. [Adding a New Provider](#adding-a-new-provider)

---

## Overview

LoreKeeper abstracts LLM access behind a unified `BaseLLMProvider` interface.
Currently supported providers: **Ollama** (local) and **Gemini** (cloud).

```python
class BaseLLMProvider(ABC):
    provider: str = "unknown"
    model: str                       # set in __init__ — no default

    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...
    def generate_stream(self, prompt: str, system_prompt: str = "",
                        stream_result: StreamResult | None = None,
                        **kwargs) -> AsyncGenerator[str, None]: ...
    async def health_check(self) -> bool: ...
```

Implementations write `generate_stream` as `async def` with `yield`. `StreamResult`
is a per-call context: the provider fills `usage`, the `Generator` sets `model` to
the model that actually answered. A new provider must set `self.model` in
`__init__` and be registered in `ProviderFactory`.

---

## Provider Comparison

| Criterion | Ollama (local) | Gemini (cloud) |
|-----------|---------------|----------------|
| **Latency** | 5–30s (hardware-dependent) | 2–8s |
| **Cost** | Power + GPU | Pay-per-token |
| **Privacy** | Fully local | Data leaves the system |
| **Context window** | Model-dependent (qwen3:8b: 128k) | 1M tokens (Flash) |
| **Quality** | Good for German, model-dependent | Very good, multimodal |
| **Offline** | ✅ | ❌ |
| **Setup** | Install Ollama + pull model | API key in `.env` |

---

## Configuration

### Ollama

```yaml
llm:
  provider: ollama
  ollama:
    base_url: "http://localhost:11434"
    model: "qwen3:14b"         # currently recommended (16 GB VRAM)
    temperature: 0.3
    top_p: 0.9
    max_tokens: 2048
    timeout: 300
```

**Model recommendations by hardware:**

| VRAM | Recommendation | Quality |
|------|---------------|---------|
| < 8 GB | `qwen3:4b` | Good |
| 8–12 GB | `qwen3:8b` | Very good |
| ≥ 16 GB | `qwen3:14b` | Excellent |

**Qwen3-specific:** LoreKeeper automatically filters `<think>...</think>` blocks out of the
stream. Reasoning is disabled via `reasoning_effort: "none"` in the request body (faster responses).

### Gemini

```yaml
llm:
  provider: gemini
  gemini:
    model: "gemini-2.5-flash"
    api_key_env: GEMINI_API_KEY   # name of the env variable
    temperature: 0.3
    top_p: 0.9
    max_tokens: 1024
    timeout: 30
```

```ini
# .env
GEMINI_API_KEY=your-key-here
```

**Rate limiting:** Gemini Free Tier allows 15 RPM. LoreKeeper implements automatic
rate limiting (4s minimum interval) + retry with exponential backoff on 429 errors.

---

## Switching Providers at Runtime

Via the Streamlit sidebar (dropdown) or directly via API:

```bash
curl -X POST http://localhost:8000/provider \
  -H "Content-Type: application/json" \
  -d '{"provider": "gemini"}'
```

The switch:
1. Creates a new provider (`ProviderFactory.create()`)
2. Creates a new `condense_provider` (if configured)
3. Creates a new `Generator` with the new provider
4. On failure (e.g. missing API key): rolls back to the old provider

The active provider remains set until the next server restart — it does **not** overwrite
`settings.yaml`, only the in-memory state.

---

## Fallback Provider

```yaml
llm:
  fallback_provider: gemini
  fallback_enabled: true   # default: false
```

When `fallback_enabled: true` and the primary provider fails (timeout, Ollama down),
the system automatically switches to the fallback. The fallback uses the same configuration
of the respective provider (same model settings).

For streaming (`/query/stream`) the fallback only kicks in if the primary fails
**before the first token** — once tokens are on the wire, a restart would duplicate
output, so the error is surfaced instead. The `done` event's `model_used` names the
model that actually answered (the fallback's model after a fallback).

---

## Condense Provider

```yaml
conversation:
  condense_model: "qwen3:4b"   # null = use primary provider
```

Optionally, a separate, faster model can be configured for question condensing (reformulating
follow-up questions into standalone questions). Always an Ollama model —
Gemini is not supported for condensing.

---

## Token Usage Reporting

Both providers expose per-request token counts that LoreKeeper surfaces in
the SSE `done` event (`usage`) and accumulates per session
(`session_usage`). The UI renders these as a per-message caption and a
header metric — see [`docs/ui-ux.md`](ui-ux.md) for the rendering details.
This section covers what each provider actually reports and the
provider-specific quirks.

| Field | Ollama (Qwen3) | Gemini 2.5 |
|---|---|---|
| `tokens_in` | ✅ `prompt_tokens` (from OpenAI-compatible response) | ✅ `usage_metadata.prompt_token_count` |
| `tokens_out` | ✅ `completion_tokens` | ✅ `usage_metadata.candidates_token_count` |
| `tokens_thinking` | Reasoning chunks still received in the `reasoning` field — normally `0`, see below | ✅ `usage_metadata.thoughts_token_count` (when thinking is enabled) |

### Ollama specifics

**Context window:** Ollama's default context (4096 tokens) is too small for
LoreKeeper's prompts — a section-expanded rule question already needs ~3600
input tokens plus up to `max_tokens` for the answer, and overflow silently cuts
off the start of the prompt (the system prompt). The OpenAI-compatible endpoint
ignores a per-request `num_ctx`, so the size is set on the Ollama side:

```powershell
setx OLLAMA_CONTEXT_LENGTH 16384   # then quit Ollama in the tray and start it again
ollama ps                          # CONTEXT column must show 16384
```

16384 costs ~2.6 GB VRAM on qwen3:14b (11 GB total, fits a 16 GB GPU fully).
If the LoreKeeper backend auto-starts `ollama serve` itself (Ollama not running
at backend start), that process inherits the backend's environment — start the
Ollama app first after changing the variable.

**Keep-alive:** Ollama unloads a model after 5 minutes idle; the next question
then waits ~16 s for the reload (measured: first token 18 s instead of 1.1 s).
`setx OLLAMA_KEEP_ALIVE 1h` keeps it loaded for an hour after the last
question (`ollama ps` → UNTIL "59 minutes from now"), at the cost of ~11 GB
VRAM held during that time; `-1` never unloads.

The OpenAI-compatible streaming API only emits a usage block when
`stream_options={"include_usage": True}` is passed on the request. Without
that flag the final stream chunk arrives with empty `usage` and
LoreKeeper would report zeros. The flag is set in
`OllamaProvider.generate_stream` and the resulting numbers are written to the
per-call `StreamResult.usage`, which the route reads after the stream
finishes (no shared provider state, so concurrent streams don't mix).

Qwen3 thinking is switched off with `extra_body={"reasoning_effort": "none"}`
(`OllamaProvider._thinking_off`). Current Ollama versions ignore the older
`/no_think` prompt suffix and stream reasoning in a separate `reasoning` delta
field instead of `<think>` tags — measured at ~7 s before the first answer
token on qwen3:14b. `think: false` in the body is ignored as well. Any
reasoning that still arrives (other thinking models, older servers) is never
shown and is counted as `tokens_thinking`, so a non-zero value means thinking
is back on. This trades reasoning quality for latency — there is no
$-per-token penalty locally, so the goal is "answer fast".

### Gemini specifics

`google-genai` exposes `usage_metadata` on stream chunks, typically only on
the final one. LoreKeeper reads it from any chunk that carries it, so the
last non-null value wins.

`thoughts_token_count` is the **cost-relevant** field on Gemini 2.5 models:
when thinking is enabled (default for `gemini-2.5-flash` and
`gemini-2.5-pro` on complex queries), thinking tokens are billed as output
tokens but do not appear in `candidates_token_count`. They can easily
exceed the visible output by a factor of 2–10× on reasoning-heavy
questions, so the UI shows them as a separate `🧠 N think` segment in the
per-message caption. **Watch this number** when comparing Gemini cost
against Ollama — it is the most common source of "why is my bill higher
than expected".

### What this means for the cost comparison

The per-session totals in the header metric give a rough live estimate of
what a session would cost on a paid provider. For Ollama runs the numbers
are still useful as a load indicator (input context size, generated length),
but the cost column in the [Provider Comparison](#provider-comparison)
table only kicks in when running on Gemini — and `tokens_thinking` is the
column that decides whether the bill is small or surprising.

---

## Adding a New Provider

1. Create `src/generation/providers/my_provider.py`, implement `BaseLLMProvider`
2. Set class attribute `provider = "my_provider"`
3. Register in `ProviderFactory` (`provider_factory.py`)
4. Add config class in `src/config/manager.py`
5. Update `settings.yaml` and `.env.example`
