# Provider Strategy

## Overview

LoreKeeper abstracts LLM access behind a unified `BaseLLMProvider` interface.
Currently supported providers: **Ollama** (local) and **Gemini** (cloud).

```python
class BaseLLMProvider(ABC):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]: ...
    async def health_check(self) -> bool: ...
```

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
    model: "qwen3:8b"          # currently recommended
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
| > 16 GB | `qwen3:14b` | Excellent |

**Qwen3-specific:** LoreKeeper automatically filters `<think>...</think>` blocks out of the
stream. Reasoning behavior is disabled via `/no_think` (faster responses).

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

## Adding a New Provider

1. Create `src/generation/providers/my_provider.py`, implement `BaseLLMProvider`
2. Set class attribute `provider = "my_provider"`
3. Register in `ProviderFactory` (`provider_factory.py`)
4. Add config class in `src/config/manager.py`
5. Update `settings.yaml` and `.env.example`
