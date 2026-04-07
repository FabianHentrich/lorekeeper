# Prompt System

## Overview

All prompts live in `config/prompts.yaml` as Jinja2 templates.
The `PromptManager` (`src/prompts/manager.py`) renders them at runtime.

```yaml
# config/prompts.yaml – structure
system:   "You are LoreKeeper..."             # system prompt for the LLM
qa:       "Answer the question..."            # main QA template
condense: "Reformulate the question..."       # follow-ups → standalone
no_context: "No relevant sources found..."   # fallback when no hits
```

---

## Templates in Detail

### `system` — System Prompt

Passed to every LLM call as a system message. Defines persona, language, and answer rules.

**No template variables** — static text.

```yaml
system: |
  You are LoreKeeper, an expert on the user's pen-and-paper world.
  You answer questions exclusively based on the provided source documents.
  If the sources do not contain an answer, say so honestly. Do not invent information.

  Rules:
  - Answer in German unless the user asks in English
  - Cite the source only as a filename in parentheses, e.g. (Arkenfeld.md)
  - Keep answers brief — at most 3–5 sentences, unless the user explicitly asks for details
  - No bullet lists if running prose suffices
  - If sources conflict: mention both versions
```

---

### `qa` — Main QA Template

Rendered for every request when context chunks are found.

**Available variables:**

| Variable | Type | Content |
|----------|------|---------|
| `chunks` | `list[dict]` | Retrieved chunks after reranking |
| `chunks[i].source_file` | `str` | Relative file path, e.g. `Locations\Arkenfeld.md` |
| `chunks[i].heading` | `str` | Heading path, e.g. `Arkenfeld > Overview` |
| `chunks[i].content` | `str` | Chunk text (with heading prefix) |
| `question` | `str` | The (possibly condensed) user question |

```yaml
qa: |
  Answer the following question based on the provided sources.

  ### Sources:
  {% for chunk in chunks %}
  [{{ chunk.source_file }} — {{ chunk.heading }}]
  {{ chunk.content }}
  {% endfor %}

  ### Question:
  {{ question }}

  ### Answer:
```

**Rendered example:**
```
Answer the following question based on the provided sources.

### Sources:
[Locations\Arkenfeld.md — Arkenfeld > Overview]
Arkenfeld > Overview

| Property    | Detail                   |
| Type        | Trading city, mid-sized  |
...

### Question:
What is Arkenfeld?

### Answer:
```

---

### `condense` — Question Condensing

Only called when `conversation.condense_question: true` and conversation history is present.
Reformulates follow-up questions into standalone questions.

**Available variables:**

| Variable | Type | Content |
|----------|------|---------|
| `history` | `list[dict]` | Previous messages (sliding window) |
| `history[i].role` | `str` | `user` or `assistant` |
| `history[i].content` | `str` | Message text |
| `question` | `str` | The new follow-up question |

```yaml
condense: |
  Given a chat history and a follow-up question, reformulate the
  follow-up question so it is understandable as a standalone question.
  Return ONLY the reformulated question, without explanation.

  ### Chat History:
  {% for msg in history %}
  {{ msg.role }}: {{ msg.content }}
  {% endfor %}

  ### Follow-Up Question:
  {{ question }}

  ### Standalone Question:
```

**Example:**
```
Chat history:
user: What is Arkenfeld?
assistant: A trading city in the central lowlands...

Follow-up question: Who rules it?
→ Standalone question: Who rules Arkenfeld?
```

---

### `no_context` — Fallback Without Hits

Returned when ChromaDB finds no chunks above the `score_threshold`.
No LLM call — direct string interpolation.

**Available variables:**

| Variable | Type | Content |
|----------|------|---------|
| `question` | `str` | The original user question |

```yaml
no_context: |
  For your question "{{ question }}" I could not find any relevant
  information in the source documents.
  ...
```

---

## Customizing Prompts

Changes to `prompts.yaml` take effect **without a restart** — wait, actually the `PromptManager`
reads the file once at server start. For live changes the server must be restarted.

**Jinja2 syntax:**
```jinja2
{{ variable }}           # output
{% for x in list %}      # loop
{% if condition %}        # condition
{# comment #}            # not rendered
```

**Adding a custom template key:**
1. Add a new key in `prompts.yaml`
2. Add a `render_*` method in `src/prompts/manager.py`
3. Call it from `src/api/routes.py`

---

## Prompt Design Notes

- **Enforce source citations:** The `system` prompt requires `(filename.md)` —
  this prevents the LLM from using training knowledge as a source
- **Length control:** "at most 3–5 sentences" in the system prompt prevents verbose answers;
  for detail questions the user explicitly asks for more
- **Hallucination prevention:** "Do not invent information" + "say so honestly if no source exists" —
  together with the `no_context` fallback, this reduces hallucinations
- **Language:** German as default with automatic English switch when the user asks in English
