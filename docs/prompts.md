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

Passed to every LLM call as a system message. Defines persona, language,
**Telegram-style answer format**, length heuristics, and source citation
rules.

**No template variables** — static text.

The current system prompt instructs the model to:

- **Answer in German** by default (English when asked in English).
- **Use a Telegram-messenger style**: short, scannable paragraphs (max 2–3
  sentences each), separated by blank lines. Mobile-friendly, casual but
  factual — no academic walls of text.
- **Open with a one-line lead** that delivers the core answer, the way a
  chat reply would.
- **Use emoji headers** instead of classic Markdown headings — an emoji
  plus bold text, e.g. `**⚔️ Fähigkeiten**`, `**📜 Hintergrund**`,
  `**🤝 Beziehungen**`, `**📍 Ort**`, `**🎲 Regeln**`. The emoji is chosen
  to fit the section's content.
- **Use bullet lists** with `•` or `–` for enumerations (stats, members,
  steps). One bullet = one line.
- **Bold proper names, key terms, and numerical values**.
- **Optionally close with `**🔗 Siehe auch:**`** when sources mention
  related topics.
- **Cite sources** as a filename in parentheses, e.g. `(Arkenfeld.md)`,
  for any non-trivial claim.
- **Acknowledge gaps explicitly** ("Quellen schweigen dazu …") instead of
  inventing facts. On contradictions, name both variants and mark the
  source of each.

#### Length heuristics

| Question type | Format |
|---|---|
| Factual lookup ("How old is X?") | 1–2 short sentences, no headers |
| Open-ended ("Tell me about …") | Multiple paragraph chunks with emoji headers, scannable but never bloated |
| Detail by type | NPCs → role · background · abilities · relationships; Locations → position · meaning · inhabitants · specifics; Rules → principle then concrete values as a list |

The prompt explicitly forbids generating length by speculating beyond the
sources.

#### Example output (Telegram style)

```markdown
**Arkenfeld** ist eine mittelgroße Handelsstadt im zentralen Tiefland und
ein Knotenpunkt für Karawanen aus dem Norden. (Arkenfeld.md)

**📍 Lage**
Im Schnittpunkt der Salzstraße und des Flusses Veren — strategisch
zwischen den Eisbergen im Norden und den südlichen Königreichen.

**🏛️ Bedeutung**
• Größter Salzumschlagplatz der Region
• Sitz der **Händlergilde der Sieben Siegel**
• Neutraler Boden für Verhandlungen zwischen den Nordstämmen (Arkenfeld.md)

**🤝 Bewohner**
Etwa 12.000 Seelen, gemischt aus Menschen, Halblingen und einer kleinen
Zwergenenklave aus den Eisbergen.

**🔗 Siehe auch:** Salzstraße, Händlergilde der Sieben Siegel
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

Changes to `prompts.yaml` require a **server restart** — the `PromptManager` reads the file once at startup.

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
- **Telegram-style formatting:** short paragraphs + emoji headers + bullets
  produce answers that are scannable on mobile and visually structured
  without forcing verbosity. The prompt shifted from "max 3–5 sentences"
  (which produced flat 3-line prose blocks) to a structure-first format
  with explicit length heuristics by question type
- **Hallucination prevention:** "Do not invent information" + "say so honestly if no source exists" —
  together with the `no_context` fallback, this reduces hallucinations
- **Language:** German as default with automatic English switch when the user asks in English
