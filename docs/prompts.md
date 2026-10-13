# Prompts

Wie LoreKeeper die Prompts fürs LLM baut: die vier Jinja2-Templates, ihre Variablen und wie man
sie über UI, API oder YAML anpasst. Die aktuelle Fassung steht immer in `config/prompts.yaml` —
dieses Dokument zeigt Aufbau und Absicht, keine vollständige Kopie.

## Inhaltsverzeichnis
1. [Überblick](#überblick)
2. [Die Templates](#die-templates)
   - [`system` — Systemprompt](#system--systemprompt)
   - [`qa` — Antwort-Prompt](#qa--antwort-prompt)
   - [`condense` — Folgefragen umschreiben](#condense--folgefragen-umschreiben)
   - [`no_context` — keine Treffer](#no_context--keine-treffer)
3. [Prompts anpassen](#prompts-anpassen)
   - [Über die UI (empfohlen)](#über-die-ui-empfohlen)
   - [Über die API](#über-die-api)
   - [Varianten](#varianten)
   - [Per Hand in der YAML](#per-hand-in-der-yaml)
4. [Gestaltungsregeln](#gestaltungsregeln)

---

## Überblick

Alle Prompts liegen als Jinja2-Templates in `config/prompts.yaml`; der `PromptManager`
(`src/prompts/manager.py`) rendert sie zur Laufzeit.

```yaml
# config/prompts.yaml – Aufbau
system:     "Du bist LoreKeeper, …"            # Systemprompt
qa:         "Beantworte die folgende Frage …"  # Antwort-Prompt: Verlauf + Quellen + Frage
condense:   "Gegeben ist ein Chat-Verlauf …"   # Folgefrage → eigenständige Frage
no_context: "Zu deiner Frage … keine …"        # keine Treffer (ohne LLM)
```

---

## Die Templates

### `system` — Systemprompt

Geht bei jedem LLM-Aufruf als Systemnachricht mit. Legt Rolle, Sprache, **Antwortformat im
Telegram-Stil**, Längenregeln und Quellenangaben fest. **Keine Variablen.**

Der Prompt verlangt:

- **Deutsch**, außer die Frage ist englisch.
- **Telegram-Stil**: kurze, scanbare Absätze (max. 2–3 Sätze), Leerzeile dazwischen; locker, aber
  sachlich — keine Textwände.
- **Lead = genau ein Satz** mit der Kernantwort, wie die erste Nachricht in einem Chat.
- **Emoji-Überschriften** statt Markdown-Überschriften — Emoji + fetter Titel, z.B.
  `**⚔️ Fähigkeiten**`, `**📜 Hintergrund**`, `**🤝 Beziehungen**`; frei passend zum Inhalt
  gewählt. `**🎲 Werte**` nur für echte Spielwerte.
- **Listen** mit `- ` direkt am Zeilenanfang (eingerückte Listen rendert Streamlit als Code-Block).
- **Fett** für Eigennamen, Schlüsselbegriffe und Werte.
- Optional am Ende **`**🔗 Siehe auch:**`** mit verwandten Themen aus den Quellen.
- **Quellen** als Dateiname in Klammern, z.B. `(Arkenfeld.md)`, bei jeder nicht-trivialen Aussage.
- **Lücken offen benennen** („Quellen schweigen dazu …“) statt zu raten; bei Widersprüchen beide
  Varianten mit Quelle.

| Fragetyp | Format |
|---|---|
| Faktenfrage („Wie alt ist X?“) | 1–2 kurze Sätze, keine Überschriften |
| Offene Frage („Erzähl mir von …“) | Mehrere Häppchen mit Emoji-Überschriften, dicht statt lang |
| Nach Typ | NPCs → Rolle · Hintergrund · Fähigkeiten · Beziehungen; Orte → Lage · Bedeutung · Bewohner · Besonderheiten; Regeln → Grundprinzip in einem Satz, dann Werte als Liste |

Länge durch Spekulation über die Quellen hinaus ist ausdrücklich verboten.

```markdown
**Arkenfeld** ist eine mittelgroße Handelsstadt im zentralen Tiefland und ein Knotenpunkt für
Karawanen aus dem Norden. (Arkenfeld.md)

**📍 Lage**
Am Schnittpunkt von Salzstraße und Fluss Veren — zwischen den Eisbergen und den südlichen Königreichen.

**🏛️ Bedeutung**
- Größter Salzumschlagplatz der Region
- Sitz der **Händlergilde der Sieben Siegel** (Arkenfeld.md)

**🔗 Siehe auch:** Salzstraße, Händlergilde der Sieben Siegel
```

---

### `qa` — Antwort-Prompt

Wird bei jeder Anfrage mit gefundenen Chunks gerendert.

| Variable | Typ | Inhalt |
|----------|-----|--------|
| `chunks` | `list[dict]` | Chunks nach Reranking, Slot und Expansion |
| `chunks[i].source_file` | `str` | Relativer Pfad, z.B. `Orte\Arkenfeld.md` |
| `chunks[i].heading` | `str` | Überschriftenpfad, z.B. `Arkenfeld > Überblick` |
| `chunks[i].content` | `str` | Chunk-Text (mit Überschrift vorangestellt) |
| `question` | `str` | Die (ggf. umgeschriebene) Frage |
| `history` | `list[dict]` | Das letzte Frage/Antwort-Paar der Session (`role`: `user`/`assistant`, `content`) — leer bei der ersten Frage. Umfang: `conversation.answer_history_pairs` (1), jede Nachricht auf `answer_history_max_chars` (1500) Zeichen gekürzt. Templates ohne Verlaufsblock ignorieren ihn |

Der Verlauf steht **vor** den Quellen und ist als „keine Quelle“ markiert: Er löst nur Bezüge auf
die vorige Antwort auf („fass das kürzer zusammen“), Fakten kommen aus den Quellen. Bewusst nur ein
Paar — mehr Verlauf verwässert die Quellen und gibt früheren, evtl. halluzinierten Antworten
Gewicht. Sessions sind rollengebunden, ein Spieler sieht hier nur seinen eigenen Verlauf.

```yaml
qa: |
  Beantworte die folgende Frage basierend auf den bereitgestellten Quellen.
  {% if history %}

  ### Bisheriger Chat (keine Quelle):
  Nur um Bezüge wie „das“, „er“ oder „kürzer“ zu verstehen. Fakten kommen
  ausschließlich aus den Quellen unten — was nur im Chat steht, gilt nicht als belegt.
  {% for msg in history %}
  {{ "Nutzer" if msg.role == "user" else "LoreKeeper" }}: {{ msg.content }}
  {% endfor %}
  {% endif %}

  ### Quellen:
  {% for chunk in chunks %}
  [{{ chunk.source_file }} — {{ chunk.heading }}]
  {{ chunk.content }}
  {% endfor %}

  ### Frage:
  {{ question }}

  ### Antwort:
```

---

### `condense` — Folgefragen umschreiben

Nur aufgerufen, wenn `conversation.condense_question: true`, ein Verlauf existiert **und** die
Frage wie eine Folgefrage aussieht (`needs_condense()` in `src/conversation/manager.py` —
Pronomen, Rückverweise, führendes „und/aber“, „das/es“ am Ende oder ≤ 2 Wörter). Eigenständige
Fragen gehen unverändert in die Suche.

| Variable | Typ | Inhalt |
|----------|-----|--------|
| `history` | `list[dict]` | Bisherige Nachrichten (Sliding Window, `window_size` Paare) |
| `history[i].role` | `str` | `user` oder `assistant` |
| `history[i].content` | `str` | Nachrichtentext |
| `question` | `str` | Die neue Folgefrage |

```yaml
condense: |
  Gegeben ist ein Chat-Verlauf und eine Follow-Up-Frage. Formuliere die
  Follow-Up-Frage so um, dass sie als eigenständige Frage verständlich ist.
  Gib NUR die umformulierte Frage zurück, ohne Erklärung.

  ### Chat-Verlauf:
  {% for msg in history %}
  {{ msg.role }}: {{ msg.content }}
  {% endfor %}

  ### Follow-Up-Frage:
  {{ question }}

  ### Eigenständige Frage:
```

Beispiel: „Was ist Arkenfeld?“ → … → „Wer regiert dort?“ wird zu „Wer regiert Arkenfeld?“. Die UI
zeigt die tatsächlich gesuchte Frage als „🔎 Gesucht nach: …“.

---

### `no_context` — keine Treffer

Kommt, wenn kein Chunk die `score_threshold` erreicht. **Kein LLM-Aufruf** — reine Textersetzung.
Einzige Variable: `question` (die ursprüngliche Frage).

```yaml
no_context: |
  Zu deiner Frage "{{ question }}" konnte ich leider keine relevanten
  Informationen in den Quellendokumenten finden.
  …
```

---

## Prompts anpassen

### Über die UI (empfohlen)

Die Seite **✏ Prompts** hat drei Tabs:

| Tab | Zweck |
|-----|-------|
| **Aktive Prompts** | Die vier Templates bearbeiten; jedes zeigt seine Variablen als Hilfe, eine Vorschau rendert mit Beispieldaten |
| **Varianten** | Aktive Prompts als benannte Variante speichern, Varianten bearbeiten, aktivieren, löschen. Die aktive Variante ist mit ✅ markiert |
| **Vergleichen** | Zwei Varianten (oder aktiv vs. Variante) nebeneinander; unterschiedliche Templates mit ⚠ |

Änderungen wirken **sofort** — das Backend ersetzt den `PromptManager` beim Speichern, kein
Neustart.

### Über die API

| Methode | Endpoint | Zweck |
|---------|----------|-------|
| GET | `/prompts/active` | Aktive Prompts lesen |
| PUT | `/prompts/active` | Aktive Prompts speichern (Jinja2 wird geprüft, Hot-Reload) |
| GET | `/prompts/variants` | Varianten auflisten (mit `is_active`) |
| GET | `/prompts/variants/{name}` | Eine Variante laden |
| PUT | `/prompts/variants/{name}` | Variante anlegen oder ändern |
| DELETE | `/prompts/variants/{name}` | Variante löschen |
| POST | `/prompts/activate/{name}` | Variante aktivieren (überschreibt `prompts.yaml`, Hot-Reload) |
| POST | `/prompts/preview` | Template mit Beispieldaten rendern |

Alle schreibenden Endpoints prüfen die Jinja2-Syntax — ein fehlerhaftes Template ergibt HTTP 422.
Gerendert wird in einer `SandboxedEnvironment` (Vorschau und Betrieb): Schleifen, Bedingungen,
Filter und Attribute der Variablen gehen wie gewohnt, Umwege zu Python-Interna
(`''.__class__.__mro__…`) werden abgelehnt (Render-Fehler, 422). Gespeichert wird atomar (temporäre
Datei + Umbenennen) — ein Absturz beim Speichern hinterlässt keine halbe `prompts.yaml`.

### Varianten

Varianten liegen als YAML in `config/prompts/` (gitignored), Aufbau wie `prompts.yaml` plus
optionalem `_meta`:

```yaml
_meta:
  name: "Knapp"
  description: "Kürzere, knackigere Antworten"
system: |
  …
qa: |
  …
condense: |
  …
no_context: |
  …
```

`_meta` wird beim Laden entfernt. „Aktiv“ ist eine Variante, wenn ihre vier Templates mit
`prompts.yaml` übereinstimmen — eine Variante, die vor einer Änderung an `prompts.yaml` gespeichert
wurde (z.B. ohne Verlaufsblock im `qa`), gilt danach als nicht aktiv; einmal neu speichern.

### Per Hand in der YAML

`config/prompts.yaml` direkt bearbeiten — wirkt erst nach einem Neustart oder einem
`PUT /prompts/active`.

```jinja2
{{ variable }}           # Ausgabe
{% for x in liste %}     # Schleife
{% if bedingung %}       # Bedingung
{# Kommentar #}          # wird nicht gerendert
```

**Eigenes Template ergänzen:** Schlüssel in `prompts.yaml`, `render_*`-Methode in
`src/prompts/manager.py`, Aufruf in `src/api/query_routes.py`.

---

## Gestaltungsregeln

- **Quellenpflicht:** Der Systemprompt verlangt `(datei.md)` — das hält das LLM davon ab,
  Trainingswissen als Quelle auszugeben.
- **Telegram-Format:** kurze Absätze + Emoji-Überschriften + Listen sind am Handy scanbar und
  strukturiert, ohne Länge zu erzwingen. Früher stand dort „max. 3–5 Sätze“, das ergab flache
  Drei-Zeilen-Blöcke; jetzt erst Struktur, dann Längenregeln nach Fragetyp.
- **Gegen Halluzinationen:** „Erfinde keine Informationen“ + „sag ehrlich, wenn die Quellen nichts
  hergeben“, dazu `no_context` ohne LLM und ein Verlauf, der ausdrücklich keine Quelle ist.
- **Sprache:** Deutsch, automatisch Englisch bei englischen Fragen.
