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
Telegram-Stil**, Längenregeln und Quellenangaben fest. **Keine Variablen.** Antworten stützen sich
auf die Quellen und den [Regelkern](#regelkern) im `qa`-Prompt.

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

  ### Regelkern (gilt immer, auch wenn die Quellen ihn nicht wiederholen):
  Bei Würfen, Schaden, Rüstung oder Ausrüstung jede Zahl der Antwort gegen diese Regeln prüfen …
  - **Probe**: …   - **Ass-Regel**: …   (14 Stichpunkte, ~450 Tokens)

  ### Frage:
  {{ question }}

  ### Antwort:
```

#### Regelkern

Ein fester Block mit 14 Stichpunkten zum Regelwerk, verdichtet aus dem Regelbuch (Kapitel
Grundregeln, Kampfregeln, Gegenstände, Ressourcen): Probe und Würfel je Rang, Ass-Regel, Angriff,
kritischer Treffer, Waffenkategorien, Qualität, Schadensreihenfolge SP → RP → LP, Rüstungswerte,
Schadenstypen, magische Waffen, Zug, Statuseffekte, Lesart der Simulationsberichte. Die
Einleitung des Blocks sagt dem Modell, nur die zur Frage passenden Punkte zu nutzen und ihn bei
reinen Lore-Fragen nicht zu erwähnen; konkrete Werte aus den Quellen haben Vorrang.

**Warum**: Grundregeln gelten für jeden Würfelwert, passen aber zu keiner konkreten Frage — die
Suche holt zu „Wie viel Schaden macht Schattenklinge?“ die Fähigkeitskarte, nicht die Ass-Regel
(Kapitel 1) oder den kritischen Treffer (Kapitel 4). Das Modell las „1W10 + 10“ deshalb als
„höchstens 20“. Besseres Retrieval löst das nicht.

**Warum im `qa`-Prompt und nicht im Systemprompt** (gemessen 14.10., qwen3:14b, je Frage dieselben
Chunks, nur der Prompt verschieden, noch mit dem alten Reranker mMiniLMv2): Im Systemprompt änderte der Block fast nichts — „höchste
Wurfzahl mit Rang 1“ wurde sogar schlechter (3 statt 4, richtig wäre „nach oben offen“). Direkt vor
der Frage wirkt er. Fünf Regelfragen, je 3 Läufe (Temperatur 0,3), richtig vorher → nachher:

| Frage | vorher | nachher | nachher, live mit jina-Reranker |
|---|---|---|---|
| Höchste Wurfzahl mit Rang 1 | 0/3 („höchstens 4“) | 3/3 („nach oben offen, z. B. 4 + 3 = 7“) | 0/3 sauber — Lead „höchstens 13“ (2×) bzw. „4“, darunter jeweils „unbegrenzt“ |
| Was fängt eine normale mittlere Rüstung ab? | 0/3 („2 je RP“) | 3/3 („3 RP → 6 Schaden“) | 3/3 |
| Wie funktioniert das Würfelsystem? | 3/3 | 3/3, beim Thema | 3/3, einmal doch der ganze Block (10,8 s) |
| Schattenklinge auf Rang 6 „maximal“ | 0/3 | 0/3 — Simulations-Durchschnitt 29,94 als Maximum | 0/3 — jetzt Fähigkeitskarte (1W10 + X), 2× „nach oben offen“, X falsch (richtig +10) |
| Dolch, STÄ 4, Wurfzahl 12 (richtig: 12) | 0/3 | 0/3 — Krit-Schwelle und Grundschaden verwechselt | 0/3 (6 / 10 / 10) |

Der Reranker ändert die Chunks und damit das Ergebnis: Mit jina (seit 14.10.) kippt die Rang-1-Frage.
Gegenprobe mit den festen jina-Chunks (je 3 Läufe): wie abgerufen 1/3 richtig, ohne das Ass-Regel-Beispiel
„W8 mit 8, dann 5 → 13“ 0/3, Ass-Regel-Chunk ans Ende verschoben 0/3 — das Beispiel ist nicht die
Ursache; das Modell hängt sich ohne Denken an „höchstens“ + W4 („höchstens 4“). Die Streuung zwischen
Läufen ist groß, das 3/3 mit mMiniLMv2 war zum Teil Glück.

**Mit Qwen3-Denken** (nur im Test, gleiche jina-Chunks, je 3 Läufe): Rang 1 3/3 richtig („keine
Obergrenze“), Dolch 1/3 richtig (12) und 2/3 **leere Antworten** (das Denken verbraucht das Token-Budget,
~40 s), Schattenklinge 3/3 „X ist in den Quellen nicht angegeben“ — ehrlich statt falsch. Kosten:
14–41 s pro Antwort statt 2–5 s. Als Standard ungeeignet; Denken nur für Rechenfragen wäre eine
Option (Roadmap 3.14).

Vier Lore-Fragen (Arkenfeld, Escanor, Orden der letzten Dämmerung, Malek) blieben gleich gut, kein
Regeltext in den Antworten; Antwortzeit +0–2 s.

Zwei Formulierungen kamen aus der Messung: „bei jedem Würfel einschließlich W4 (eine 4 auf dem W4 →
erneut werfen)“ — ohne das Beispiel meinte das Modell, beim W4 gelte die Ass-Regel nicht; „den
Regelkern nie als Ganzes wiedergeben“ — sonst schrieb es bei der Würfelsystem-Frage alle 14 Punkte
ab (17 s). Verworfen: „waffenlos“ als eigener Stichpunkt (Dolch weiter 0/3).

**Grenzen**: Regeln *anwenden und rechnen* bleibt ohne Denken unzuverlässig (Roadmap 3.14, am 14.10.
so gelassen). Beim Dolch fehlt zusätzlich die Waffentabelle im Kontext — die Frage holt den Abschnitt
„Kritischer Treffer“ und NPC-Werte (Roadmap 3.10). **Pflege**: Der Block ist eine Kopie
des Regelbuchs — ändert sich dort eine Regel, muss er von Hand nachgezogen werden.

---

### `condense` — Folgefragen umschreiben

Nur aufgerufen, wenn `conversation.condense_question: true`, ein Verlauf existiert **und** die
Frage wie eine Folgefrage aussieht (`needs_condense()` in `src/conversation/manager.py` —
Pronomen, Rückverweise, führendes „und/aber“, „das/es“ am Ende, ≤ 2 Wörter oder **kein eigenes
Inhaltswort**: besteht die Frage nur aus Funktionswörtern und Wörtern über die Form der Antwort
(„Fass das kürzer zusammen“, „Gib mir ein Beispiel“, „Erklär das genauer“), bezieht sie sich auf die
vorige Antwort. Jedes andere Wort zählt als Inhaltswort — „Fass die Geschichte von Arkenfeld
zusammen“ bleibt eigenständig. Eigenständige Fragen gehen unverändert in die Suche. Bekannte
Fehlauslösung: „Welche Klassen gibt es?“ („es“ am Ende) — kostet nur einen Condense-Aufruf.

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
