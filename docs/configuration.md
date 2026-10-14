# Konfiguration

Die vollständige Referenz aller Einstellungen und Umgebungsvariablen. Andere Dokumente verweisen
hierher, statt Werte zu wiederholen; die Datei mit den aktuellen Werten ist `config/settings.yaml`.

## Inhaltsverzeichnis
1. [Reihenfolge und Dateien](#reihenfolge-und-dateien)
2. [Bearbeiten über die UI](#bearbeiten-über-die-ui)
3. [`ingestion`](#ingestion)
   - [Quellen — `config/sources.yaml`](#quellen--configsourcesyaml)
   - [PDF — `ingestion.pdf`](#pdf--ingestionpdf)
4. [`chunking`](#chunking)
5. [`embeddings`](#embeddings)
6. [`vectorstore`](#vectorstore)
7. [`retrieval`](#retrieval)
8. [`llm`](#llm)
9. [`conversation`](#conversation)
10. [`logging`](#logging)
11. [Umgebungsvariablen](#umgebungsvariablen)

---

## Reihenfolge und Dateien

Höchste Priorität zuerst:

```
Umgebungsvariablen  >  .env  >  config/settings.local.yaml  >  config/settings.yaml  >  Code-Defaults
```

| Datei | Versioniert | Inhalt |
|-------|-------------|--------|
| `config/settings.yaml` | ja | Basiswerte mit Kommentaren; wird nie von der App umgeschrieben |
| `config/settings.local.yaml` | nein (gitignored) | Änderungen der Settings-Seite; löschen = zurück zu `settings.yaml` |
| `config/sources.yaml` | nein (gitignored) | Quellen (Vorlage: `config/sources_example.yaml`) |
| `config/prompts.yaml` | ja | Aktive Prompts ([prompts.md](prompts.md)) |
| `config/prompts/` | nein (gitignored) | Gespeicherte Prompt-Varianten |
| `.env` | nein (gitignored) | Geheimnisse und Overrides (Vorlage: `.env.example`) |

Verschachtelte Schlüssel werden in Umgebungsvariablen mit doppeltem Unterstrich geschrieben:
`llm.ollama.model` → `LLM__OLLAMA__MODEL`. Kurzformen wie `OLLAMA_MODEL` oder `CHROMA_MODE` gibt
es **nicht**.

---

## Bearbeiten über die UI

Ein Teil der Werte ist live über die Seite **🛠 Settings** bzw. `GET/PUT /config` änderbar.
Gespeichert wird nach `config/settings.local.yaml`, zugleich werden die laufenden Dienste
angepasst — kein Neustart.

| Bereich | Über die UI | Wirkung |
|---|---|---|
| `retrieval.top_k`, `score_threshold`, `reranking.{enabled,top_k_rerank,max_per_source}`, `hybrid.*` | ✅ | nächste Frage |
| `retrieval.section_expansion.*`, `rules_slot.*`, `entry_slot.*`, `reranking.{model,backend}` | ❌ nur YAML | nächster Start |
| `llm.ollama` / `llm.gemini`: `temperature`, `top_p`, `max_tokens`, `timeout` | ✅ | nächste Frage |
| `llm.fallback_enabled` | ✅ | nächste Frage (Provider, Condense-Provider und Generator werden beim Speichern neu gebaut) |
| `conversation.{window_size,condense_question,session_timeout_minutes}` | ✅ | nächste Frage / nächster GC-Lauf |
| `conversation.answer_history_*`, `condense_model` | ❌ nur YAML | nächster Start |
| `chunking.*` | ✅ | **erst nach Neu-Indizierung** (die UI warnt) |
| `ingestion.*` (`skip_sections`, `exclude_patterns`, `pdf`) | ❌ nur YAML | nächster Ingest für geänderte Dateien; für alle erst nach **Alles neu indizieren** |
| `llm.provider` | Provider-Wechsel in der Chat-Sidebar | sofort |
| Quellen | Seite ⚙ Sources | Neu-Indizierung nötig (Freigabe `player_visible` sofort) |
| Prompts | Seite ✏ Prompts | sofort |
| `embeddings.*`, `vectorstore.*`, `logging.*`, Geheimnisse | ❌ | Neustart |

Die Allow-List steht in `src/config/manager.py` (`_EDITABLE_KEYS`); andere Schlüssel verwirft
`ConfigManager.save_settings()` stillschweigend.

---

## `ingestion`

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `supported_formats` | `list[str]` | `[".md", ".pdf", ".png", ".jpg", ".jpeg", ".webp"]` | Verarbeitete Dateiendungen |
| `exclude_patterns` | `list[str]` | siehe unten | Glob-Muster für ausgeschlossene Dateien/Ordner |
| `skip_sections` | `list[str]` | `[]` (`settings.yaml`: `["Alte Vault-Werte"]`) | Markdown-Abschnitte, die nicht indiziert werden, nach Überschrift (Groß-/Kleinschreibung egal) — bis zur nächsten Überschrift gleicher oder höherer Ebene, Überschriften in Callouts zählen nicht ([parsing.md](parsing.md#was-er-macht)). Wirkt für geänderte Dateien; nach einer Änderung der Liste **alles neu indizieren** |

Standard-Ausschlüsse: `.obsidian/*`, `.trash/*`, `*alt.md` (Sicherungskopien), `*(1).md`
(Duplikate), `*.draft.*` (Entwürfe). `.tex`-Dateien werden ohnehin nicht gelesen.

### Quellen — `config/sources.yaml`

Eine Quelle ist ein Ordner oder eine einzelne Datei und gehört zu einer der vier Gruppen `lore`,
`adventure`, `rules`, `simulation`.

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `id` | `str` | ja | Stabile ID; steht als `source_id` an jedem Chunk und begrenzt Reindex/Löschen. Freitext (Leerzeichen, `&`, `#`, `/` gehen). Neue IDs dürfen nicht leer, `.`/`..` sein oder mit Leerzeichen beginnen/enden. Umbenennen = neue Quelle (siehe unten) |
| `path` | `str` | ja | Absoluter oder relativer Pfad zu Ordner **oder** Datei |
| `group` | `lore`\|`adventure`\|`rules`\|`simulation` | ja | Gruppe für die Filter im Chat; pro Ordner über `category_map` überschreibbar |
| `default_category` | `str` | nein (bei Datei-Quellen ja) | `content_category`, wenn kein `category_map`-Eintrag passt |
| `category_map` | `dict[str, str\|dict]` | nein | Top-Level-Ordner → Kategorie, optional mit Gruppe: `npc` oder `{category: story, group: adventure}`. Groß-/Kleinschreibung egal. Nur bei Ordnern |
| `exclude_patterns` | `list[str]` | nein | Zusätzlich zu den globalen Ausschlüssen |
| `player_visible` | `bool` | nein (`false`) | Im **Spieler-Modus** durchsuchbar. Nur Quellen ohne Spielleiter-Wissen freigeben (Regelwerk, Charakterbogen). Spalte „🎲 Spieler“ auf der Sources-Seite; wirkt sofort, ohne Neu-Indizierung. Das Backend erzwingt die Freigabe ([ARCHITECTURE.md](../ARCHITECTURE.md) → 4.8) |

```yaml
# config/sources.yaml
sources:
  - id: pnp-welt
    path: C:/Users/du/Obsidian/PnP-Welt
    group: lore                          # für Ordner ohne Eintrag
    default_category: misc
    category_map:
      NPCs: npc                          # Kurzform → erbt group: lore
      Orte: location
      Gegner: enemy
      Geschichte:                        # Langform → eigene Gruppe
        category: story
        group: adventure
      Regelwerk:
        category: rules
        group: rules
  - id: pnp_regelbuch
    path: C:/Users/du/Regeln/pnp_regelbuch.pdf
    group: rules
    default_category: rules
    player_visible: true
```

**Simulation (Roadmap 6.0, Stand 13.10.):** Aus `PnP_Simulation` sind nur drei Markdown-Dateien
Einzeldatei-Quellen der Gruppe `simulation`: Begegnungsbericht (`sim-begegnungen-bericht`),
Begegnungsbaukasten (`sim-begegnungsbaukasten`) und Fähigkeitsbericht (`sim-faehigkeiten-bericht`).
HTML, JSON, CSV und die übrige Entwickler-Doku bleiben draußen. Für Spieler freigegeben ist nur der
Fähigkeitsbericht (Charakterbau); Begegnungsbericht und Baukasten nicht — sie enthalten Gegnerwerte
und Kampagnen-Kämpfe. Spieler haben keinen Quellenfilter: Bei 47 von 124 reinen Regelfragen bekommen
sie im Schnitt 2,3 von 8 Chunks aus dem Fähigkeitsbericht (selten in den Top 3).

**Quellen verwalten:** `config/sources.yaml` direkt bearbeiten, die Seite **⚙ Sources** nutzen
oder die REST-Endpoints (`GET/PUT /sources`, `POST /sources/{id}/reindex`, `DELETE /sources/{id}`,
`POST /sources/recategorize`, `POST /admin/wipe`).

**Welche Aktion wann:**
- *Recategorize* (`python -m src.ingestion.recategorize` oder Button) — schreibt nur `group` /
  `content_category` / `source_id` der vorhandenen Chunks neu. Kein Embedding, Sekunden.
- *Reindex einer Quelle* — alle Dateien der Quelle neu parsen + chunken + embedden, auch unveränderte;
  nötig nach geänderten Chunking-/Parser-Einstellungen oder `path`. Die Quelle wird nicht vorher
  gelöscht und bleibt während des Laufs durchsuchbar.
- *Inkrementell* (`POST /ingest`, „⚡ Nur Änderungen indizieren“) — alle Quellen, nur neue,
  geänderte und gelöschte Dateien. Reicht nach dem Bearbeiten von Notizen. Löscht auch die Chunks von
  Quellen, die nicht mehr in `sources.yaml` stehen — wer eine Quelle dort nur vorübergehend
  auskommentiert, muss sie danach neu indizieren (Log-Zeile `Deleted N chunks of source '…'`).
- *Alles* (`POST /ingest?full=true`, „🔄 Alle Dokumente neu indizieren“) — jede Datei; nötig nach
  Änderungen an Chunking oder PDF-Parsing.

**Eine `id` umbenennen:** Die Ingestion betrachtet nur Chunks *konfigurierter* IDs, Chunks unter
der alten ID bleiben liegen. Erst die Quelle entfernen (Sources-Seite → „Source entfernen“ löscht
ihre Chunks), dann unter neuer ID anlegen — oder die Collection leeren und neu indizieren.

### PDF — `ingestion.pdf`

Nicht in `settings.yaml` gesetzt, die Defaults gelten. OCR braucht `rapidocr_onnxruntime`
(in `requirements.txt`).

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `ocr_enabled` | `bool` | `true` | OCR für Bereiche ohne extrahierbaren Text |
| `ocr_language` | `str` | `deu` | Sprachcode (ISO 639-3) |
| `ocr_dpi` | `int` | `300` | Auflösung fürs OCR (höher = besser, langsamer) |
| `extract_images` | `bool` | `true` | Eingebettete Bilder als eigene Einträge extrahieren |
| `image_format` | `str` | `png` | Format der extrahierten Bilder |

Ablauf und Fallbacks: [parsing.md](parsing.md). Änderungen hier ändern den Datei-Hash nicht —
danach „🔄 Alle Dokumente neu indizieren“ oder die Quelle neu indizieren.

---

## `chunking`

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `strategy` | `str` | `heading_aware` | `heading_aware` \| `recursive` \| `fixed_size` |
| `max_chunk_size` | `int` | `256` | Höchstgröße in geschätzten Tokens (`len(text) / 3,5`, ≈ 900 Zeichen); gilt für Fließtext, Tabellen bleiben ganz |
| `chunk_overlap` | `int` | `30` | Überlappung beim Teilen von Fließtext; Tabellen nie. Muss **kleiner als `max_chunk_size`** sein — sonst käme das Teilen nicht voran (Speichern wird mit 422 und Begründung abgelehnt) |
| `min_chunk_size` | `int` | `20` | Kleinere Chunks werden mit dem Vorgänger zusammengelegt — **nur bei gleicher Überschrift**, sonst würden die Überschrift-Metadaten über den halben Inhalt lügen |

| Strategie | Verhalten | Geeignet für |
|-----------|-----------|--------------|
| `heading_aware` | Jede Überschrift ein Chunk, Tabellen atomar (Split nur an Zeilen, Kopf wird wiederholt), zu langer Text rekursiv geteilt | Obsidian-Vaults, PDFs |
| `recursive` | Rekursiv an `\n\n`, `\n`, `. ` | Fließtext ohne Struktur |
| `fixed_size` | Feste Wortfenster mit Überlappung | Unstrukturierter Text |

Änderungen wirken erst nach „🔄 Alle Dokumente neu indizieren“ (`POST /ingest?full=true`).

---

## `embeddings`

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `model` | `str` | `intfloat/multilingual-e5-base` | Hugging-Face-Modell; e5-Modelle bekommen automatisch `query:`/`passage:`-Präfixe |
| `device` | `str` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` (auf AMD unter Windows läuft es auf der CPU) |
| `batch_size` | `int` | `64` | Batchgröße beim Indizieren |
| `normalize` | `bool` | `true` | L2-Normalisierung (nötig für Cosine) |

Das Modell wird beim Start geladen (Warm-up im Lifespan), nicht bei der ersten Frage. Ein
Modellwechsel erfordert eine komplette Neu-Indizierung.

---

## `vectorstore`

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `persist_directory` | `str` | `./chroma_data` | Speicherort der eingebetteten ChromaDB |
| `collection_name` | `str` | `lorekeeper` | ChromaDB-Collection |
| `distance_metric` | `str` | `cosine` | Nur `cosine` liefert richtige Scores (`1 − Distanz`); `l2`/`ip` würden den `score_threshold` verfälschen. Wird beim Anlegen der Collection festgelegt — eine Abweichung steht als Warnung im Log |
| `hnsw_ef_search` | `int` (≥ 1) | `2000` | Suchbreite der HNSW-Suche. 2000 sucht bei ~4–5 k Chunks exakt, auch nachdem Re-Index-Läufe den Graphen verschlechtert haben (1000 reichte danach nicht mehr) und in jedem Prozess gleich (+~2 ms pro Abfrage); wird beim Start auf eine bestehende Collection übertragen (Log-Zeile `hnsw ef_search … → …`). Eine Änderung wirkt erst nach einem Neustart des Backends (ChromaDB übernimmt sie beim Neuladen des Index). Bei deutlich mehr Chunks: Evaluation → 🎯 Suchgenauigkeit prüfen ([evaluation.md](evaluation.md#suchgenauigkeit-prüfen)) |

ChromaDB läuft immer eingebettet im Backend-Prozess. Kein zweiter Prozess darf gleichzeitig auf
`chroma_data/` zugreifen.

---

## `retrieval`

Ablauf und Begründungen: [embedding-strategy.md](embedding-strategy.md).

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `top_k` | `int` | `20` | Kandidaten aus ChromaDB vor dem Reranking. 15 → 20 am 13.10., mit jina am 14.10. bestätigt (Messung: [embedding-strategy.md](embedding-strategy.md#warum-top_k-20)). Größere Werte kosten Rerank-Zeit (in 32er-Batches, 200 Kandidaten ~0,8 s) |
| `score_threshold` | `float` | `0.5` | Mindest-Cosine. 0,5 ≈ „ähnlich“; 0,3 liefert fast orthogonales Rauschen — nicht senken |
| `reranking.enabled` | `bool` | `true` | Cross-Encoder-Reranking |
| `reranking.model` | `str` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (`settings.yaml`: `jinaai/jina-reranker-v2-base-multilingual`) | Reranker (mehrsprachig); das Hugging-Face-Repo muss einen ONNX-Export und `tokenizer.json` enthalten. Auswahl und Messung: [embedding-strategy.md](embedding-strategy.md#warum-jina-reranker-v2-entscheidung-1410) |
| `reranking.onnx_file` | `str` | `onnx/model.onnx` (`settings.yaml`: `onnx/model_fp16.onnx`) | ONNX-Datei im Modell-Repo, die auf der GPU läuft. Der CPU-Weg nimmt immer `onnx/model.onnx` (fp32) — fp16-Exporte laden auf der CPU nicht. Wirkt beim nächsten Start |
| `reranking.backend` | `str` | `cpu` (`settings.yaml`: `directml`) | `cpu` = ONNX Runtime auf der CPU (`torch` aus älteren Configs zählt als `cpu`). `directml` = GPU über `onnxruntime-directml` (jede DX12-Karte, auch AMD), unabhängig von CPU-Last. Fehlt DirectML oder der ONNX-Export, **Warnung im Log** und CPU; fällt die GPU im Betrieb aus (Treiber-Reset, VRAM voll), wechselt der Reranker mit Warnung für den Rest der Laufzeit auf die CPU |
| `reranking.top_k_rerank` | `int` | `8` | Chunks nach dem Reranking → LLM-Kontext |
| `reranking.max_per_source` | `int` | `3` | Weicher Deckel je Datei (0 = keiner). Zwei Durchgänge: erst divers füllen, dann aus dem Überlauf nachfüllen — es kommen nie weniger als `top_k_rerank` Chunks zurück, solange Kandidaten da sind |
| `hybrid.enabled` | `bool` | `false` | Vektor + BM25 parallel, Fusion per Reciprocal Rank Fusion. Pro Anfrage über `hybrid_search` |
| `hybrid.bm25_weight` | `float` | `0.3` | BM25-Anteil in der Fusion (0 = nur Vektor, 1 = nur BM25) |
| `hybrid.bm25_top_k` | `int` | `15` | BM25-Kandidaten vor der Fusion |
| `section_expansion.enabled` | `bool` | `true` | Teilen sich ≥ 2 Treffer einer Datei eine Eltern-Überschrift mit verschiedenen Unter-Überschriften, wird die ganze Sektion in Dokumentreihenfolge übergeben (vollständige Antworten auf „welche Fähigkeiten …“) |
| `section_expansion.max_chunks` | `int` | `25` | Größere Sektionen werden nicht expandiert |
| `section_expansion.top_rank` | `int` | `3` | Nur expandieren, wenn ein Treffer der Sektion unter den ersten N liegt (0 = egal). Golden Set: Ø Chunks 13,9 → 12,6, Abschnitt-Treffer unverändert; 2/1 schnitten schon relevante Übersichten ab |
| `section_expansion.max_total_chunks` | `int` (≥ 0) | `30` | Obergrenze für das ganze Ergebnis über alle expandierten Sektionen (0 = keine). Sektionen werden in der Reihenfolge ihres besten Treffers expandiert, solange die Summe passt; eine, die nicht mehr passt, behält ihre ursprünglichen Treffer. 30 lässt eine volle Sektion samt Treffern durch und blockiert nur eine zweite gestapelte — Messung in [embedding-strategy.md](embedding-strategy.md#warum-section-expansion) |
| `rules_slot.enabled` | `bool` | `false` (`settings.yaml`: `true`) | Fragen mit Vault-Entität *und* Regelbegriff („Was bewirkt Ritterlichkeit bei Escanor?“) füllen alle Kandidaten mit der Datei der Entität. Der Slot sucht zusätzlich nur in Gruppe `rules` (mit demselben Filter wie die Anfrage, der Spieler-Modus bleibt dicht); diese Kandidaten werden mit gereranked, aber nicht in den Haupt-Pool gemischt. Fehlt in den Top-`top_k_rerank` ein `rules`-Chunk, ersetzt der beste mit Score ≥ `min_score` den schwächsten — höchstens einer, und nur ab 3 finalen Chunks (bei 1–2 würde er den Großteil des Kontexts verdrängen) |
| `rules_slot.candidates` | `int` | `8` (`settings.yaml`: `5`) | Größe der Regelwerk-Suche |
| `entry_slot.enabled` | `bool` | `false` (`settings.yaml`: `true`) | Nennt die Frage einen Tabelleneintrag (Wörter und Folgen aus 2–3 Wörtern; „Dolch-Angriff“ zählt auch als „Dolch“), kommt die Tabelle mit dieser Zeile in den Rerank-Pool — mit demselben Filter wie die Anfrage. Der Reranker entscheidet, ob sie bleibt; kein reservierter Platz. ~75 ms pro Frage. Begründung und Messung: [embedding-strategy.md](embedding-strategy.md#warum-ein-eintrags-slot) |
| `entry_slot.max_chunks` | `int` | `3` | Höchstens so viele Tabellen pro Frage (0 = aus); 1–3 maßen gleich |
| `rules_slot.min_score` | `float` | `-5.0` (`settings.yaml`: `-4.0`) | Mindest-Rerank-Score (roher Logit). Golden Set im selben Prozess: aus 132/153 Abschnitte, x001–x008 1/8 · −3: 133, 2/8 · **−4: 136, 5/8** · −5/−6/−8: 136, 5/8, aber mehr Regel-Chunks in reinen Lore-Fragen (64 → 91 → 119 → 136 von 138) |

**`metadata_filters`** (pro Anfrage) werden vor der Suche geprüft. Erlaubt ist nur, was ChromaDB
**und** die BM25-Suche gleich auswerten: Gleichheit (`"group": "rules"`), `$eq`/`$ne` mit einem
Wert, `$in` mit einer nicht-leeren Liste eines Typs, `$and` mit mindestens zwei Bedingungen aus je
genau einem Schlüssel (oben dürfen mehrere Schlüssel stehen — jeder wird eine eigene Bedingung).
`$or`, `$nin`, `$gt`/`$gte`/`$lt`/`$lte` und alles andere beantwortet die API mit **422** — BM25
würde sie sonst still ignorieren bzw. bei `$or` alle Treffer verwerfen. Die App sendet nur
Erlaubtes; `tests/integration/test_filter_contract.py` prüft, dass beide Suchwege dasselbe liefern.

**Faustregeln:**
- `top_k` mindestens doppelt so groß wie `top_k_rerank`.
- `score_threshold` gilt immer auf die **Cosine**-Scores vor der Fusion — RRF-Scores liegen bei
  höchstens ~0,016 und würden sonst komplett weggefiltert.
- Pro Anfrage überschreibbar: `top_k`, `top_k_rerank`, `max_per_source`, `hybrid_search`.

---

## `llm`

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `provider` | `str` | `ollama` | Provider beim Start: `ollama` \| `gemini` |
| `fallback_provider` | `str\|null` | `gemini` | Ausweich-Provider |
| `fallback_enabled` | `bool` | `false` | Ausweichen aktivieren (greift, wenn der Hauptprovider vor dem ersten Token scheitert) |

**`llm.ollama`**

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `base_url` | `str` | `http://127.0.0.1:11434` | Ollama-Server — `127.0.0.1`, nicht `localhost` (unter Windows ~2 s pro neuer Verbindung) |
| `model` | `str` | `qwen3:14b` | Modellname (vorher `ollama pull`) |
| `temperature` | `float` | `0.3` | Kreativität (0 = deterministisch) |
| `top_p` | `float` | `0.9` | Nucleus Sampling |
| `max_tokens` | `int` | `1024` (`settings.yaml`: `2048`) | Höchstzahl Ausgabe-Tokens |
| `timeout` | `int` | `120` (`settings.yaml`: `300`) | Timeout in Sekunden; der Chat leitet daraus sein Lese-Timeout ab |

Ollama selbst (Benutzer-Umgebungsvariablen, nicht LoreKeeper): `OLLAMA_CONTEXT_LENGTH=16384` — der
OpenAI-kompatible Endpoint ignoriert `num_ctx` pro Anfrage, ohne die Variable wird der Prompt bei
4096 Tokens still abgeschnitten; `OLLAMA_KEEP_ALIVE=1h` — sonst entlädt Ollama das Modell nach 5
Minuten und die nächste Frage wartet ~16 s; `OLLAMA_MODELS=<Modellordner der Ollama-App>` — nur nötig,
wenn die App einen eigenen Ordner nutzt: dann findet ein von LoreKeeper selbst gestartetes Ollama die
Modelle sonst nicht. `OLLAMA_NUM_PARALLEL` bleibt ungesetzt (= 1, eine Frage nach der anderen;
Begründung in [provider-strategy.md](provider-strategy.md)).

**`llm.gemini`**

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `model` | `str` | `gemini-2.5-flash` | Gemini-Modell |
| `api_key_env` | `str` | `GEMINI_API_KEY` | Name der Umgebungsvariable mit dem Key |
| `temperature` | `float` | `0.3` | Kreativität |
| `top_p` | `float` | `0.9` | Nucleus Sampling |
| `max_tokens` | `int` | `1024` (`settings.yaml`: `8192`) | Höchstzahl Ausgabe-Tokens |
| `timeout` | `int` | `30` | Timeout in Sekunden |

**Gemini-Key zur Laufzeit:** Der Key lässt sich auch in der Chat-Sidebar eintragen. Er liegt nur
im Prozessspeicher, wird nie auf Platte geschrieben und ist nach einem Neustart weg. Reihenfolge:
Laufzeit-Override (`POST /provider/gemini/key`, Body `{"api_key": "..."}`, `null`/leer löscht) →
Umgebungsvariable → `.env`. `GET /provider/gemini/status` liefert nur
`{has_key, source: env|runtime|none}`. Nach jedem Setzen/Löschen werden Provider, Condense-Provider
und Generator neu gebaut — der Key wirkt sofort für ein aktives Gemini **und** für Gemini als
Fallback. Lässt sich der Fallback nicht bauen (z.B. kein Key), steht `Fallback provider gemini
unavailable` im Log.

---

## `conversation`

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `window_size` | `int` | `8` | Frage/Antwort-**Paare** im Verlauf für das Umschreiben von Folgefragen |
| `max_context_tokens` | `int` | `4096` | Harte Grenze für diesen Verlauf (ältere Paare fallen weg) |
| `answer_history_pairs` | `int` | `1` | Paare, die der **Antwort**-Prompt sieht, als „keine Quelle“ markiert (0 = aus). Bewusst klein — mehr Verlauf verwässert die Quellen |
| `answer_history_max_chars` | `int` | `1500` | Jede Nachricht dieses Verlaufs (Frage und Antwort) wird hier gekürzt (`[…]`) |
| `condense_question` | `bool` | `true` | Folgefragen mit Rückbezug für die Suche umschreiben |
| `condense_model` | `str\|null` | `null` | Eigenes (schnelleres) Ollama-Modell fürs Umschreiben; `null` = Hauptmodell |
| `session_timeout_minutes` | `int` | `60` | Inaktive Sessions nach N Minuten löschen |
| `session_gc_interval_seconds` | `int` | `300` | Wie oft die Bereinigung läuft |

---

## `logging`

| Schlüssel | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `level` | `str` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `file` | `str\|null` | `logs/lorekeeper.log` | Logdatei; `null` = nur Konsole |
| `rotate_hour` | `int` | `6` | Eine Logdatei pro Tag; der Tag wechselt zu dieser Stunde, nicht um Mitternacht — ein Spielabend bleibt in einer Datei. Der laufende Tag steht in `file`, vergangene in `lorekeeper.<JJJJ-MM-TT>.log` (Datum = Beginn des Tages). Läuft das Backend um 6 Uhr nicht, wechselt die Datei mit der ersten Zeile danach |
| `backup_count` | `int` | `30` | So viele vergangene Tage bleiben, ältere werden gelöscht |
| `format` | `str` | `%(asctime)s [%(levelname)s] %(name)s: %(message)s` | Zeilenformat (Python-`logging`) für Konsole und Datei |
| `suppress` | `list[str]` | siehe YAML | Fremd-Logger auf ERROR setzen (der uvicorn-Zugriffslog steht fest auf WARNING). Die YAML-Liste **ersetzt** die im Code — beide gleich halten (ein Test prüft das) |

---

## Umgebungsvariablen

**LoreKeeper** (in `.env` oder als Umgebungsvariable):

| Variable | Wirkung | Beispiel |
|----------|---------|----------|
| `GEMINI_API_KEY` | Gemini-Key (Geheimnis, nur Env) | `AIza...` |
| `LOREKEEPER_ADMIN_PASSWORD` | Schützt Sources, Evaluation, Prompts, Settings und Provider-/Key-Wechsel; leer = alles offen | — |
| `LOREKEEPER_API_URL` | Backend-URL für die UI | `http://127.0.0.1:8000` |
| `LLM__PROVIDER` | `llm.provider` | `gemini` |
| `LLM__OLLAMA__MODEL` | `llm.ollama.model` | `qwen3:14b` |
| `LLM__OLLAMA__BASE_URL` | `llm.ollama.base_url` | `http://127.0.0.1:11434` |
| `LLM__GEMINI__MODEL` | `llm.gemini.model` | `gemini-2.5-flash` |
| `VECTORSTORE__COLLECTION_NAME` | `vectorstore.collection_name` | `lorekeeper` |
| `RETRIEVAL__TOP_K` | `retrieval.top_k` | `15` |
| `RETRIEVAL__RERANKING__MAX_PER_SOURCE` | `retrieval.reranking.max_per_source` | `5` |
| `CONVERSATION__WINDOW_SIZE` | `conversation.window_size` | `12` |
| `LOGGING__LEVEL` | `logging.level` | `DEBUG` |
| `PERF_UPDATE_BASELINE` | `pytest -m perf` übernimmt neue Werte als Baseline | `1` |

Jeder andere Schlüssel geht nach demselben Muster (`ABSCHNITT__UNTERSCHLÜSSEL`).

**Hugging Face** (nur als echte Umgebungsvariable — der Wert in `.env` erreicht die Modell-Bibliotheken
nicht, weil das Backend `.env` nur in die Einstellungen liest):

| Variable | Default | Wirkung |
|----------|---------|---------|
| `HF_HUB_OFFLINE` | `1` (gesetzt in `src/__init__.py`, seit 14.10.) | Modelle (Embedding, Reranker) nur aus dem lokalen Cache `~/.cache/huggingface/hub/`, keine Anfrage an Hugging Face beim Start. `0` erlaubt Downloads — nötig für ein neues oder geändertes `embeddings.model`/`reranking.model`, siehe [operations.md](operations.md#modelle-fehlen-beim-ersten-start) |

Warum offline: Ohne die Variable fragt jeder Start bei Hugging Face nach neuen Modellversionen; ist
der Hub langsam oder antwortet mit 429/500, kosten die Wiederholungen Startzeit, obwohl alle Modelle
im Cache liegen. Gilt für Backend, Ingestion- und Eval-CLI und die Tests (alle importieren `src`).

**Ollama** (Benutzer-Umgebungsvariablen von Windows): `OLLAMA_CONTEXT_LENGTH=16384`,
`OLLAMA_KEEP_ALIVE=1h`, ggf. `OLLAMA_MODELS` — siehe `llm.ollama`.
