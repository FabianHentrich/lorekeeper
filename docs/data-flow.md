# Datenfluss

Die Abläufe von LoreKeeper als Mermaid-Diagramme: Ingestion, Beantwortung einer Frage,
Embed-Text vs. gespeicherter Inhalt, Lebenszyklus einer Session und Health-Check. Begründungen
für die einzelnen Stufen: [embedding-strategy.md](embedding-strategy.md); Parameter:
[configuration.md](configuration.md).

## Inhaltsverzeichnis
1. [Ingestion](#ingestion)
2. [Beantwortung einer Frage](#beantwortung-einer-frage)
3. [Embed-Text vs. gespeicherter Inhalt](#embed-text-vs-gespeicherter-inhalt)
4. [Lebenszyklus einer Session](#lebenszyklus-einer-session)
5. [Health-Check](#health-check)

---

## Ingestion

```mermaid
flowchart TD
    FS[("Quellen\nconfig/sources.yaml")] --> ORCH[IngestionOrchestrator]

    ORCH --> HASH{{"content_hash (SHA-256)\nvs. gespeicherter Hash"}}

    HASH -->|"unverändert\n(inkrementell)"| SKIP["überspringen\ndocuments_skipped += 1"]
    HASH -->|"neu / geändert\n(oder full=true)"| PARSE

    PARSE{{"Parser nach Endung"}}
    PARSE -->|".md"| MD[MarkdownParser\nFrontmatter · Überschriften · Obsidian-Syntax]
    PARSE -->|".pdf"| PDF[PDFParser\npymupdf4llm · OCR · Inhaltsverzeichnis]
    PARSE -->|".png/.jpg/.webp"| IMG[ImageMetaParser\nDateiname + Ordnerpfad]

    MD --> CHUNK
    PDF --> CHUNK
    IMG --> CHUNK

    CHUNK["chunk_documents()\nheading_aware: Abschnitt → Chunk\nTabellen atomar (kein Split, kein Overlap)\nzu langer Text → rekursiv teilen\nkleine Chunks → mit Nachbarn gleicher Überschrift"]

    CHUNK --> BATCH["Batch\n(Chunks einer Datei nie über Batches verteilt)"]
    BATCH --> DEL["geänderte Dateien:\nalte Chunks löschen"]
    DEL --> EMBED

    EMBED["_build_embed_text()\n① Dateiname | Alias1 | Alias2\n② Überschrift > Unter\n③ Inhalt"]

    EMBED --> E5["e5-base\n'passage: '+Text\n768-dim. Vektor"]
    E5 --> CHROMA[("ChromaDB\nupsert:\nid · document=Inhalt\nembedding · Metadaten")]

    ORCH -.->|"nach allen Dateien"| ORPHAN["von der Platte verschwunden:\nChunks löschen\n(chunks_deleted)"]
    ORPHAN -.-> CHROMA

    style SKIP fill:#f5f5f5,stroke:#ccc
    style CHROMA fill:#dbeafe,stroke:#3b82f6
```

Der Hash-Vergleich kommt **vor** dem Parsen; unveränderte Dateien kosten nur den Hash.
`full=true` („🔄 Alle Dokumente neu indizieren“) umgeht ihn — nötig nach Änderungen an Chunking
oder Parsern, die den Hash einer Datei nicht ändern. Eine Datei, die nicht geparst werden kann,
behält ihre bisherigen Chunks. Ingestion läuft immer als Background-Job (`job_id` sofort zurück,
Fortschritt über `/ingest/status/{id}` bzw. `/ingest/active`).

**Wichtig:** ChromaDB speichert den ursprünglichen Inhalt (den später das LLM sieht), nicht den
angereicherten Embed-Text.

---

## Beantwortung einer Frage

```mermaid
flowchart TD
    USER([Frage]) --> API["POST /query/stream\n{question, session_id, role, metadata_filters, hybrid_search}"]

    API --> ACCESS["effective_filters(role)\nSpieler → source_id ∈ player_visible\nAND Anfrage-Filter (nur einengen)\nkeine Freigabe → 403"]
    ACCESS --> CM["ConversationManager\nget_or_create_session(id, role)\nSessions sind an die Rolle gebunden"]

    CM -->|"Verlauf da + Rückbezug\n(needs_condense)"| COND["Condense\nrender_condense(Verlauf, Frage)\n→ LLM → eigenständige Frage"]
    CM -->|"eigenständige Frage\noder erste Frage"| RET

    COND --> RET

    RET["Retrieval"]

    RET --> VEC[("Vektorsuche\ne5-base 'query: '\nCosine top_k=15, ≥ 0,5")]
    RET -.->|"wenn hybrid"| BM25[("BM25\nrank_bm25 top_k=15\n(lazy gebaut)")]
    RET -.->|"wenn rules_slot"| RULES[("Regelwerk-Suche\ngroup = rules UND gleicher Filter\ntop 5, eigener Pool")]

    VEC --> RRF["Reciprocal Rank Fusion\n(nur bei hybrid)"]
    BM25 -.->|"Treffer"| RRF

    RRF --> RERANK["Cross-Encoder-Rerank (GPU via DirectML, CPU als Rückfall)\nmmarco-mMiniLMv2-L12-H384-v1\n→ top_k_rerank=8\n+ Deckel max_per_source=3 (divers füllen → nachfüllen)\n+ Regelwerk-Slot: kein rules-Chunk in den Top-8 →\nbester mit Score ≥ −4 ersetzt den schwächsten (max. 1)"]
    RULES -.->|"mitbewertet, getrennt"| RERANK

    RERANK --> EXP["Section Expansion\n≥ 2 Treffer unter einer Eltern-Überschrift,\neiner davon unter den ersten 3 → ganze Sektion (≤ 25 Chunks)"]

    EXP -->|"keine Chunks"| NOCX["render_no_context(Frage)\ndirekt zurück"]
    EXP -->|"Chunks"| GEN

    GEN["Generierung\nrender_qa(Chunks, Frage, Verlauf)\nVerlauf = letztes Q/A-Paar, als 'keine Quelle' markiert\n→ Provider.generate_stream()"]

    GEN --> SSE["SSE-Stream\ndata: {type: status} (nur wenn das LLM noch andere Fragen beantwortet)\ndata: {type: token, content}\ndata: {type: done, session_id, sources, timings, usage}"]

    SSE --> UI["Streamlit\nToken für Token · Quellen · 👍/👎"]

    style VEC fill:#dbeafe,stroke:#3b82f6
    style BM25 fill:#dbeafe,stroke:#3b82f6
    style RULES fill:#dbeafe,stroke:#3b82f6
    style NOCX fill:#fef3c7,stroke:#f59e0b
    style UI fill:#d1fae5,stroke:#10b981
```

Gespeichert wird die Frage erst nach einer vollständigen Antwort; bricht der Stream ab, bleibt der
Verlauf unverändert.

---

## Embed-Text vs. gespeicherter Inhalt

```mermaid
flowchart LR
    subgraph INGEST["Beim Indizieren"]
        direction TB
        RAW["Arkenfeld.md\n# Arkenfeld\n## Überblick\n| Typ | Handelsstadt |"]
        PARSE["MarkdownParser\n→ Abschnitt: heading=['Arkenfeld','Überblick']\n   content='| Typ | Handelsstadt |'"]
        DCHUNK["_doc_to_chunk()\ncontent = 'Arkenfeld > Überblick\n\n| Typ | Handelsstadt |'"]
        ETEXT["_build_embed_text()\n'Arkenfeld | Stadt Arkenfeld\n\nArkenfeld > Überblick\n\n| Typ | Handelsstadt |'"]
        VEC["e5-base → 768-dim. Vektor"]
        DB[("ChromaDB\ndocument = Inhalt ← sieht das LLM\nembedding = Vektor(Embed-Text)")]

        RAW --> PARSE --> DCHUNK --> ETEXT --> VEC
        DCHUNK -->|"Inhalt"| DB
        VEC -->|"Embedding"| DB
    end

    style DB fill:#dbeafe,stroke:#3b82f6
```

---

## Lebenszyklus einer Session

```mermaid
stateDiagram-v2
    [*] --> Neu : erste Nachricht\nsession_id = null

    Neu --> Aktiv : Backend erzeugt UUID\ndone-Event liefert session_id

    Aktiv --> Aktiv : Folgefragen mit session_id und gleicher Rolle\nVerlauf wächst (Fenster 8 Paare)

    Aktiv --> Neu : Anfrage mit anderer Rolle\n→ neue Session, alte bleibt unberührt

    Aktiv --> Abgelaufen : keine Aktivität\n> 60 Minuten

    Abgelaufen --> Gelöscht : GC-Schleife\n(alle 5 Minuten)

    Gelöscht --> [*]
```

Sessions liegen nur im Speicher und überleben keinen Neustart. „🔄 Rolle wechseln“ und „Neue
Session“ in der UI starten ebenfalls eine neue Session.

---

## Health-Check

```mermaid
flowchart TD
    LOOP["Hintergrund-Schleife im Backend\nalle 60 s"]
    LOOP --> CHECK["ChromaDB: collection.count()\nOllama: GET /api/tags\nGemini: models.get()"]
    CHECK --> CACHE[("Health-Cache\n{chroma, llm}")]
    CHECK -.->|"nur bei Änderung"| LOG["Log: Health changed …\n(WARNING bei Ausfall, INFO bei Erholung)"]

    REQ1["GET /health"] --> CACHE
    REQ2["GET /sidebar-state\n(UI, 30 s gecacht)"] --> CACHE
    CACHE --> RESP["status: healthy | degraded\nchromadb · llm · sources_configured"]

    style CACHE fill:#dbeafe,stroke:#3b82f6
    style CHECK fill:#fef3c7,stroke:#f59e0b
```

Keine Anfrage wartet auf einen Health-Check: `/health` und `/sidebar-state` lesen nur den Cache,
den die Schleife aktuell hält. Ins Log schreibt die Schleife nur Wechsel des Zustands, nicht jeden
Durchlauf.
