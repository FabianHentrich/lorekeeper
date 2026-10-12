# Data Flow

This document provides visual representations of LoreKeeper's core pipelines and lifecycles using Mermaid diagrams. It covers the ingestion process, the hybrid retrieval query path, embed text construction, session management, and health checking.

## Table of Contents
1. [Ingestion Pipeline](#ingestion-pipeline)
2. [Query Pipeline](#query-pipeline)
3. [Embed Text vs. Stored Content](#embed-text-vs-stored-content)
4. [Session Lifecycle](#session-lifecycle)
5. [Health Check Caching](#health-check-caching)

---

## Ingestion Pipeline

```mermaid
flowchart TD
    FS[("Sources\nconfig/sources.yaml")] --> ORCH[IngestionOrchestrator]

    ORCH --> HASH{{"content_hash (SHA-256)\nvs. stored hash"}}

    HASH -->|"unchanged\n(incremental run)"| SKIP["skip\ndocuments_skipped += 1"]
    HASH -->|"new / changed\n(or full=true)"| PARSE

    PARSE{{"parser by extension"}}
    PARSE -->|".md"| MD[MarkdownParser\nFrontmatter · Headings · Obsidian syntax]
    PARSE -->|".pdf"| PDF[PDFParser\npymupdf4llm · heading-aware]
    PARSE -->|".png/.jpg/.webp"| IMG[ImageMetaParser\nFilename + full path hierarchy]

    MD --> CHUNK
    PDF --> CHUNK
    IMG --> CHUNK

    CHUNK["chunk_documents()\nheading_aware: Section → Chunk\nTables atomic (no split, no overlap)\nOversized prose → split recursively\nSmall chunks → merge"]

    CHUNK --> BATCH["pending batch\n(a file's chunks never split across batches)"]
    BATCH --> DEL["changed files:\ndelete old chunks"]
    DEL --> EMBED

    EMBED["_build_embed_text()\n① Stem | Alias1 | Alias2\n② Heading > Sub\n③ Content"]

    EMBED --> E5["e5-base\n'passage: '+text\n768-dim vector"]
    E5 --> CHROMA[("ChromaDB\nupsert:\nid · document=content\nembedding · metadata")]

    ORCH -.->|"after all files"| ORPHAN["files gone from disk:\ndelete their chunks\n(chunks_deleted)"]
    ORPHAN -.-> CHROMA

    style SKIP fill:#f5f5f5,stroke:#ccc
    style CHROMA fill:#dbeafe,stroke:#3b82f6
```

The hash check happens **before** parsing, so unchanged files cost only a hash.
`full=true` (Sources page "🔄 Alle Dokumente neu indizieren") bypasses it — needed
after chunking or parser changes, which don't change a file's hash. A file that
fails to parse keeps its previously indexed chunks.

**Important:** ChromaDB stores the original `content` (what the LLM sees later),
not the enriched `embed_text`.

---

## Query Pipeline

```mermaid
flowchart TD
    USER([User question]) --> API["POST /query/stream\n{question, session_id, role, metadata_filters, hybrid_search}"]

    API --> ACCESS["effective_filters(role)\nplayer → source_id ∈ player_visible\nAND request filter (can only narrow)\nno released source → 403"]
    ACCESS --> CM["ConversationManager\nget_or_create_session(id, role)\nsessions are bound to their role"]

    CM -->|"History present\n+ condense=true"| COND["Condense step\nrender_condense(history, question)\n→ LLM → standalone question"]
    CM -->|"New session\nor first question"| RET

    COND --> RET

    RET["Retrieval Engine"]

    RET --> VEC[("ChromaDB Vector Search\nembed_text(query) → e5-base\nCosine top_k=15")]
    RET -.->|"if hybrid=true"| BM25[("BM25 Keyword Search\nrank_bm25 top_k=15\n(lazy initialized)")]

    RET -.->|"if rules_slot"| RULES[("Rules search\ngroup = rules AND same filter\ntop 5, separate pool")]

    VEC --> RRF["Reciprocal Rank Fusion (RRF)\nMerge & rescore candidates"]
    BM25 -.->|"results"| RRF

    RRF --> RERANK["CrossEncoder Reranking (GPU via DirectML, CPU fallback)\nmmarco-mMiniLMv2-L12-H384-v1\npairs: query × chunk.content\n→ top_k_rerank=8\n+ soft cap max_per_source=3\n(diverse fill → backfill)\n+ rules slot: no rules chunk in top 8 →\nbest one ≥ -4 replaces the weakest (max 1)"]
    RULES -.->|"scored, kept apart"| RERANK

    RERANK -->|"no chunks"| NOCX["render_no_context(question)\nreturn directly"]
    RERANK -->|"chunks found"| GEN

    GEN["Generation\nrender_qa(chunks, question, history)\nhistory = last QA pair, marked 'no source'\n→ LLMProvider.generate_stream()"]

    GEN --> SSE["SSE stream\ndata: {type: token, content: ...}\ndata: {type: done, session_id, sources}"]

    SSE --> UI["Streamlit UI\ntoken-by-token · sources expander"]

    style VEC fill:#dbeafe,stroke:#3b82f6
    style BM25 fill:#dbeafe,stroke:#3b82f6
    style RULES fill:#dbeafe,stroke:#3b82f6
    style NOCX fill:#fef3c7,stroke:#f59e0b
    style UI fill:#d1fae5,stroke:#10b981
```

---

## Embed Text vs. Stored Content

```mermaid
flowchart LR
    subgraph INGEST["During Ingest"]
        direction TB
        RAW["Arkenfeld.md\n# Arkenfeld\n## Overview\n| Type | Trading city |"]
        PARSE["MarkdownParser\n→ section: heading=['Arkenfeld','Overview']\n   content='| Type | Trading city |'"]
        DCHUNK["_doc_to_chunk()\ncontent = 'Arkenfeld > Overview\n\n| Type | Trading city |'"]
        ETEXT["_build_embed_text()\n'Arkenfeld | City of Arkenfeld\n\nArkenfeld > Overview\n\n| Type | Trading city |'"]
        VEC["e5-base → 768-dim vector"]
        DB[("ChromaDB\ndocument = content ← LLM sees this\nembedding = vector(embed_text)")]

        RAW --> PARSE --> DCHUNK --> ETEXT --> VEC
        DCHUNK -->|"content"| DB
        VEC -->|"embedding"| DB
    end

    style DB fill:#dbeafe,stroke:#3b82f6
```

---

## Session Lifecycle

```mermaid
stateDiagram-v2
    [*] --> New : First message\nsession_id = null

    New --> Active : Backend generates UUID\ndone event delivers session_id

    Active --> Active : Follow-up questions with session_id\nHistory grows (max window_size=8)

    Active --> GC_Candidate : No activity\n> 60 minutes

    GC_Candidate --> Deleted : GC loop\n(every 5 minutes)

    Deleted --> [*]
```

---

## Health Check Caching

```mermaid
flowchart TD
    REQ["GET /health\n(Streamlit: on every rerun)"]

    REQ --> CACHE{{"Cache\nfresh < 30s?"}}

    CACHE -->|"yes"| FAST["cached result\n< 1ms"]
    CACHE -->|"no"| CHECK

    CHECK["ChromaDB: collection.count()\nOllama: GET /api/tags (3s timeout)\nGemini: models.get()"]

    CHECK --> UPDATE["Update cache\nts = now"]
    UPDATE --> RESP["HealthResponse\n{status, chromadb, llm}"]
    FAST --> RESP

    style FAST fill:#d1fae5,stroke:#10b981
    style CHECK fill:#fef3c7,stroke:#f59e0b
```
