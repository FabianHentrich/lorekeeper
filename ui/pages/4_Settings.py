"""Settings page.

Central place for editing runtime-tunable LoreKeeper parameters. Changes are
persisted to config/settings.yaml and applied in place to running services
(Retriever, Reranker, ConversationManager, LLM providers).

Chunking settings persist immediately but only take effect after the next
reindex — the UI renders an explicit warning for that tab.
"""

import requests
import streamlit as st

st.set_page_config(page_title="Settings — LoreKeeper", page_icon="🛠", layout="wide")

API_URL = st.session_state.get("_api_url", "http://localhost:8000")

st.title("🛠 Settings")
st.caption(
    "Zentrale Laufzeit-Einstellungen. Änderungen wirken sofort auf neue Queries — "
    "bis auf den Tab **Ingestion**, der beim nächsten Reindex greift."
)


# ─── Helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=5, show_spinner=False)
def _fetch_config(api_url: str) -> dict:
    """Retrieve the current editable settings snapshot from the backend."""
    r = requests.get(f"{api_url}/config", timeout=5)
    r.raise_for_status()
    return r.json()


def _put_config(updates: dict) -> dict:
    """Persist a partial update. Raises on non-2xx."""
    r = requests.put(f"{API_URL}/config", json=updates, timeout=10)
    r.raise_for_status()
    return r.json()


def _diff(section_current: dict, section_edited: dict) -> dict:
    """Return a dict with only the keys whose value changed. Recurses into
    nested dicts so we never send unchanged fields back to the server."""
    out: dict = {}
    for key, new_val in section_edited.items():
        old_val = section_current.get(key)
        if isinstance(new_val, dict) and isinstance(old_val, dict):
            sub = _diff(old_val, new_val)
            if sub:
                out[key] = sub
        elif new_val != old_val:
            out[key] = new_val
    return out


try:
    cfg = _fetch_config(API_URL)
except Exception as e:
    st.error(f"Backend nicht erreichbar: {e}")
    st.stop()


tab_retrieval, tab_llm, tab_conv, tab_ingest = st.tabs(
    ["🔎 Retrieval", "🧠 LLM", "💬 Conversation", "📥 Ingestion"],
)


# ─── Retrieval ─────────────────────────────────────────────────────────
with tab_retrieval:
    st.subheader("Retrieval & Reranking")
    st.caption("Wirkt sofort auf neue Queries.")

    r = cfg["retrieval"]
    rr = r["reranking"]
    rh = r["hybrid"]

    col1, col2 = st.columns(2)
    with col1:
        new_top_k = st.slider(
            "Kandidaten (Top-K)", 1, 50, value=r["top_k"],
            help="Wie viele Chunks der Vectorstore liefert (Bi-Encoder Recall).",
            key="ret_top_k",
        )
        new_score_threshold = st.slider(
            "Score-Threshold (Cosine)", 0.0, 0.9, value=r["score_threshold"], step=0.05,
            help="Chunks unter diesem Score werden verworfen. Default 0.5. "
                 "Unter 0.3 landet viel Rauschen im Kontext.",
            key="ret_score_threshold",
        )
    with col2:
        new_rerank_enabled = st.checkbox(
            "Reranking aktiv", value=rr["enabled"],
            help="Cross-Encoder wählt aus den Top-K die besten Chunks. "
                 "Deaktivieren beschleunigt, verschlechtert aber die Präzision.",
            key="ret_rerank_enabled",
        )
        new_top_k_rerank = st.slider(
            "Finale Chunks (nach Reranking)", 1, new_top_k, value=min(rr["top_k_rerank"], new_top_k),
            help="Wie viele Chunks der Cross-Encoder am Ende auswählt.",
            key="ret_top_k_rerank", disabled=not new_rerank_enabled,
        )
        new_max_per_source = st.slider(
            "Max. Chunks pro Quelle (Soft-Cap)", 0, new_top_k_rerank,
            value=min(rr["max_per_source"], new_top_k_rerank),
            help="Soft-Cap pro Datei. 0 = kein Cap.",
            key="ret_max_per_source", disabled=not new_rerank_enabled,
        )

    st.markdown("**Hybrid Search (BM25 + Vektor)**")
    col3, col4, col5 = st.columns(3)
    with col3:
        new_hybrid_enabled = st.checkbox(
            "Hybrid-Default aktiv", value=rh["enabled"],
            help="Setzt den Default für alle Queries. Die Sidebar-Checkbox kann "
                 "das pro Session überschreiben.",
            key="ret_hybrid_enabled",
        )
    with col4:
        new_bm25_weight = st.slider(
            "BM25-Anteil (RRF)", 0.0, 1.0, value=rh["bm25_weight"], step=0.05,
            help="0.0 = rein Vektor · 1.0 = rein BM25 · 0.3 = leicht Vektor-dominiert.",
            key="ret_bm25_weight", disabled=not new_hybrid_enabled,
        )
    with col5:
        new_bm25_top_k = st.slider(
            "BM25-Kandidaten", 1, 50, value=rh["bm25_top_k"],
            help="Wie viele BM25-Treffer in die RRF-Fusion eingehen.",
            key="ret_bm25_top_k", disabled=not new_hybrid_enabled,
        )

    if st.button("💾 Retrieval speichern", type="primary", key="save_retrieval"):
        edited = {
            "top_k": new_top_k,
            "score_threshold": new_score_threshold,
            "reranking": {
                "enabled": new_rerank_enabled,
                "top_k_rerank": new_top_k_rerank,
                "max_per_source": new_max_per_source,
            },
            "hybrid": {
                "enabled": new_hybrid_enabled,
                "bm25_weight": new_bm25_weight,
                "bm25_top_k": new_bm25_top_k,
            },
        }
        delta = _diff(r, edited)
        if not delta:
            st.info("Keine Änderungen.")
        else:
            try:
                _put_config({"retrieval": delta})
                _fetch_config.clear()
                st.success("Gespeichert.")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler: {e}")


# ─── LLM ───────────────────────────────────────────────────────────────
with tab_llm:
    st.subheader("LLM-Parameter")
    st.caption(
        f"Aktiver Provider: **{cfg['llm']['provider']}** · "
        "Provider-Wechsel erfolgt in der Sidebar, Modellwahl nur per `settings.yaml`."
    )

    new_fallback = st.checkbox(
        "Fallback aktiv (auf Fehler → zweiten Provider probieren)",
        value=cfg["llm"]["fallback_enabled"],
        key="llm_fallback",
    )

    col_o, col_g = st.columns(2)

    with col_o:
        st.markdown("**🖥️ Ollama**")
        ol = cfg["llm"]["ollama"]
        st.caption(f"Modell: `{ol['model']}`")
        new_ol_temp = st.slider("Temperature", 0.0, 2.0, value=ol["temperature"], step=0.05,
                                key="llm_ol_temp")
        new_ol_top_p = st.slider("Top-P", 0.0, 1.0, value=ol["top_p"], step=0.05,
                                 key="llm_ol_top_p")
        new_ol_max_tokens = st.number_input("Max Tokens", min_value=64, max_value=32768,
                                            value=ol["max_tokens"], step=64, key="llm_ol_maxtok")
        new_ol_timeout = st.number_input("Timeout (s)", min_value=10, max_value=3600,
                                         value=ol["timeout"], step=10, key="llm_ol_timeout")

    with col_g:
        st.markdown("**☁️ Gemini**")
        ge = cfg["llm"]["gemini"]
        st.caption(f"Modell: `{ge['model']}`")
        new_ge_temp = st.slider("Temperature", 0.0, 2.0, value=ge["temperature"], step=0.05,
                                key="llm_ge_temp")
        new_ge_top_p = st.slider("Top-P", 0.0, 1.0, value=ge["top_p"], step=0.05,
                                 key="llm_ge_top_p")
        new_ge_max_tokens = st.number_input("Max Tokens", min_value=64, max_value=32768,
                                            value=ge["max_tokens"], step=64, key="llm_ge_maxtok")
        new_ge_timeout = st.number_input("Timeout (s)", min_value=5, max_value=600,
                                         value=ge["timeout"], step=5, key="llm_ge_timeout")

    if st.button("💾 LLM-Parameter speichern", type="primary", key="save_llm"):
        edited = {
            "fallback_enabled": new_fallback,
            "ollama": {
                "temperature": new_ol_temp, "top_p": new_ol_top_p,
                "max_tokens": int(new_ol_max_tokens), "timeout": int(new_ol_timeout),
            },
            "gemini": {
                "temperature": new_ge_temp, "top_p": new_ge_top_p,
                "max_tokens": int(new_ge_max_tokens), "timeout": int(new_ge_timeout),
            },
        }
        delta = _diff(cfg["llm"], edited)
        if not delta:
            st.info("Keine Änderungen.")
        else:
            try:
                _put_config({"llm": delta})
                _fetch_config.clear()
                st.success("Gespeichert.")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler: {e}")


# ─── Conversation ──────────────────────────────────────────────────────
with tab_conv:
    st.subheader("Conversation")
    st.caption("Wirkt sofort auf neue Queries und den Session-GC.")

    c = cfg["conversation"]
    new_window = st.slider(
        "Context-Window (QA-Paare)", 1, 20, value=c["window_size"],
        help="Wie viele letzte Q&A-Paare beim Condense-Schritt und für den LLM-Prompt "
             "berücksichtigt werden.",
        key="conv_window",
    )
    new_condense = st.checkbox(
        "Condense-Question aktiv",
        value=c["condense_question"],
        help="Frage wird vor Retrieval mit History zu einer Standalone-Query umformuliert. "
             "Deaktivieren = schneller, aber Folgefragen ohne Kontext schlechter.",
        key="conv_condense",
    )
    new_timeout = st.slider(
        "Session-Timeout (Minuten)", 5, 240, value=c["session_timeout_minutes"],
        help="Idle-Sessions werden nach dieser Zeit vom GC entfernt.",
        key="conv_timeout",
    )

    if st.button("💾 Conversation speichern", type="primary", key="save_conv"):
        edited = {
            "window_size": new_window,
            "condense_question": new_condense,
            "session_timeout_minutes": new_timeout,
        }
        delta = _diff(c, edited)
        if not delta:
            st.info("Keine Änderungen.")
        else:
            try:
                _put_config({"conversation": delta})
                _fetch_config.clear()
                st.success("Gespeichert.")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler: {e}")


# ─── Ingestion (Chunking) ──────────────────────────────────────────────
with tab_ingest:
    st.subheader("Ingestion & Chunking")
    st.warning(
        "⚠ Änderungen wirken **nicht** auf bestehende Chunks. Nach Speichern "
        "unter **Sources → Alle Dokumente neu indizieren** einen vollen Reindex "
        "auslösen, damit die neuen Chunking-Parameter greifen."
    )

    ch = cfg["chunking"]
    new_strategy = st.selectbox(
        "Strategie",
        options=["heading_aware", "recursive", "fixed_size"],
        index=["heading_aware", "recursive", "fixed_size"].index(ch["strategy"])
            if ch["strategy"] in ("heading_aware", "recursive", "fixed_size") else 0,
        help="heading_aware = splittet an Markdown-Headings. "
             "recursive = hierarchisches Splitten. "
             "fixed_size = stumpfe Fixgrößen-Chunks.",
        key="chunk_strategy",
    )
    new_max = st.number_input(
        "Max Chunk-Size (Tokens)", min_value=50, max_value=2048,
        value=ch["max_chunk_size"], step=16,
        help="Obere Grenze pro Chunk. e5-base hat 512-Token-Limit — darunter bleiben.",
        key="chunk_max",
    )
    new_overlap = st.number_input(
        "Chunk-Overlap (Tokens)", min_value=0, max_value=200,
        value=ch["chunk_overlap"], step=5,
        help="Überlappung zwischen aufeinanderfolgenden Chunks. Nur für Fließtext, "
             "Tabellen erhalten kein Overlap.",
        key="chunk_overlap",
    )
    new_min = st.number_input(
        "Min Chunk-Size (Tokens)", min_value=1, max_value=200,
        value=ch["min_chunk_size"], step=1,
        help="Kleinere Chunks werden verworfen.",
        key="chunk_min",
    )

    if st.button("💾 Chunking speichern", type="primary", key="save_chunking"):
        edited = {
            "strategy": new_strategy,
            "max_chunk_size": int(new_max),
            "chunk_overlap": int(new_overlap),
            "min_chunk_size": int(new_min),
        }
        delta = _diff(ch, edited)
        if not delta:
            st.info("Keine Änderungen.")
        else:
            try:
                _put_config({"chunking": delta})
                _fetch_config.clear()
                st.success("Gespeichert. Reindex auslösen, damit Änderungen greifen.")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler: {e}")


# ─── Read-only info ────────────────────────────────────────────────────
st.divider()
with st.expander("Read-only: Infrastruktur (nur per .env / settings.yaml änderbar)"):
    col_e, col_v = st.columns(2)
    with col_e:
        st.markdown("**Embeddings**")
        st.code(
            f"model:  {cfg['embeddings']['model']}\n"
            f"device: {cfg['embeddings']['device']}",
            language="yaml",
        )
    with col_v:
        st.markdown("**Vectorstore**")
        st.code(
            f"mode:       {cfg['vectorstore']['mode']}\n"
            f"collection: {cfg['vectorstore']['collection_name']}",
            language="yaml",
        )
