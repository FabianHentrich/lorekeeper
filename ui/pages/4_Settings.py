"""Settings page.

Central place for editing runtime-tunable LoreKeeper parameters. Changes are
persisted to the config/settings.local.yaml sidecar and applied in place to running
services (Retriever, Reranker, ConversationManager, LLM providers).

Chunking settings persist immediately but only take effect after the next
reindex — the UI renders an explicit warning for that tab.
"""

import requests
import streamlit as st
from admin_gate import api_url, require_admin
from backend import raise_for_status
from shutdown import render_stop_button

st.set_page_config(page_title="Settings — LoreKeeper", page_icon="🛠", layout="wide")

require_admin()
with st.sidebar:
    render_stop_button()

API_URL = api_url()

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
    raise_for_status(r)
    return r.json()


def _put_config(updates: dict) -> dict:
    """Persist a partial update. Raises with the backend's readable reason on non-2xx."""
    r = requests.put(f"{API_URL}/config", json=updates, timeout=10)
    raise_for_status(r)
    return r.json()


def _save(section: str, current: dict, edited: dict, done_msg: str = "Gespeichert.",
          clear_all_caches: bool = False) -> None:
    """Send only the changed keys of one section, then reload the page with a success message.

    The message goes through ``_flash`` — an ``st.success`` right before ``st.rerun()`` would be
    wiped by the rerun. LLM and retrieval changes clear every cache (the chat's cached /sidebar-state
    holds the LLM timeout and the hybrid default); the other sections only need the page's own config
    cache cleared.
    """
    delta = _diff(current, edited)
    if not delta:
        st.info("Keine Änderungen.")
        return
    try:
        _put_config({section: delta})
    except Exception as e:
        st.error(f"Speichern fehlgeschlagen — {e}")
        return
    if clear_all_caches:
        st.cache_data.clear()
    else:
        _fetch_config.clear()
    st.session_state["_flash"] = done_msg
    st.rerun()


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


def _in(value, lo, hi, name: str):
    """``value`` clamped to the widget's range, with a warning when it was outside.

    The config has no range checks, so a value set by hand in ``settings.yaml`` (e.g. ``top_k: 60``)
    made the slider raise and the whole page fail (roadmap 13.6). Saving the section writes the shown value.
    """
    if lo <= value <= hi:
        return value
    shown = min(max(value, lo), hi)
    st.warning(f"`{name}` = {value} aus `settings.yaml` liegt außerhalb von {lo}–{hi} — hier als {shown} "
               "angezeigt; Speichern übernimmt diesen Wert.")
    return shown


if _flash := st.session_state.pop("_flash", None):
    st.success(_flash)

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
            "Kandidaten (Top-K)", 1, 50, value=_in(r["top_k"], 1, 50, "retrieval.top_k"),
            help="Wie viele Chunks der Vectorstore liefert (Bi-Encoder Recall).",
            key="ret_top_k",
        )
        new_score_threshold = st.slider(
            "Score-Threshold (Cosine)", 0.0, 0.9, step=0.05,
            value=_in(r["score_threshold"], 0.0, 0.9, "retrieval.score_threshold"),
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
        # Fixed maxima: a max that follows another slider makes a new widget, which jumped back to its
        # default when "Kandidaten" changed — and the save then sent only top_k (roadmap 13.6)
        new_top_k_rerank = st.slider(
            "Finale Chunks (nach Reranking)", 1, 50,
            value=_in(rr["top_k_rerank"], 1, 50, "retrieval.reranking.top_k_rerank"),
            help="Wie viele Chunks der Cross-Encoder am Ende auswählt.",
            key="ret_top_k_rerank", disabled=not new_rerank_enabled,
        )
        new_max_per_source = st.slider(
            "Max. Chunks pro Quelle (Soft-Cap)", 0, 50,
            value=_in(rr["max_per_source"], 0, 50, "retrieval.reranking.max_per_source"),
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
            "BM25-Anteil (RRF)", 0.0, 1.0, step=0.05,
            value=_in(rh["bm25_weight"], 0.0, 1.0, "retrieval.hybrid.bm25_weight"),
            help="0.0 = rein Vektor · 1.0 = rein BM25 · 0.3 = leicht Vektor-dominiert.",
            key="ret_bm25_weight", disabled=not new_hybrid_enabled,
        )
    with col5:
        new_bm25_top_k = st.slider(
            "BM25-Kandidaten", 1, 50, value=_in(rh["bm25_top_k"], 1, 50, "retrieval.hybrid.bm25_top_k"),
            help="Wie viele BM25-Treffer in die RRF-Fusion eingehen.",
            key="ret_bm25_top_k", disabled=not new_hybrid_enabled,
        )

    if st.button("💾 Retrieval speichern", type="primary", key="save_retrieval"):
        if new_top_k_rerank > new_top_k or new_max_per_source > new_top_k_rerank:
            st.error("Finale Chunks dürfen nicht über den Kandidaten liegen, Max. pro Quelle nicht über "
                     "den finalen Chunks.")
        else:
            _save("retrieval", r, {
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
            }, clear_all_caches=True)


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
        new_ol_temp = st.slider("Temperature", 0.0, 2.0, step=0.05,
                                value=_in(ol["temperature"], 0.0, 2.0, "llm.ollama.temperature"),
                                key="llm_ol_temp")
        new_ol_top_p = st.slider("Top-P", 0.0, 1.0, step=0.05,
                                 value=_in(ol["top_p"], 0.0, 1.0, "llm.ollama.top_p"),
                                 key="llm_ol_top_p")
        new_ol_max_tokens = st.number_input("Max Tokens", min_value=64, max_value=32768,
                                            value=_in(ol["max_tokens"], 64, 32768, "llm.ollama.max_tokens"),
                                            step=64, key="llm_ol_maxtok")
        new_ol_timeout = st.number_input("Timeout (s)", min_value=10, max_value=3600,
                                         value=_in(ol["timeout"], 10, 3600, "llm.ollama.timeout"),
                                         step=10, key="llm_ol_timeout")

    with col_g:
        st.markdown("**☁️ Gemini**")
        ge = cfg["llm"]["gemini"]
        st.caption(f"Modell: `{ge['model']}`")
        new_ge_temp = st.slider("Temperature", 0.0, 2.0, step=0.05,
                                value=_in(ge["temperature"], 0.0, 2.0, "llm.gemini.temperature"),
                                key="llm_ge_temp")
        new_ge_top_p = st.slider("Top-P", 0.0, 1.0, step=0.05,
                                 value=_in(ge["top_p"], 0.0, 1.0, "llm.gemini.top_p"),
                                 key="llm_ge_top_p")
        new_ge_max_tokens = st.number_input("Max Tokens", min_value=64, max_value=32768,
                                            value=_in(ge["max_tokens"], 64, 32768, "llm.gemini.max_tokens"),
                                            step=64, key="llm_ge_maxtok")
        new_ge_timeout = st.number_input("Timeout (s)", min_value=5, max_value=600,
                                         value=_in(ge["timeout"], 5, 600, "llm.gemini.timeout"),
                                         step=5, key="llm_ge_timeout")

    if st.button("💾 LLM-Parameter speichern", type="primary", key="save_llm"):
        _save("llm", cfg["llm"], {
            "fallback_enabled": new_fallback,
            "ollama": {
                "temperature": new_ol_temp, "top_p": new_ol_top_p,
                "max_tokens": int(new_ol_max_tokens), "timeout": int(new_ol_timeout),
            },
            "gemini": {
                "temperature": new_ge_temp, "top_p": new_ge_top_p,
                "max_tokens": int(new_ge_max_tokens), "timeout": int(new_ge_timeout),
            },
        }, clear_all_caches=True)


# ─── Conversation ──────────────────────────────────────────────────────
with tab_conv:
    st.subheader("Conversation")
    st.caption("Wirkt sofort auf neue Queries und den Session-GC.")

    c = cfg["conversation"]
    new_window = st.slider(
        "Context-Window (QA-Paare)", 1, 20, value=_in(c["window_size"], 1, 20, "conversation.window_size"),
        help="Wie viele letzte Q&A-Paare beim Umschreiben von Folgefragen (Condense) "
             "berücksichtigt werden. Der Antwort-Prompt sieht nur das letzte Paar "
             "(answer_history_pairs, nur per settings.yaml).",
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
        "Session-Timeout (Minuten)", 5, 240,
        value=_in(c["session_timeout_minutes"], 5, 240, "conversation.session_timeout_minutes"),
        help="Idle-Sessions werden nach dieser Zeit vom GC entfernt.",
        key="conv_timeout",
    )

    if st.button("💾 Conversation speichern", type="primary", key="save_conv"):
        _save("conversation", c, {
            "window_size": new_window,
            "condense_question": new_condense,
            "session_timeout_minutes": new_timeout,
        })


# ─── Ingestion (Chunking) ──────────────────────────────────────────────
_STRATEGIES = ["heading_aware", "recursive", "fixed_size"]

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
        options=_STRATEGIES,
        index=_STRATEGIES.index(ch["strategy"]) if ch["strategy"] in _STRATEGIES else 0,
        help="heading_aware = splittet an Markdown-Headings. "
             "recursive = hierarchisches Splitten. "
             "fixed_size = stumpfe Fixgrößen-Chunks.",
        key="chunk_strategy",
    )
    new_max = st.number_input(
        "Max Chunk-Size (Tokens)", min_value=50, max_value=2048,
        value=_in(ch["max_chunk_size"], 50, 2048, "chunking.max_chunk_size"), step=16,
        help="Obere Grenze pro Chunk. e5-base hat 512-Token-Limit — darunter bleiben.",
        key="chunk_max",
    )
    new_overlap = st.number_input(
        "Chunk-Overlap (Tokens)", min_value=0, max_value=200,
        value=_in(ch["chunk_overlap"], 0, 200, "chunking.chunk_overlap"), step=5,
        help="Überlappung zwischen aufeinanderfolgenden Chunks. Nur für Fließtext, "
             "Tabellen erhalten kein Overlap. Muss kleiner als die Max Chunk-Size sein.",
        key="chunk_overlap",
    )
    new_min = st.number_input(
        "Min Chunk-Size (Tokens)", min_value=1, max_value=200,
        value=_in(ch["min_chunk_size"], 1, 200, "chunking.min_chunk_size"), step=1,
        help="Kleinere Chunks werden mit dem vorigen zusammengelegt (nur unter derselben Überschrift).",
        key="chunk_min",
    )

    if st.button("💾 Chunking speichern", type="primary", key="save_chunking"):
        _save("chunking", ch, {
            "strategy": new_strategy,
            "max_chunk_size": int(new_max),
            "chunk_overlap": int(new_overlap),
            "min_chunk_size": int(new_min),
        }, done_msg="Gespeichert. Reindex auslösen, damit die Änderungen greifen.")


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
            f"collection: {cfg['vectorstore']['collection_name']}",
            language="yaml",
        )
