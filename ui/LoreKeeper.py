import json
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st


def _render_sources(sources: list[dict]):
    """
    Render retrieved source documents inline within the chat interface.
    Groups identical file origins and displays respective semantic chunk extracts
    beneath their document headers alongside confidence scores.
    """
    # Render images individually first
    images = [s for s in sources if s.get("document_type") == "image"]
    docs = [s for s in sources if s.get("document_type") != "image"]

    for src in images:
        source_path = src.get("source_path", "")
        file_label = src.get("file", "")
        if source_path and Path(source_path).is_file():
            st.image(source_path, caption=file_label)
        else:
            st.warning(f"Bild nicht gefunden: {file_label}")

    # Group docs by file (preserve order of first appearance)
    grouped: dict[str, list[dict]] = {}
    for src in docs:
        key = src.get("file", "")
        grouped.setdefault(key, []).append(src)

    for file_label, chunks in grouped.items():
        source_path = chunks[0].get("source_path", "")
        best_score = max(c.get("score", 0) for c in chunks)
        n = len(chunks)
        suffix = f" · {n} Chunks" if n > 1 else ""
        if source_path:
            file_url = "file:///" + quote(source_path.replace("\\", "/"), safe=":/")
            header = f"📄 [**{file_label}**]({file_url}) — Best Score: {best_score:.2f}{suffix}"
        else:
            header = f"📄 **{file_label}** — Best Score: {best_score:.2f}{suffix}"
        st.markdown(header)

        for c in chunks:
            heading = c.get("heading") or "—"
            score = c.get("score", 0)
            preview = c.get("chunk_preview", "")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ *{heading}* (Score: {score:.2f})", unsafe_allow_html=True)
            if preview:
                st.caption(preview)

def _fmt_usage(u: dict) -> str:
    """
    Format token usage statistics into a terse inline string representation
    for displaying alongside assistant messages.
    """
    if not u:
        return ""
    tin = u.get("tokens_in", 0) or 0
    tout = u.get("tokens_out", 0) or 0
    tth = u.get("tokens_thinking", 0) or 0
    parts = [f"⬇ {tin} in", f"⬆ {tout} out"]
    if tth:
        parts.append(f"🧠 {tth} think")
    return " · ".join(parts)


st.set_page_config(page_title="LoreKeeper", page_icon="📜", layout="wide")

if "session_usage" not in st.session_state:
    st.session_state.session_usage = {"tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0}

_title_col, _usage_col = st.columns([4, 1])
with _title_col:
    st.title("📜 LoreKeeper")
    st.caption("Frag deine Welt.")
with _usage_col:
    su = st.session_state.session_usage
    st.metric(
        "Session-Tokens",
        f"{(su['tokens_in'] + su['tokens_out'] + su['tokens_thinking']):,}".replace(",", "."),
        help=f"In: {su['tokens_in']} · Out: {su['tokens_out']} · Thinking: {su['tokens_thinking']}",
    )

# ─── Sidebar (fragment: only re-renders itself, not the chat area) ────
@st.cache_data(ttl=30, show_spinner=False)
def _fetch_sidebar_state(api_url: str):
    """
    Poll the backend health and current configuration states periodically.
    Cached briefly to prevent flickering during rapid page redraws.
    """
    return requests.get(f"{api_url}/sidebar-state", timeout=5).json()


@st.fragment
def _render_sidebar():
    """
    Render the isolated sidebar configuration panel. Includes provider switching,
    API key overrides, source filter checkboxes, and server status readouts.
    """
    st.header("Einstellungen")

    API_URL = st.text_input("API URL", value="http://localhost:8000")
    st.session_state["_api_url"] = API_URL

    try:
        _sidebar = _fetch_sidebar_state(API_URL)
    except Exception:
        _sidebar = None

    # Provider-Umschaltung
    st.subheader("LLM Provider")

    current_provider = None
    current_model = None
    if _sidebar:
        pinfo = _sidebar.get("provider", {})
        current_provider = pinfo.get("provider", "unknown")
        current_model = pinfo.get("model", "unknown")

    provider_options = ["ollama", "gemini"]

    if "_selected_provider" not in st.session_state:
        if current_provider in provider_options:
            st.session_state["_selected_provider"] = current_provider

    def _on_provider_change():
        new_provider = st.session_state["_selected_provider"]
        try:
            resp = requests.post(
                f"{API_URL}/provider",
                json={"provider": new_provider},
                timeout=10,
            )
            resp.raise_for_status()
            st.session_state["_provider_switch_ok"] = True
            _fetch_sidebar_state.clear()
        except Exception as e:
            st.session_state["_provider_switch_error"] = str(e)
            st.session_state["_selected_provider"] = current_provider

    selected_provider = st.selectbox(
        "Provider",
        options=provider_options,
        key="_selected_provider",
        format_func=lambda x: f"{'🖥️' if x == 'ollama' else '☁️'} {x.capitalize()}",
        on_change=_on_provider_change,
    )

    if st.session_state.pop("_provider_switch_ok", False):
        st.success(f"Provider gewechselt zu **{selected_provider}**")
    if err := st.session_state.pop("_provider_switch_error", None):
        st.error(f"Wechsel fehlgeschlagen: {err}")

    if current_model:
        st.caption(f"Aktiv: **{current_model}**")

    # Gemini API key
    gem_status = _sidebar.get("gemini_status", {}) if _sidebar else {"has_key": False, "source": "none"}

    if gem_status.get("has_key"):
        src_label = {"env": "Umgebungsvariable", "runtime": "UI-Eingabe"}.get(gem_status.get("source"), "?")
        st.caption(f"🔑 Gemini-Key: ✅ ({src_label})")
        with st.expander("Gemini-Key überschreiben"):
            new_key_override = st.text_input(
                "Neuer API-Key", type="password", key="_gemini_key_override",
                help="Wird nur im Speicher gehalten, nicht in .env geschrieben.",
            )
            if st.button("Key setzen", key="_gemini_set_override"):
                try:
                    r = requests.post(f"{API_URL}/provider/gemini/key",
                                      json={"api_key": new_key_override}, timeout=10)
                    r.raise_for_status()
                    st.success("Gemini-Key aktualisiert.")
                    _fetch_sidebar_state.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler: {e}")
    else:
        st.warning("🔑 Gemini-Key fehlt — Provider 'gemini' ist nicht nutzbar.")
        new_key = st.text_input(
            "Gemini API-Key eingeben", type="password", key="_gemini_key_new",
            help="Wird nur im Speicher gehalten, nicht in .env geschrieben.",
        )
        if st.button("Key speichern", key="_gemini_set_new"):
            if not new_key.strip():
                st.error("Kein Key eingegeben.")
            else:
                try:
                    r = requests.post(f"{API_URL}/provider/gemini/key",
                                      json={"api_key": new_key}, timeout=10)
                    r.raise_for_status()
                    st.success("Gemini-Key gesetzt.")
                    _fetch_sidebar_state.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler: {e}")

    st.divider()

    # Provider status
    if _sidebar:
        health = _sidebar.get("health", {})
        status_emoji = "🟢" if health.get("status") == "healthy" else "🟡"
        st.markdown(f"{status_emoji} **Status:** {health.get('status', '?')}")
        st.markdown(f"ChromaDB: {'✅' if health.get('chromadb') else '❌'}")
        st.markdown(f"LLM: {'✅' if health.get('llm') else '❌'}")
        if not health.get("sources_configured", True):
            st.warning(
                "Keine Quellen konfiguriert. Lege unter **Sources** "
                "mindestens einen Pfad zu deinen Dokumenten an und starte "
                "dann die Indizierung."
            )
    else:
        st.markdown("🔴 **API nicht erreichbar**")

    st.divider()

    # Source type filter
    st.subheader("Quellen")
    st.caption("Schränkt die Suche auf bestimmte Dokumenttypen ein. Nützlich um "
               "z.B. den Regelwerk-Zeitmagier vom Lore-NPC zu trennen.")
    _SOURCE_GROUPS = {
        "🗺️ Lore": ("lore", "Welt-Lore: NPCs, Orte, Items, Organisationen, …"),
        "📖 Abenteuer": ("adventure", "Abenteuer- und Story-Dokumente"),
        "📋 Regelwerk": ("rules", "Regelbuch, Klassen, Mechaniken, Tools"),
    }
    selected_groups = []
    selected_labels = []
    for label, (group_id, _help) in _SOURCE_GROUPS.items():
        if st.checkbox(label, value=True, key=f"src_{label}", help=_help):
            selected_groups.append(group_id)
            selected_labels.append(label)

    all_groups = [g for g, _ in _SOURCE_GROUPS.values()]
    if not selected_groups:
        st.warning("⚠️ Mindestens eine Quellenart auswählen, sonst werden Anfragen blockiert.")
        st.session_state["_category_filter"] = "__BLOCKED__"
    elif set(selected_groups) == set(all_groups):
        st.session_state["_category_filter"] = None
    else:
        st.session_state["_category_filter"] = {"$in": selected_groups}
        st.caption(f"🔍 Suche eingeschränkt auf: {', '.join(selected_labels)}")

    # Optional: further narrow by content_category
    available_cats = _sidebar.get("available_categories", []) if _sidebar else []
    st.session_state["_content_category_filter"] = None
    if available_cats:
        with st.expander("Kategorie-Filter (optional)"):
            selected_cats = st.multiselect(
                "Kategorien",
                options=available_cats,
                default=available_cats,
                help="Schränkt die Suche zusätzlich auf bestimmte Inhaltskategorien ein.",
            )
            if selected_cats and set(selected_cats) != set(available_cats):
                st.session_state["_content_category_filter"] = {"$in": selected_cats}
                st.caption(f"🏷 Kategorie: {', '.join(selected_cats)}")

    st.divider()

    st.subheader("Suche")
    _hybrid = st.checkbox(
        "🔍 Hybrid Search (BM25 + Vektor)",
        value=False,
        help="Kombiniert Keyword-Suche mit Vektor-Suche. Besser für exakte Begriffe (Namen, Regeln).",
        key="_hybrid_search_cb",
    )
    st.session_state["_hybrid_search"] = _hybrid

    st.divider()

    if st.button("🗑️ Neue Session"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.session_usage = {"tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0}
        st.rerun()


with st.sidebar:
    _render_sidebar()


# ─── Session State ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Read sidebar outputs from session_state
API_URL = st.session_state.get("_api_url", "http://localhost:8000")
_category_filter = st.session_state.get("_category_filter")
_content_category_filter = st.session_state.get("_content_category_filter")

# ─── Chat History ─────────────────────────────────────────
_MAX_VISIBLE_MESSAGES = 50
_all_messages = st.session_state.messages
if len(_all_messages) > _MAX_VISIBLE_MESSAGES:
    _hidden = len(_all_messages) - _MAX_VISIBLE_MESSAGES
    with st.expander(f"Ältere Nachrichten ({_hidden} ausgeblendet)"):
        for msg in _all_messages[:_hidden]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    _visible = _all_messages[_hidden:]
else:
    _visible = _all_messages

for msg in _visible:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("usage"):
            st.caption(_fmt_usage(msg["usage"]))
        if msg.get("sources"):
            with st.expander("📎 Quellen"):
                _render_sources(msg["sources"])

# ─── Chat Input ───────────────────────────────────────────
if prompt := st.chat_input("Stelle eine Frage über deine Welt..."):
    if _category_filter == "__BLOCKED__":
        st.error("Bitte mindestens eine Quellenart in der Seitenleiste auswählen.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build request (retrieval params use server defaults from settings.yaml)
    request_body = {
        "question": prompt,
        "session_id": st.session_state.session_id,
    }
    _filters = {}
    if _category_filter is not None:
        _filters["group"] = _category_filter
    if _content_category_filter is not None:
        _filters["content_category"] = _content_category_filter
    if _filters:
        request_body["metadata_filters"] = _filters
    if st.session_state.get("_hybrid_search"):
        request_body["hybrid_search"] = True

    # Stream response
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/query/stream",
                json=request_body,
                stream=True,
                timeout=180,
            )
            response.raise_for_status()

            full_response = ""
            sources = []
            usage = {}
            placeholder = st.empty()

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])

                if data["type"] == "token":
                    full_response += data["content"]
                    placeholder.markdown(full_response + "▌")

                elif data["type"] == "done":
                    st.session_state.session_id = data.get("session_id")
                    sources = data.get("sources", [])
                    usage = data.get("usage", {}) or {}
                    sess_usage = data.get("session_usage")
                    if sess_usage:
                        st.session_state.session_usage = sess_usage

            placeholder.markdown(full_response)

            if usage:
                st.caption(_fmt_usage(usage))

            if sources:
                with st.expander("📎 Quellen"):
                    _render_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "usage": usage,
            })

        except requests.exceptions.ConnectionError:
            st.error("Verbindung zum Backend fehlgeschlagen. Läuft der Server?")
        except Exception as e:
            st.error(f"Fehler: {e}")
