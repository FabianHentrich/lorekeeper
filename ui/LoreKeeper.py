"""Streamlit chat page — entry point of the UI.

The role (player / game master) is picked before the chat opens; the backend enforces the player filter,
the UI's own filter is only a second guard. Answers stream from ``/query/stream``; the ``session_id`` of the
done event ties follow-ups to the backend session. Sidebar data comes from one cached ``/sidebar-state``
request. Admin pages sit behind ui/admin_gate.py. Details: docs/ui-ux.md.
"""
import json
import uuid
from pathlib import Path
from urllib.parse import quote

import requests
import roles
import streamlit as st
from admin_gate import admin_login_form, api_url, is_admin
from backend import raise_for_status


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

def _fmt_timings(t: dict) -> str:
    """Format per-query stage timings (seconds) and context size for the caption under an answer."""
    if not t:
        return ""
    parts = []
    if "first_token_ms" in t:
        parts.append(f"⏱ 1. Token {t['first_token_ms'] / 1000:.1f} s")
    parts.append(f"gesamt {t.get('total_ms', 0) / 1000:.1f} s")
    if t.get("condense_ms"):
        parts.append(f"Umformulieren {t['condense_ms'] / 1000:.1f} s")
    parts.append(f"Suche {t.get('retrieval_ms', 0) / 1000:.1f} s")
    parts.append(f"{t.get('chunks', 0)} Chunks")
    return " · ".join(parts)


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

_EMPTY_USAGE = {"tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0}
if "session_usage" not in st.session_state:
    st.session_state.session_usage = dict(_EMPTY_USAGE)

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


# ─── Role: player or game master (picked before the chat opens) ────────
def _reset_chat():
    """Fresh chat and backend session — a role change must not carry answers (or their
    condensed follow-ups) from the game-master world into player mode."""
    st.session_state.messages = []
    st.session_state.session_id = None
    st.session_state.session_usage = dict(_EMPTY_USAGE)


def _render_role_picker():
    """Two buttons before the chat: player (rulebook only) or game master (everything)."""
    st.subheader("Wer fragt?")
    col_player, col_gm = st.columns(2)
    with col_player:
        st.caption("Nur das Regelwerk — keine Lore, Abenteuer oder Spielleiter-Geheimnisse.")
        if st.button(roles.LABELS[roles.PLAYER], key="_role_player", use_container_width=True, type="primary"):
            st.session_state["_role"] = roles.PLAYER
            _reset_chat()
            st.rerun()
    with col_gm:
        st.caption("Die ganze Welt: Lore, Abenteuer, Regelwerk.")
        if st.button(roles.LABELS[roles.GAME_MASTER], key="_role_gm", use_container_width=True):
            st.session_state["_role"] = roles.GAME_MASTER
            _reset_chat()
            st.rerun()


if st.session_state.get("_role") not in roles.LABELS:
    _render_role_picker()
    st.stop()

_ROLE = st.session_state["_role"]

# ─── Sidebar (fragment: only re-renders itself, not the chat area) ────
@st.cache_data(ttl=30, show_spinner=False)
def _fetch_sidebar_state(api_url: str):
    """Health, provider, Gemini-key status, player sources and LLM timeout in one request.

    Cached for 30 s, so reruns (every click) cost no request; actions that change one of these
    values clear the cache.
    """
    return requests.get(f"{api_url}/sidebar-state", timeout=5).json()


def _sidebar_state() -> dict | None:
    """The cached /sidebar-state, or None when the backend is unreachable."""
    try:
        return _fetch_sidebar_state(api_url())
    except Exception:
        return None


@st.fragment
def _render_sidebar():
    """
    Render the isolated sidebar configuration panel. Includes provider switching,
    API key overrides, source filter checkboxes, and server status readouts.
    """
    st.header("Einstellungen")

    st.markdown(f"**Rolle:** {roles.LABELS[_ROLE]}")
    if st.button("🔄 Rolle wechseln", help="Zurück zur Rollenauswahl — startet eine neue Session."):
        st.session_state.pop("_role", None)
        _reset_chat()
        st.rerun()

    API_URL = api_url()
    _sidebar = _sidebar_state()

    # Provider-Umschaltung
    st.subheader("LLM Provider")

    current_provider = None
    current_model = None
    if _sidebar:
        pinfo = _sidebar.get("provider", {})
        current_provider = pinfo.get("provider", "unknown")
        current_model = pinfo.get("model", "unknown")

    # Switching the provider (Gemini costs money) and setting the key are admin-only
    if is_admin():
        provider_options = ["ollama", "gemini"]

        if "_selected_provider" not in st.session_state:
            if current_provider in provider_options:
                st.session_state["_selected_provider"] = current_provider

        def _on_provider_change():
            """Switch the backend provider; on failure the picker falls back to the active provider."""
            new_provider = st.session_state["_selected_provider"]
            try:
                resp = requests.post(
                    f"{API_URL}/provider",
                    json={"provider": new_provider},
                    timeout=10,
                )
                raise_for_status(resp)
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

        # Gemini API key — held in backend memory only, never written to .env
        gem_status = (_sidebar or {}).get("gemini_status", {"has_key": False, "source": "none"})
        has_key = gem_status.get("has_key", False)
        if has_key:
            src_label = {"env": "Umgebungsvariable", "runtime": "UI-Eingabe"}.get(gem_status.get("source"), "?")
            st.caption(f"🔑 Gemini-Key: ✅ ({src_label})")
        else:
            st.warning("🔑 Gemini-Key fehlt — Provider 'gemini' ist nicht nutzbar.")
        if msg := st.session_state.pop("_gemini_key_msg", None):
            st.success(msg)

        def _post_gemini_key(key: str | None, done: str):
            """Set (or with None remove) the runtime Gemini key, then rerun with ``done`` as message."""
            try:
                r = requests.post(f"{API_URL}/provider/gemini/key", json={"api_key": key}, timeout=10)
                raise_for_status(r)
            except Exception as e:
                st.error(f"Fehler: {e}")
                return
            _fetch_sidebar_state.clear()
            st.session_state["_gemini_key_msg"] = done   # survives the rerun (st.success right here wouldn't)
            st.rerun()

        with st.expander("Gemini-Key ändern" if has_key else "Gemini-Key eingeben", expanded=not has_key):
            new_key = st.text_input("API-Key", type="password", key="_gemini_key",
                                    help="Wird nur im Speicher gehalten, nicht in .env geschrieben.")
            if st.button("Key setzen", key="_gemini_set"):
                if new_key.strip():
                    _post_gemini_key(new_key.strip(), "Gemini-Key gesetzt.")
                else:
                    st.error("Kein Key eingegeben.")
            # Only a key entered here can be removed here — an env key stays until .env changes
            if gem_status.get("source") == "runtime" and st.button("UI-Key entfernen", key="_gemini_clear"):
                _post_gemini_key(None, "UI-Key entfernt.")

    else:
        if current_provider:
            st.caption(f"Aktiv: **{current_provider}** · {current_model}")
        with st.expander("🔒 Admin-Login"):
            st.caption("Provider-Wechsel, Gemini-Key und die Admin-Seiten")
            admin_login_form("admin_login_sidebar")

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

    if _ROLE == roles.GAME_MASTER:
        # Source type filter
        st.subheader("Quellen")
        st.caption("Schränkt die Suche auf bestimmte Dokumenttypen ein. Nützlich um "
                   "z.B. den Regelwerk-Zeitmagier vom Lore-NPC zu trennen.")
        _SOURCE_GROUPS = {
            "🗺️ Lore": ("lore", "Welt-Lore: NPCs, Orte, Items, Organisationen, …"),
            "📖 Abenteuer": ("adventure", "Abenteuer- und Story-Dokumente"),
            "📋 Regelwerk": ("rules", "Regelbuch, Klassen, Mechaniken, Tools"),
            "📊 Simulation": ("simulation", "Balance-Simulation: Begegnungs- und Fähigkeitswertung"),
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

    else:
        # Players: fixed to the released sources — no filter widgets that could widen the search
        st.subheader("Quellen")
        st.caption("🎲 Spieler-Modus: Suche nur in den für Spieler freigegebenen Quellen (Regelwerk).")
        st.session_state["_category_filter"] = None
        st.session_state["_content_category_filter"] = None

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
        _reset_chat()
        st.rerun()


with st.sidebar:
    _render_sidebar()


# ─── Session State ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Read sidebar outputs from session_state
API_URL = api_url()
_category_filter = st.session_state.get("_category_filter")
_content_category_filter = st.session_state.get("_content_category_filter")

_player_source_ids: list[str] = []
if _ROLE == roles.PLAYER:
    _player_source_ids = (_sidebar_state() or {}).get("player_source_ids", [])
    if not _player_source_ids:
        st.warning("Für Spieler ist noch keine Quelle freigegeben. Die Spielleitung kann das auf der "
                   "Sources-Seite in der Spalte „🎲 Spieler“ einstellen.")

# ─── Answer feedback (thumbs up/down, both roles) ─────────
def _send_feedback(msg: dict):
    """Post the rating of one answer; withdrawing it is sent as rating None (the last entry counts)."""
    rating = {1: "up", 0: "down"}.get(st.session_state.get(f"fb_{msg['id']}"))
    try:
        r = requests.post(f"{API_URL}/feedback", json={
            "message_id": msg["id"],
            "rating": rating,
            "role": _ROLE,
            "question": msg.get("question", ""),
            "answer": msg["content"],
            "sources": [src.get("file", "") for src in msg.get("sources", [])],
            "model": msg.get("model", ""),
            "search_query": msg.get("search_query", ""),
            "session_id": st.session_state.get("session_id"),
            "timings": msg.get("timings", {}),
        }, timeout=5)
        raise_for_status(r)
        if rating:
            st.toast("Danke für die Bewertung!")
    except Exception as e:
        st.toast(f"Bewertung nicht gespeichert: {e}", icon="⚠️")


def _render_feedback(msg: dict):
    """Thumbs under an answer — only answers with an id (not errors, not older messages)."""
    if msg.get("id"):
        st.feedback("thumbs", key=f"fb_{msg['id']}", on_change=_send_feedback, args=(msg,))


def _render_extras(msg: dict):
    """Everything under an assistant answer: rewritten query, tokens, timings, sources."""
    if msg.get("search_query"):
        st.caption(f"🔎 Gesucht nach: {msg['search_query']}")
    if msg.get("usage"):
        st.caption(_fmt_usage(msg["usage"]))
    if msg.get("timings"):
        st.caption(_fmt_timings(msg["timings"]))
    if msg.get("sources"):
        with st.expander("📎 Quellen"):
            _render_sources(msg["sources"])


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
        _render_extras(msg)
        if msg["role"] == "assistant":
            _render_feedback(msg)

# ─── Chat Input ───────────────────────────────────────────
_placeholder = "Frag nach den Regeln..." if _ROLE == roles.PLAYER else "Stelle eine Frage über deine Welt..."
_chat_blocked = _ROLE == roles.PLAYER and not _player_source_ids
if prompt := st.chat_input(_placeholder, disabled=_chat_blocked):
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
        "role": _ROLE,  # the backend enforces the player restriction itself (the filter below is a second guard)
    }
    _filters = roles.query_filters(_ROLE, _player_source_ids, _category_filter, _content_category_filter)
    if _filters:
        request_body["metadata_filters"] = _filters
    if st.session_state.get("_hybrid_search"):
        request_body["hybrid_search"] = True

    # Read timeout follows the configured LLM timeout: condense + answer can each
    # take up to that long before the first token arrives. From the cached
    # /sidebar-state — no extra request per message.
    _llm_timeout = (_sidebar_state() or {}).get("llm_timeout", 300)

    # Stream response
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/query/stream",
                json=request_body,
                stream=True,
                timeout=(10, 2 * _llm_timeout + 60),
            )
            raise_for_status(response)

            full_response = ""
            sources = []
            usage = {}
            timings = {}
            search_query = ""
            model_used = ""
            error_message = None
            placeholder = st.empty()

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])

                if data["type"] == "token":
                    full_response += data["content"]
                    placeholder.markdown(full_response + "▌")

                elif data["type"] == "status":
                    # e.g. "the LLM is answering another question" — replaced by the first token
                    placeholder.info(f"⏳ {data.get('content', '')}")

                elif data["type"] == "error":
                    error_message = data.get("content", "Unbekannter Fehler")
                    break

                elif data["type"] == "done":
                    st.session_state.session_id = data.get("session_id")
                    sources = data.get("sources", [])
                    model_used = data.get("model_used", "")
                    usage = data.get("usage", {}) or {}
                    timings = data.get("timings", {}) or {}
                    # Only shown when a follow-up was rewritten before retrieval
                    if data.get("search_query", prompt) != prompt:
                        search_query = data["search_query"]
                    sess_usage = data.get("session_usage")
                    if sess_usage:
                        st.session_state.session_usage = sess_usage

            if error_message:
                placeholder.error(f"Fehler vom LLM-Provider: {error_message}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Fehler vom LLM-Provider: {error_message}",
                    "sources": [],
                    "usage": {},
                })
            else:
                placeholder.markdown(full_response)
                answer = {
                    "role": "assistant",
                    "id": uuid.uuid4().hex,
                    "question": prompt,
                    "model": model_used,
                    "content": full_response,
                    "sources": sources,
                    "usage": usage,
                    "timings": timings,
                    "search_query": search_query,
                }
                st.session_state.messages.append(answer)
                _render_extras(answer)
                _render_feedback(answer)

        except requests.exceptions.ConnectionError:
            st.error("Verbindung zum Backend fehlgeschlagen. Läuft der Server?")
        except Exception as e:
            st.error(f"Fehler: {e}")
