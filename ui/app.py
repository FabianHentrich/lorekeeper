import json
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st


def _render_source(src: dict):
    """Render a single source reference with clickable link and image support."""
    source_path = src.get("source_path", "")
    file_label = src.get("file", "")
    heading = f" — {src['heading']}" if src.get("heading") else ""
    score = src.get("score", 0)

    if src.get("document_type") == "image":
        if source_path and Path(source_path).is_file():
            st.image(source_path, caption=file_label)
        else:
            st.warning(f"Bild nicht gefunden: {file_label}")
    else:
        if source_path:
            file_url = "file:///" + quote(source_path.replace("\\", "/"), safe=":/")
            st.markdown(
                f"📄 [**{file_label}{heading}**]({file_url}) (Score: {score:.2f})"
            )
        else:
            st.markdown(f"📄 **{file_label}{heading}** (Score: {score:.2f})")
        if src.get("chunk_preview"):
            st.caption(src["chunk_preview"])

st.set_page_config(page_title="LoreKeeper", page_icon="📜", layout="wide")
st.title("📜 LoreKeeper")
st.caption("Frag deine Welt.")

# ─── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("Einstellungen")

    API_URL = st.text_input("API URL", value="http://localhost:8000")

    # Provider-Umschaltung
    st.subheader("LLM Provider")
    current_provider = None
    current_model = None
    try:
        pinfo = requests.get(f"{API_URL}/provider", timeout=5).json()
        current_provider = pinfo.get("provider", "unknown")
        current_model = pinfo.get("model", "unknown")
    except Exception:
        pass

    provider_options = ["ollama", "gemini"]

    # Sync with backend on first load (before widget is created)
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
        except Exception as e:
            st.session_state["_provider_switch_error"] = str(e)
            # Revert: will take effect on next rerun
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

    st.divider()

    # Provider status (cached 30s to avoid hammering the API)
    @st.cache_data(ttl=30, show_spinner=False)
    def _fetch_health(api_url: str):
        return requests.get(f"{api_url}/health", timeout=5).json()

    @st.cache_data(ttl=60, show_spinner=False)
    def _fetch_stats(api_url: str):
        return requests.get(f"{api_url}/stats", timeout=5).json()

    try:
        health = _fetch_health(API_URL)
        status_emoji = "🟢" if health["status"] == "healthy" else "🟡"
        st.markdown(f"{status_emoji} **Status:** {health['status']}")
        st.markdown(f"ChromaDB: {'✅' if health['chromadb'] else '❌'}")
        st.markdown(f"LLM: {'✅' if health['llm'] else '❌'}")
    except Exception:
        st.markdown("🔴 **API nicht erreichbar**")

    st.divider()

    # Retrieval settings
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=20, value=10)

    # Source type filter
    st.subheader("Quellen")
    st.caption("Schränkt die Suche auf bestimmte Dokumenttypen ein. Nützlich um "
               "z.B. den Regelwerk-Zeitmagier vom Lore-NPC zu trennen.")
    _SOURCE_GROUPS = {
        "🗺️ Lore": (
            ["npc", "location", "enemy", "item", "organization", "daemon", "god", "backstory", "misc"],
            "NPCs, Orte, Gegenstände, Organisationen, Götter, Dämonen, Hintergrund",
        ),
        "📖 Abenteuer": (
            ["story"],
            "Abenteuer- und Story-Dokumente",
        ),
        "📋 Regelwerk": (
            ["tool", "rules"],
            "Spielregeln, Klassen, Mechaniken, Tools",
        ),
    }
    selected_groups = []
    selected_labels = []
    for label, (_cats, _help) in _SOURCE_GROUPS.items():
        if st.checkbox(label, value=True, key=f"src_{label}", help=_help):
            selected_groups.extend(_cats)
            selected_labels.append(label)

    # Build filter: None = no filter (all selected), $in = subset, blocked = none
    all_categories = [c for cats, _ in _SOURCE_GROUPS.values() for c in cats]
    if not selected_groups:
        st.warning("⚠️ Mindestens eine Quellenart auswählen, sonst werden Anfragen blockiert.")
        _category_filter = "__BLOCKED__"
    elif set(selected_groups) == set(all_categories):
        _category_filter = None
    else:
        _category_filter = {"$in": selected_groups}
        st.caption(f"🔍 Suche eingeschränkt auf: {', '.join(selected_labels)}")

    st.divider()

    # Index stats
    try:
        stats = _fetch_stats(API_URL)
        st.metric("Indizierte Chunks", stats["chunk_count"])
    except Exception:
        pass

    # Ingestion trigger
    if st.button("🔄 Dokumente neu indizieren"):
        try:
            resp = requests.post(f"{API_URL}/ingest", timeout=10).json()
            st.info(f"Ingestion gestartet (Job: {resp['job_id'][:8]}...)")
        except Exception as e:
            st.error(f"Fehler: {e}")

    st.divider()

    if st.button("🗑️ Neue Session"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()


# ─── Session State ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ─── Chat History ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Quellen"):
                for src in msg["sources"]:
                    _render_source(src)

# ─── Chat Input ───────────────────────────────────────────
if prompt := st.chat_input("Stelle eine Frage über deine Welt..."):
    if _category_filter == "__BLOCKED__":
        st.error("Bitte mindestens eine Quellenart in der Seitenleiste auswählen.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build request
    request_body = {
        "question": prompt,
        "session_id": st.session_state.session_id,
        "top_k": top_k,
    }
    if _category_filter is not None:
        request_body["metadata_filters"] = {"content_category": _category_filter}

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

            placeholder.markdown(full_response)

            if sources:
                with st.expander("📎 Quellen"):
                    for src in sources:
                        _render_source(src)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })

        except requests.exceptions.ConnectionError:
            st.error("Verbindung zum Backend fehlgeschlagen. Läuft der Server?")
        except Exception as e:
            st.error(f"Fehler: {e}")
