import json
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st


def _render_sources(sources: list[dict]):
    """Render sources grouped by file: one entry per document, chunks listed below."""
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
    with st.expander("⚙️ Erweitert: Retrieval-Tuning"):
        top_k = st.slider(
            "Kandidaten (Top-K)",
            min_value=1, max_value=50, value=15,
            help="Wie viele Chunks der Vectorstore liefert (Bi-Encoder Recall). "
                 "Mehr = höhere Wahrscheinlichkeit, dass die richtige Stelle dabei ist.",
        )
        top_k_rerank = st.slider(
            "Finale Chunks für den LLM (nach Reranking)",
            min_value=1, max_value=top_k, value=min(8, top_k),
            help="Wie viele Chunks der Cross-Encoder am Ende auswählt und in den Prompt packt. "
                 "Mehr = mehr Kontext, aber höhere Latenz und Token-Kosten.",
        )
        max_per_source = st.slider(
            "Max. Chunks pro Quelle (Soft-Cap)",
            min_value=0, max_value=top_k_rerank, value=min(3, top_k_rerank),
            help="Soft-Cap auf Chunks aus einer einzelnen Datei. Verhindert, dass ein "
                 "dichtes Dokument alle Slots belegt und andere Quellen verdrängt. "
                 "0 = kein Cap (reine Reranker-Reihenfolge). Wird unterschritten, wenn "
                 "sonst weniger als 'Finale Chunks' herauskämen (Backfill).",
        )

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
        st.session_state.session_usage = {"tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0}
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

    # Build request
    request_body = {
        "question": prompt,
        "session_id": st.session_state.session_id,
        "top_k": top_k,
        "top_k_rerank": top_k_rerank,
        "max_per_source": max_per_source,
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
            st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("Verbindung zum Backend fehlgeschlagen. Läuft der Server?")
        except Exception as e:
            st.error(f"Fehler: {e}")
