"""Prompt management page.

Edit active prompts, save/load/delete variants, compare side-by-side.
"""

import requests
import streamlit as st

st.set_page_config(page_title="Prompts — LoreKeeper", page_icon="✏", layout="wide")

API_URL = st.session_state.get("_api_url", "http://localhost:8000")

_TEMPLATE_KEYS = ("system", "qa", "condense", "no_context")
_TEMPLATE_HELP = {
    "system": "Statischer System-Prompt. Keine Jinja2-Variablen.",
    "qa": "Variablen: `chunks` (Liste mit `.source_file`, `.heading`, `.content`), `question`",
    "condense": "Variablen: `history` (Liste mit `.role`, `.content`), `question`",
    "no_context": "Variablen: `question`",
}
_TEMPLATE_LABELS = {
    "system": "System-Prompt",
    "qa": "QA-Template",
    "condense": "Condense-Template",
    "no_context": "No-Context-Template",
}


# ─── Helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=10, show_spinner=False)
def _fetch_active(api_url: str):
    """Retrieve currently active prompt templates.
    
    Uses Streamlit's @st.cache_data with a 10s TTL to prevent spamming
    the backend when switching tabs or rerendering components.
    
    Args:
        api_url (str): The base URL of the backend API.
        
    Returns:
        dict: The active prompt variants containing 'system', 'qa', 'condense' and 'no_context' templates.
    """
    return requests.get(f"{api_url}/prompts/active", timeout=5).json()


@st.cache_data(ttl=10, show_spinner=False)
def _fetch_variants(api_url: str):
    """Retrieve metadata for all stored prompt template variants.
    
    Cached for 10s to minimize backend load. The cache can be cleared manually
    via _fetch_variants.clear() after mutations (save/delete/activate).
    
    Args:
        api_url (str): The base URL of the backend API.
        
    Returns:
        list[dict]: List of metadata dictionaries for each variant.
    """
    return requests.get(f"{api_url}/prompts/variants", timeout=5).json()


@st.cache_data(ttl=10, show_spinner=False)
def _fetch_variant(api_url: str, name: str):
    """Retrieve the full text for a specific stored prompt template variant.
    
    Cached per variant name for 10s. Used when expanding a variant's edit panel
    or loading variants for side-by-side comparison.
    
    Args:
        api_url (str): The base URL of the backend API.
        name (str): The unique name of the variant to load.
        
    Returns:
        dict: The full variant data including the actual template strings.
    """
    return requests.get(f"{api_url}/prompts/variants/{name}", timeout=5).json()


# ─── Page ─────────────────────────────────────────────────────────────

st.title("✏ Prompts")
st.caption("Prompt-Templates bearbeiten, Varianten verwalten und vergleichen.")

tab_active, tab_variants, tab_compare = st.tabs([
    "Aktive Prompts", "Varianten", "Vergleichen",
])


# ═══ Tab 1: Aktive Prompts ════════════════════════════════════════════

with tab_active:
    try:
        active = _fetch_active(API_URL)
        prompts = active.get("prompts", {})
    except Exception as e:
        st.error(f"Backend nicht erreichbar: {e}")
        prompts = {}

    if prompts:
        edited = {}
        for key in _TEMPLATE_KEYS:
            label = _TEMPLATE_LABELS[key]
            help_text = _TEMPLATE_HELP[key]
            edited[key] = st.text_area(
                label,
                value=prompts.get(key, ""),
                height=300,
                help=help_text,
                key=f"active_{key}",
            )

            with st.expander(f"Preview: {label}"):
                sample = {}
                if key == "qa":
                    sample = {
                        "chunks": [
                            {"source_file": "Orte/Arkenfeld.md", "heading": "Lage", "content": "Arkenfeld liegt im Norden..."},
                            {"source_file": "NPCs/Malek.md", "heading": "Hintergrund", "content": "Malek ist ein Händler..."},
                        ],
                        "question": "Was ist Arkenfeld?",
                    }
                elif key == "condense":
                    sample = {
                        "history": [
                            {"role": "user", "content": "Was ist Arkenfeld?"},
                            {"role": "assistant", "content": "Arkenfeld ist eine Stadt im Norden."},
                        ],
                        "question": "Wer lebt dort?",
                    }
                elif key == "no_context":
                    sample = {"question": "Was ist Arkenfeld?"}

                if st.button("Rendern", key=f"preview_{key}"):
                    try:
                        resp = requests.post(
                            f"{API_URL}/prompts/preview",
                            json={"template_text": edited[key], "sample_data": sample},
                            timeout=5,
                        )
                        resp.raise_for_status()
                        st.code(resp.json()["rendered"], language=None)
                    except Exception as e:
                        st.error(f"Fehler: {e}")

        if st.button("💾 Aktive Prompts speichern", type="primary", key="save_active"):
            try:
                resp = requests.put(
                    f"{API_URL}/prompts/active",
                    json={
                        "name": "active",
                        "description": "",
                        "prompts": edited,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                _fetch_active.clear()
                st.success("Prompts gespeichert und neu geladen.")
            except Exception as e:
                st.error(f"Speichern fehlgeschlagen: {e}")
    else:
        st.warning("Keine aktiven Prompts gefunden.")


# ═══ Tab 2: Varianten ═════════════════════════════════════════════════

with tab_variants:
    # Save active as variant
    st.subheader("Aktive Prompts als Variante sichern")
    st.caption("Speichert eine Kopie der aktuell aktiven Prompts (aus Tab \"Aktive Prompts\") "
               "als neue Variante.")
    col_name, col_desc, col_btn = st.columns([2, 3, 2])
    with col_name:
        var_name = st.text_input("Name", key="var_save_name",
                                  placeholder="z.B. concise-german",
                                  help="Dateiname ohne .yaml")
    with col_desc:
        var_desc = st.text_input("Beschreibung", key="var_save_desc",
                                  placeholder="z.B. Kürzere Antworten")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Kopie speichern", key="var_save_btn", type="primary"):
            if not var_name or not var_name.strip():
                st.error("Name ist Pflichtfeld.")
            else:
                try:
                    current = _fetch_active(API_URL)
                    resp = requests.put(
                        f"{API_URL}/prompts/variants/{var_name.strip()}",
                        json={
                            "name": var_name.strip(),
                            "description": var_desc.strip(),
                            "prompts": current.get("prompts", {}),
                        },
                        timeout=10,
                    )
                    resp.raise_for_status()
                    _fetch_variants.clear()
                    _fetch_variant.clear()
                    st.success(f"Variante '{var_name}' gespeichert.")
                except Exception as e:
                    st.error(f"Fehler: {e}")

    st.divider()

    # List variants
    st.subheader("Gespeicherte Varianten")
    try:
        variants = _fetch_variants(API_URL)
    except Exception:
        variants = []

    if not variants:
        st.info("Noch keine Varianten gespeichert.")
    else:
        for v in variants:
            active_marker = " ✅ aktiv" if v.get("is_active") else ""
            desc = f" — {v['description']}" if v.get("description") else ""
            with st.expander(f"**{v['name']}**{desc}{active_marker}"):
                # Load full variant content
                try:
                    full = _fetch_variant(API_URL, v["name"])
                    vp = full.get("prompts", {})
                except Exception as e:
                    st.error(f"Laden fehlgeschlagen: {e}")
                    continue

                # Editable text areas for each template
                edited_v = {}
                for key in _TEMPLATE_KEYS:
                    edited_v[key] = st.text_area(
                        _TEMPLATE_LABELS[key],
                        value=vp.get(key, ""),
                        height=250,
                        help=_TEMPLATE_HELP[key],
                        key=f"var_{v['name']}_{key}",
                    )

                # Description edit
                edited_desc = st.text_input(
                    "Beschreibung",
                    value=full.get("description", ""),
                    key=f"var_{v['name']}_desc",
                )

                # Action buttons
                col_save, col_activate, col_del = st.columns(3)
                with col_save:
                    if st.button("💾 Speichern", key=f"save_{v['name']}", type="primary"):
                        try:
                            resp = requests.put(
                                f"{API_URL}/prompts/variants/{v['name']}",
                                json={
                                    "name": v["name"],
                                    "description": edited_desc.strip(),
                                    "prompts": edited_v,
                                },
                                timeout=10,
                            )
                            resp.raise_for_status()
                            _fetch_variants.clear()
                            _fetch_variant.clear()
                            st.success("Variante gespeichert.")
                        except Exception as e:
                            st.error(f"Fehler: {e}")
                with col_activate:
                    if st.button("Aktivieren", key=f"activate_{v['name']}"):
                        try:
                            resp = requests.post(
                                f"{API_URL}/prompts/activate/{v['name']}",
                                timeout=10,
                            )
                            resp.raise_for_status()
                            _fetch_active.clear()
                            _fetch_variants.clear()
                            st.success(f"Variante '{v['name']}' aktiviert.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fehler: {e}")
                with col_del:
                    if st.button("🗑 Löschen", key=f"del_{v['name']}"):
                        try:
                            requests.delete(
                                f"{API_URL}/prompts/variants/{v['name']}",
                                timeout=5,
                            )
                            _fetch_variants.clear()
                            _fetch_variant.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fehler: {e}")


# ═══ Tab 3: Vergleichen ═══════════════════════════════════════════════

with tab_compare:
    try:
        variants_list = _fetch_variants(API_URL)
    except Exception:
        variants_list = []

    options = ["(Aktiv)"] + [v["name"] for v in variants_list]

    if len(options) < 2:
        st.info("Mindestens eine gespeicherte Variante nötig für einen Vergleich.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            sel_a = st.selectbox("Variante A", options, index=0, key="cmp_a")
        with col_b:
            sel_b = st.selectbox("Variante B", options,
                                  index=min(1, len(options) - 1), key="cmp_b")

        if sel_a == sel_b:
            st.caption("Wähle zwei verschiedene Varianten.")
        elif st.button("Vergleichen", key="cmp_run"):
            try:
                if sel_a == "(Aktiv)":
                    data_a = _fetch_active(API_URL)
                else:
                    data_a = _fetch_variant(API_URL, sel_a)

                if sel_b == "(Aktiv)":
                    data_b = _fetch_active(API_URL)
                else:
                    data_b = _fetch_variant(API_URL, sel_b)

                prompts_a = data_a.get("prompts", {})
                prompts_b = data_b.get("prompts", {})

                for key in _TEMPLATE_KEYS:
                    label = _TEMPLATE_LABELS[key]
                    text_a = prompts_a.get(key, "")
                    text_b = prompts_b.get(key, "")
                    differs = text_a != text_b
                    marker = " ⚠ unterschiedlich" if differs else " ✅ identisch"

                    st.markdown(f"### {label}{marker}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption(sel_a)
                        st.code(text_a, language=None)
                    with c2:
                        st.caption(sel_b)
                        st.code(text_b, language=None)

            except Exception as e:
                st.error(f"Fehler: {e}")
