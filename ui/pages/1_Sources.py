"""Sources management page.

Lets the user view, add, edit, reindex, recategorize, and remove sources
without touching config/sources.yaml manually. Also exposes the danger-zone
'wipe collection' button.
"""

import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


def _poll_ingest_job(api_url: str, job_id: str, label: str = "Indizierung"):
    """
    Poll an active ingest job endpoint until terminal state is reached.
    Renders live progress updates showing chunks processed versus total files.
    """
    with st.status(f"{label} läuft...", expanded=True) as status:
        progress_line = st.empty()
        while True:
            try:
                info = requests.get(f"{api_url}/ingest/status/{job_id}", timeout=5).json()
            except Exception:
                time.sleep(1)
                continue

            state = info.get("status", "unknown")
            phase = info.get("phase", "starting")
            docs = info.get("documents_processed", 0)
            total = info.get("documents_total", 0)
            created = info.get("chunks_created", 0)
            updated = info.get("chunks_updated", 0)
            deleted = info.get("chunks_deleted", 0)
            errors = info.get("errors", [])
            duration = info.get("duration_seconds", 0)

            if state in ("queued", "running"):
                phase_labels = {
                    "starting": "Vorbereitung …",
                    "parsing": f"Parsing & Chunking: **{docs}** / {total} Dokumente",
                    "embedding": "Embedding & Speichern …",
                }
                progress_line.markdown(phase_labels.get(phase, phase))
                time.sleep(1.5)
                continue

            # Terminal state
            progress_line.empty()
            if state == "done":
                status.update(label=f"{label} abgeschlossen", state="complete", expanded=True)
                st.write(
                    f"**{docs}** Dokumente in **{duration:.1f}s**  \n"
                    f"Chunks: **{created}** neu · **{updated}** aktualisiert · **{deleted}** gelöscht"
                )
            else:
                status.update(label=f"{label} fehlgeschlagen", state="error", expanded=True)
                st.write(f"Status: {state}")

            if errors:
                for err in errors:
                    st.warning(err)
            break

st.set_page_config(page_title="Sources — LoreKeeper", page_icon="⚙", layout="wide")

API_URL = st.session_state.get("_api_url", "http://localhost:8000")

st.title("⚙ Sources")
st.caption(
    "Verwalte die Ingestion-Quellen. Eine Source ist entweder ein Ordner "
    "oder eine einzelne Datei. Änderungen werden in `config/sources.yaml` "
    "persistiert."
)


# ─── Load current sources ───────────────────────────────────────────────
@st.cache_data(ttl=5)
def _fetch_sources():
    """Fetch all configuring Document ingestion sources from the backend API."""
    return requests.get(f"{API_URL}/sources", timeout=5).json()["sources"]


def _refresh():
    """Clear the local component cache and trigger a full UI reload."""
    _fetch_sources.clear()
    st.rerun()


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_stats(api_url: str):
    """Retrieve aggregate VectorStore telemetry (e.g. total indexed chunks)."""
    return requests.get(f"{api_url}/stats", timeout=5).json()


try:
    sources = _fetch_sources()
except Exception as e:
    st.error(f"Backend nicht erreichbar: {e}")
    st.stop()


# ─── Global stats + full reindex ─────────────────────────────────────
col_stat, col_reindex = st.columns([2, 3])
with col_stat:
    try:
        stats = _fetch_stats(API_URL)
        st.metric("Indizierte Chunks (gesamt)", stats["chunk_count"])
    except Exception:
        st.metric("Indizierte Chunks (gesamt)", "?")
with col_reindex:
    st.caption("Alle Sources komplett neu einlesen:")
    if st.button("🔄 Alle Dokumente neu indizieren"):
        try:
            resp = requests.post(f"{API_URL}/ingest", timeout=10).json()
            _poll_ingest_job(API_URL, resp["job_id"], label="Vollständige Indizierung")
        except Exception as e:
            st.error(f"Fehler: {e}")

st.divider()


# ─── Editable table ─────────────────────────────────────────────────────
st.subheader("Konfigurierte Quellen")

if sources:
    df_rows = []
    for s in sources:
        path = Path(s["path"])
        df_rows.append({
            "id": s["id"],
            "path": s["path"],
            "type": "file" if path.is_file() else ("folder" if path.exists() else "missing"),
            "group": s["group"],
            "default_category": s.get("default_category") or "",
            "category_map": ", ".join(
                f"{k}→{v['category']}:{v['group']}" if isinstance(v, dict) and "group" in v
                else f"{k}→{v['category']}" if isinstance(v, dict)
                else f"{k}→{v}"
                for k, v in (s.get("category_map") or {}).items()
            ),
        })
    df = pd.DataFrame(df_rows)

    edited = st.data_editor(
        df,
        column_config={
            "id": st.column_config.TextColumn("ID", disabled=True, width="small"),
            "path": st.column_config.TextColumn("Pfad", disabled=True),
            "type": st.column_config.TextColumn("Typ", disabled=True, width="small"),
            "group": st.column_config.SelectboxColumn(
                "Group", options=["lore", "adventure", "rules"], required=True, width="small",
            ),
            "default_category": st.column_config.TextColumn("Default-Kategorie", width="small"),
            "category_map": st.column_config.TextColumn(
                "Category-Map",
                help="Ordner→Kategorie oder Ordner→Kategorie:Gruppe. "
                     "Beispiel: NPCs→npc, Geschichte→story:adventure",
            ),
        },
        hide_index=True,
        width="stretch",
        key="sources_editor",
    )

    col_save, col_reload = st.columns([1, 5])
    with col_save:
        if st.button("💾 Änderungen speichern", type="primary"):
            new_sources = []
            for _, row in edited.iterrows():
                cmap = {}
                cm_str = (row["category_map"] or "").strip()
                if cm_str:
                    for part in cm_str.split(","):
                        if "→" in part:
                            k, v = part.split("→", 1)
                            v = v.strip()
                            if ":" in v:
                                cat, grp = v.rsplit(":", 1)
                                cmap[k.strip()] = {"category": cat.strip(), "group": grp.strip()}
                            else:
                                cmap[k.strip()] = v
                # Preserve fields the editor doesn't expose
                original = next((s for s in sources if s["id"] == row["id"]), {})
                new_sources.append({
                    "id": row["id"],
                    "path": row["path"],
                    "group": row["group"],
                    "default_category": (row["default_category"] or None),
                    "category_map": cmap,
                    "exclude_patterns": original.get("exclude_patterns", []),
                })

            try:
                resp = requests.put(f"{API_URL}/sources", json={"sources": new_sources}, timeout=10)
                resp.raise_for_status()
                st.success("Gespeichert. Du musst u.U. Recategorize oder Reindex anwerfen, "
                           "damit bestehende Chunks die neuen Werte übernehmen.")
                _refresh()
            except Exception as e:
                st.error(f"Speichern fehlgeschlagen: {e}")
    with col_reload:
        if st.button("🔄 Neu laden"):
            _refresh()
else:
    st.warning("Keine Quellen konfiguriert. Lege unten die erste Source an "
               "(Pfad zu deinem Obsidian-Vault oder einer einzelnen Datei), "
               "dann starte die Indizierung.")


# ─── Per-source actions ────────────────────────────────────────────────
st.subheader("Aktionen pro Quelle")

_GROUPS = ["lore", "adventure", "rules"]

for s in sources:
    with st.expander(f"📁 {s['id']} — {s['path']}"):
        # ── Action buttons ──
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔄 Reindex", key=f"reindex_{s['id']}",
                         help="Löscht alle Chunks dieser Source und liest sie neu ein. "
                              "Pflicht, wenn sich Inhalte oder der Pfad geändert haben."):
                try:
                    r = requests.post(f"{API_URL}/sources/{s['id']}/reindex", timeout=10)
                    r.raise_for_status()
                    _poll_ingest_job(API_URL, r.json()["job_id"], label=f"Reindex '{s['id']}'")
                except Exception as e:
                    st.error(f"Fehler: {e}")
        with col_b:
            if st.button("🏷 Recategorize (alle)", key=f"recat_{s['id']}",
                         help="Schnell, kein Re-Embedding. Bringt bestehende Chunks "
                              "auf die aktuelle category_map / group. Wirkt auf alle Sources."):
                try:
                    r = requests.post(f"{API_URL}/sources/recategorize", timeout=60)
                    r.raise_for_status()
                    st.success(f"Recategorize: {r.json()}")
                except Exception as e:
                    st.error(f"Fehler: {e}")
        with col_c:
            confirm = st.checkbox("Wirklich löschen", key=f"confirm_del_{s['id']}")
            if st.button("🗑 Source entfernen", key=f"del_{s['id']}", disabled=not confirm,
                         help="Entfernt die Source aus der Config UND löscht alle "
                              "zugehörigen Chunks aus der Datenbank."):
                try:
                    r = requests.delete(f"{API_URL}/sources/{s['id']}", timeout=10)
                    r.raise_for_status()
                    st.success(f"Entfernt. Gelöschte Chunks: {r.json()['deleted_chunks']}")
                    _refresh()
                except Exception as e:
                    st.error(f"Fehler: {e}")

        # ── Folder mapping ──
        st.markdown("**Ordner-Zuordnung**")
        try:
            tree = requests.get(f"{API_URL}/sources/{s['id']}/folders", timeout=5).json()
            folders = tree.get("folders", [])
        except Exception:
            folders = []

        if not folders:
            st.caption("Keine Unterordner gefunden (Einzeldatei-Source oder Pfad nicht erreichbar).")
        else:
            cmap = s.get("category_map") or {}
            src_group = s.get("group", "lore")
            src_default_cat = s.get("default_category") or "misc"

            mapping_rows = []
            for entry in folders:
                name = entry["name"]
                existing = cmap.get(name)
                if isinstance(existing, dict):
                    cur_cat = existing.get("category", src_default_cat)
                    cur_grp = existing.get("group", src_group)
                elif isinstance(existing, str):
                    cur_cat = existing
                    cur_grp = src_group
                else:
                    cur_cat = src_default_cat
                    cur_grp = src_group
                mapping_rows.append({
                    "name": name,
                    "type": entry["type"],
                    "category": cur_cat,
                    "group": cur_grp,
                })

            map_df = pd.DataFrame(mapping_rows)
            edited_map = st.data_editor(
                map_df,
                column_config={
                    "name": st.column_config.TextColumn("Ordner / Datei", disabled=True),
                    "type": st.column_config.TextColumn("Typ", disabled=True, width="small"),
                    "category": st.column_config.TextColumn("Kategorie", width="small"),
                    "group": st.column_config.SelectboxColumn(
                        "Group", options=_GROUPS, required=True, width="small",
                    ),
                },
                hide_index=True,
                width="stretch",
                key=f"foldermap_{s['id']}",
            )

            if st.button("💾 Zuordnung speichern", key=f"savemap_{s['id']}", type="primary"):
                new_cmap = {}
                for _, row in edited_map.iterrows():
                    cat = (row["category"] or src_default_cat).strip()
                    grp = row["group"]
                    if grp != src_group:
                        new_cmap[row["name"]] = {"category": cat, "group": grp}
                    else:
                        new_cmap[row["name"]] = cat

                # Update this source's category_map and save
                updated_sources = []
                for src in sources:
                    if src["id"] == s["id"]:
                        updated_src = dict(src)
                        updated_src["category_map"] = new_cmap
                        updated_sources.append(updated_src)
                    else:
                        updated_sources.append(src)
                try:
                    resp = requests.put(
                        f"{API_URL}/sources", json={"sources": updated_sources}, timeout=10,
                    )
                    resp.raise_for_status()
                    st.success("Zuordnung gespeichert. Recategorize starten, "
                               "damit bestehende Chunks aktualisiert werden.")
                    _refresh()
                except Exception as e:
                    st.error(f"Speichern fehlgeschlagen: {e}")


# ─── Add new source ─────────────────────────────────────────────────────
st.subheader("Neue Source hinzufügen")

# Step 1: Enter path and scan
st.markdown("**1. Pfad eingeben und scannen**")
col_path, col_scan = st.columns([4, 1])
with col_path:
    new_path = st.text_input("Pfad zum Ordner oder zur Datei",
                             key="_new_source_path",
                             help="Absoluter oder relativer Pfad zu deinem Vault, "
                                  "Ordner oder einer einzelnen Datei.")
with col_scan:
    st.markdown("<br>", unsafe_allow_html=True)
    scan_clicked = st.button("🔍 Scannen", key="_scan_btn")

# Persist scan results in session state
if scan_clicked and new_path.strip():
    try:
        scan_resp = requests.post(
            f"{API_URL}/sources/scan", json={"path": new_path.strip()}, timeout=10,
        ).json()
        st.session_state["_scan_result"] = scan_resp
        st.session_state["_scanned_path"] = new_path.strip()
    except Exception as e:
        st.error(f"Scan fehlgeschlagen: {e}")
        st.session_state.pop("_scan_result", None)

scan_result = st.session_state.get("_scan_result")
scan_source_path = st.session_state.get("_scanned_path", "")

if scan_result is not None and scan_source_path:
    if scan_result.get("error"):
        st.error(scan_result["error"])
    else:
        is_file = scan_result.get("is_file", False)
        scanned_folders = scan_result.get("folders", [])

        # Step 2: ID, default group, default category
        st.markdown("**2. Source-Einstellungen**")
        c1, c2, c3 = st.columns(3)
        with c1:
            # Auto-suggest ID from folder/file name
            suggested_id = Path(scan_source_path).stem.lower().replace(" ", "-")
            new_id = st.text_input("ID", value=suggested_id, key="_new_source_id",
                                   help="Stabiler Identifier. Wird zum Filtern und Reindex verwendet.")
        with c2:
            new_group = st.selectbox("Default-Group", _GROUPS, key="_new_source_group",
                                     help="Fallback-Gruppe für nicht zugeordnete Ordner/Dateien.")
        with c3:
            new_default = st.text_input("Default-Kategorie", value="misc",
                                        key="_new_source_default_cat",
                                        help="Fallback content_category.")

        # Step 3: Folder tree mapping (only for folder sources)
        new_cmap = {}
        if is_file:
            st.caption("Einzeldatei-Source — keine Unterordner zum Zuordnen.")
        elif scanned_folders:
            st.markdown("**3. Ordner und Dateien zuordnen**")
            scan_rows = []
            for entry in scanned_folders:
                scan_rows.append({
                    "name": entry["name"],
                    "type": entry["type"],
                    "category": new_default if new_default else "misc",
                    "group": new_group,
                })
            scan_df = pd.DataFrame(scan_rows)
            edited_scan = st.data_editor(
                scan_df,
                column_config={
                    "name": st.column_config.TextColumn("Ordner / Datei", disabled=True),
                    "type": st.column_config.TextColumn("Typ", disabled=True, width="small"),
                    "category": st.column_config.TextColumn("Kategorie", width="small"),
                    "group": st.column_config.SelectboxColumn(
                        "Group", options=_GROUPS, required=True, width="small",
                    ),
                },
                hide_index=True,
                width="stretch",
                key="_new_source_foldermap",
            )

            # Build category_map from edited table
            for _, row in edited_scan.iterrows():
                cat = (row["category"] or new_default or "misc").strip()
                grp = row["group"]
                if grp != new_group:
                    new_cmap[row["name"]] = {"category": cat, "group": grp}
                elif cat != (new_default or "misc"):
                    new_cmap[row["name"]] = cat
        else:
            st.caption("Keine Unterordner oder unterstützten Dateien gefunden.")

        # Step 4: Save
        if st.button("➕ Source hinzufügen", type="primary", key="_add_source_final"):
            if not new_id or not new_id.strip():
                st.error("ID ist ein Pflichtfeld.")
            elif any(s["id"] == new_id.strip() for s in sources):
                st.error(f"ID '{new_id}' existiert bereits.")
            else:
                updated = sources + [{
                    "id": new_id.strip(),
                    "path": scan_source_path,
                    "group": new_group,
                    "default_category": new_default or None,
                    "category_map": new_cmap,
                    "exclude_patterns": [],
                }]
                try:
                    r = requests.put(f"{API_URL}/sources",
                                     json={"sources": updated}, timeout=10)
                    r.raise_for_status()
                    st.success(f"Source '{new_id}' hinzugefügt. Starte jetzt Reindex "
                               "um die Dokumente zu indizieren.")
                    st.session_state.pop("_scan_result", None)
                    st.session_state.pop("_scanned_path", None)
                    _refresh()
                except Exception as e:
                    st.error(f"Fehler: {e}")


# ─── Danger zone ───────────────────────────────────────────────────────
st.divider()
st.subheader("⚠ Danger Zone")
st.caption("Komplettes Leeren der Vector-Collection. Alle Embeddings sind danach weg "
           "und müssen via Ingest neu erzeugt werden.")
confirm_text = st.text_input("Tippe DELETE zum Bestätigen", key="wipe_confirm")
if st.button("🔥 Collection komplett leeren", type="secondary"):
    if confirm_text != "DELETE":
        st.error("Bestätigung fehlt.")
    else:
        try:
            r = requests.post(f"{API_URL}/admin/wipe", json={"confirm": "DELETE"}, timeout=10)
            r.raise_for_status()
            st.success("Collection geleert.")
        except Exception as e:
            st.error(f"Fehler: {e}")
