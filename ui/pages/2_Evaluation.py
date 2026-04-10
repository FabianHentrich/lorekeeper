"""Evaluation page.

Golden Set management, single-question retrieval preview, retrieval eval,
end-to-end eval, and result comparison — all from the UI.
"""

import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Evaluation — LoreKeeper", page_icon="🎯", layout="wide")

API_URL = st.session_state.get("api_url", "http://localhost:8000")


# ─── Helpers ──────────────────────────────────────────────────────────

def _render_tuning_sliders(prefix: str) -> tuple[int, int, int]:
    """Shared retrieval-parameter sliders, reused across tabs."""
    top_k = st.slider(
        "Kandidaten (Top-K)",
        min_value=1, max_value=50, value=15,
        help="Wie viele Chunks der Vectorstore liefert (Bi-Encoder Recall).",
        key=f"{prefix}_top_k",
    )
    top_k_rerank = st.slider(
        "Finale Chunks (nach Reranking)",
        min_value=1, max_value=top_k, value=min(8, top_k),
        help="Wie viele Chunks der Cross-Encoder am Ende auswählt.",
        key=f"{prefix}_top_k_rerank",
    )
    max_per_source = st.slider(
        "Max. Chunks pro Quelle (Soft-Cap)",
        min_value=0, max_value=top_k_rerank, value=min(3, top_k_rerank),
        help="Soft-Cap pro Datei. 0 = kein Cap.",
        key=f"{prefix}_max_per_source",
    )
    return top_k, top_k_rerank, max_per_source


def _poll_eval_job(job_id: str, label: str = "Evaluation"):
    """Poll an eval job until done/error."""
    with st.status(f"{label} läuft...", expanded=True) as status:
        progress_line = st.empty()
        while True:
            try:
                info = requests.get(f"{API_URL}/eval/status/{job_id}", timeout=5).json()
            except Exception:
                time.sleep(1)
                continue

            state = info.get("status", "unknown")
            progress = info.get("progress", 0)
            total = info.get("total", 0)

            if state in ("queued", "running"):
                if total > 0:
                    progress_line.markdown(f"Frage **{progress}** / {total}")
                else:
                    progress_line.markdown("Vorbereitung …")
                time.sleep(1)
                continue

            progress_line.empty()
            if state == "done":
                status.update(label=f"{label} abgeschlossen", state="complete", expanded=True)
                return info.get("result_file")
            else:
                status.update(label=f"{label} fehlgeschlagen", state="error", expanded=True)
                st.error(info.get("error", "Unbekannter Fehler"))
                return None


def _render_retrieval_result(result: dict):
    """Render a retrieval eval result: metrics, breakdown, details."""
    details = result.get("details", [])
    misses = result.get("misses", [])
    errors = result.get("errors", [])
    breakdown = result.get("breakdown_by_source_type", {})
    config = result.get("config", {})

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hit Rate", f"{result.get('hit_rate', 0):.1%}")
    c2.metric("Fragen", result.get("total_questions", 0))
    avg_lat = sum(d.get("latency_ms", 0) for d in details) / max(len(details), 1)
    c3.metric("Ø Latenz", f"{avg_lat:.0f} ms")
    c4.metric("Config", f"K={config.get('top_k', '?')} R={config.get('top_k_rerank', '?')} C={config.get('max_per_source', '?')}")

    # Breakdown
    if breakdown:
        st.markdown("**Breakdown nach Source-Type**")
        bd_rows = [
            {"Type": st_name, "Hit Rate": f"{v['hit_rate']:.1%}", "Hits": v["hits"], "Total": v["total"]}
            for st_name, v in breakdown.items()
        ]
        st.dataframe(pd.DataFrame(bd_rows), hide_index=True, use_container_width=True)

    # Details table
    if details:
        st.markdown("**Details**")
        for d in details:
            hit = d.get("hit", False)
            icon = "✅" if hit else "❌"
            qid = d.get("id", "?")
            question = d.get("question", "")
            score = d.get("top_score", 0)

            with st.expander(f"{icon} [{qid}] {question[:80]} — Score: {score:.3f}"):
                col_exp, col_ret = st.columns(2)
                with col_exp:
                    st.markdown("**Erwartet:**")
                    for s in d.get("expected_sources", []):
                        st.code(s, language=None)
                with col_ret:
                    st.markdown("**Gefunden:**")
                    for s in d.get("retrieved_files", [])[:5]:
                        st.code(s, language=None)

    if errors:
        st.warning(f"{len(errors)} Fehler aufgetreten")
        for e in errors:
            st.code(f"[{e.get('id', '?')}] {e.get('error', '')}", language=None)


# ─── Page ─────────────────────────────────────────────────────────────

st.title("🎯 Evaluation")
st.caption("Golden Set verwalten, Retrieval testen und tunen, Ergebnisse vergleichen.")

tab_gs, tab_test, tab_ret, tab_e2e, tab_results = st.tabs([
    "Golden Set", "Frage testen", "Retrieval Eval", "End-to-End", "Ergebnisse",
])


# ═══ Tab 1: Golden Set ═══════════════════════════════════════════════

@st.cache_data(ttl=10, show_spinner=False)
def _fetch_qa_pairs(api_url: str):
    return requests.get(f"{api_url}/eval/qa-pairs", timeout=5).json()


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_eval_results(api_url: str):
    return requests.get(f"{api_url}/eval/results", timeout=5).json()


with tab_gs:
    try:
        qa_data = _fetch_qa_pairs(API_URL)
        pairs = qa_data.get("pairs", [])
    except Exception as e:
        st.error(f"Backend nicht erreichbar: {e}")
        pairs = []

    if pairs or True:  # always show editor
        rows = []
        for p in pairs:
            rows.append({
                "id": p.get("id", ""),
                "question": p.get("question", ""),
                "source_type": p.get("source_type", "markdown"),
                "category": p.get("category", ""),
                "expected_sources": ", ".join(p.get("expected_sources", [])),
                "expected_answer_contains": ", ".join(p.get("expected_answer_contains", [])),
                "notes": p.get("notes", ""),
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["id", "question", "source_type", "category",
                     "expected_sources", "expected_answer_contains", "notes"]
        )

        edited = st.data_editor(
            df,
            num_rows="dynamic",
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "question": st.column_config.TextColumn("Frage"),
                "source_type": st.column_config.SelectboxColumn(
                    "Source Type", options=["markdown", "pdf", "image"], width="small",
                ),
                "category": st.column_config.TextColumn("Kategorie", width="small"),
                "expected_sources": st.column_config.TextColumn(
                    "Expected Sources",
                    help="Komma-getrennt, z.B.: Orte/Arkenfeld.md, NPCs/Malek.md",
                ),
                "expected_answer_contains": st.column_config.TextColumn(
                    "Answer Contains",
                    help="Komma-getrennte Keywords für die E2E-Prüfung",
                ),
                "notes": st.column_config.TextColumn("Notizen"),
            },
            hide_index=True,
            use_container_width=True,
            key="gs_editor",
        )

        if st.button("💾 Golden Set speichern", type="primary", key="gs_save"):
            save_pairs = []
            for _, row in edited.iterrows():
                pid = (row.get("id") or "").strip()
                pq = (row.get("question") or "").strip()
                if not pid or not pq:
                    continue
                save_pairs.append({
                    "id": pid,
                    "question": pq,
                    "source_type": row.get("source_type") or "markdown",
                    "category": (row.get("category") or "").strip(),
                    "expected_sources": [
                        s.strip() for s in (row.get("expected_sources") or "").split(",") if s.strip()
                    ],
                    "expected_answer_contains": [
                        s.strip() for s in (row.get("expected_answer_contains") or "").split(",") if s.strip()
                    ],
                    "notes": (row.get("notes") or "").strip(),
                })
            try:
                resp = requests.put(
                    f"{API_URL}/eval/qa-pairs",
                    json={"pairs": save_pairs},
                    timeout=10,
                )
                resp.raise_for_status()
                _fetch_qa_pairs.clear()
                st.success(f"{len(save_pairs)} QA-Paare gespeichert.")
            except Exception as e:
                st.error(f"Speichern fehlgeschlagen: {e}")


# ═══ Tab 2: Frage testen ═════════════════════════════════════════════

with tab_test:
    st.markdown("Einzelne Frage testen — zeigt welche Chunks der Retriever liefert.")

    test_question = st.text_input("Frage", key="test_question",
                                  placeholder="z.B.: Was ist Arkenfeld?")

    with st.expander("Retrieval-Parameter", expanded=False):
        t_top_k, t_top_k_rerank, t_max_per_source = _render_tuning_sliders("test")

    if st.button("🔍 Testen", key="test_run", disabled=not test_question.strip()):
        try:
            resp = requests.post(
                f"{API_URL}/eval/preview",
                json={
                    "question": test_question.strip(),
                    "top_k": t_top_k,
                    "top_k_rerank": t_top_k_rerank,
                    "max_per_source": t_max_per_source,
                },
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            chunks = result.get("chunks", [])
            latency = result.get("latency_ms", 0)

            st.caption(f"{len(chunks)} Chunks in {latency:.0f} ms")

            if chunks:
                chunk_rows = []
                for c in chunks:
                    chunk_rows.append({
                        "Score": round(c["score"], 4),
                        "Source": c["source_file"],
                        "Heading": c.get("heading") or "",
                        "Preview": c["content_preview"][:100],
                    })
                st.dataframe(pd.DataFrame(chunk_rows), hide_index=True, use_container_width=True)

                # Convenience: add to Golden Set
                st.divider()
                st.markdown("**Zum Golden Set hinzufügen:**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    add_id = st.text_input("ID", key="test_add_id", placeholder="m099")
                with c2:
                    add_type = st.selectbox("Source Type", ["markdown", "pdf", "image"],
                                            key="test_add_type")
                with c3:
                    add_cat = st.text_input("Kategorie", key="test_add_cat")

                add_sources = st.text_input(
                    "Expected Sources (komma-getrennt)", key="test_add_sources",
                    value=", ".join(c["source_file"] for c in chunks[:3]),
                )

                if st.button("➕ Hinzufügen", key="test_add_btn"):
                    if not add_id.strip():
                        st.error("ID ist Pflichtfeld.")
                    else:
                        try:
                            existing = requests.get(f"{API_URL}/eval/qa-pairs", timeout=5).json()
                            existing_pairs = existing.get("pairs", [])
                            existing_pairs.append({
                                "id": add_id.strip(),
                                "question": test_question.strip(),
                                "source_type": add_type,
                                "category": (add_cat or "").strip(),
                                "expected_sources": [
                                    s.strip() for s in add_sources.split(",") if s.strip()
                                ],
                                "expected_answer_contains": [],
                                "notes": "",
                            })
                            resp = requests.put(
                                f"{API_URL}/eval/qa-pairs",
                                json={"pairs": existing_pairs},
                                timeout=10,
                            )
                            resp.raise_for_status()
                            _fetch_qa_pairs.clear()
                            st.success(f"'{add_id}' zum Golden Set hinzugefügt.")
                        except Exception as e:
                            st.error(f"Fehler: {e}")
            else:
                st.info("Keine Chunks gefunden.")

        except Exception as e:
            st.error(f"Fehler: {e}")


# ═══ Tab 3: Retrieval Evaluation ═════════════════════════════════════

with tab_ret:
    st.markdown("Führt das komplette Golden Set durch den Retriever (kein LLM).")

    with st.expander("Retrieval-Parameter", expanded=False):
        r_top_k, r_top_k_rerank, r_max_per_source = _render_tuning_sliders("ret")

    if st.button("▶ Retrieval Evaluation starten", type="primary", key="ret_run"):
        try:
            resp = requests.post(
                f"{API_URL}/eval/run",
                json={
                    "top_k": r_top_k,
                    "top_k_rerank": r_top_k_rerank,
                    "max_per_source": r_max_per_source,
                    "eval_type": "retrieval",
                },
                timeout=10,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            result_file = _poll_eval_job(job_id, label="Retrieval Evaluation")
            _fetch_eval_results.clear()

            if result_file:
                result = requests.get(f"{API_URL}/eval/results/{result_file}", timeout=10).json()
                _render_retrieval_result(result)

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                st.warning("Es läuft bereits eine Evaluation.")
            else:
                st.error(f"Fehler: {e}")
        except Exception as e:
            st.error(f"Fehler: {e}")


# ═══ Tab 4: End-to-End ═══════════════════════════════════════════════

with tab_e2e:
    st.markdown("Führt das Golden Set durch die komplette Pipeline inkl. LLM.")
    st.warning("Langsam und verbraucht LLM-Tokens. Backend muss laufen.")

    if st.button("▶ End-to-End Evaluation starten", type="primary", key="e2e_run"):
        try:
            resp = requests.post(
                f"{API_URL}/eval/run",
                json={"eval_type": "e2e"},
                timeout=10,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            result_file = _poll_eval_job(job_id, label="End-to-End Evaluation")
            _fetch_eval_results.clear()

            if result_file:
                result = requests.get(f"{API_URL}/eval/results/{result_file}", timeout=10).json()

                c1, c2, c3 = st.columns(3)
                c1.metric("Hit Rate", f"{result.get('hit_rate', 0):.1%}")
                c2.metric("Answer Contains", f"{result.get('answer_contains_rate', 0):.1%}")
                c3.metric("Ø Latenz", f"{result.get('avg_latency_ms', 0):.0f} ms")

                details = result.get("details", [])
                if details:
                    st.markdown("**Details**")
                    for d in details:
                        hit = d.get("hit", False)
                        ac = d.get("answer_contains_match", False)
                        icon = "✅" if hit else "❌"
                        ac_icon = "✅" if ac else "⚠️"

                        with st.expander(
                            f"{icon} {d.get('question', '')[:80]} — "
                            f"Retrieval: {icon} Answer: {ac_icon}"
                        ):
                            st.markdown(f"**Antwort:** {d.get('answer', '')}")
                            col_exp, col_ret = st.columns(2)
                            with col_exp:
                                st.markdown("**Erwartet:**")
                                for s in d.get("expected_sources", []):
                                    st.code(s, language=None)
                            with col_ret:
                                st.markdown("**Gefunden:**")
                                for s in d.get("sources_retrieved", [])[:5]:
                                    st.code(s, language=None)

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                st.warning("Es läuft bereits eine Evaluation.")
            else:
                st.error(f"Fehler: {e}")
        except Exception as e:
            st.error(f"Fehler: {e}")


# ═══ Tab 5: Ergebnisse ═══════════════════════════════════════════════

with tab_results:
    try:
        results_list = _fetch_eval_results(API_URL)
    except Exception:
        results_list = []

    if not results_list:
        st.info("Noch keine Ergebnisse vorhanden. Starte eine Evaluation in den Tabs oben.")
    else:
        # List results
        st.markdown("**Gespeicherte Ergebnisse** (max. 3 pro Typ)")
        for r in results_list:
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f"**{r['filename']}** — {r['eval_type']} · "
                    f"Hit Rate: {r['hit_rate']:.1%} · "
                    f"{r['total_questions']} Fragen · "
                    f"Config: {r.get('config', {})}"
                )
            with col_del:
                if st.button("🗑", key=f"del_{r['filename']}"):
                    try:
                        requests.delete(f"{API_URL}/eval/results/{r['filename']}", timeout=5)
                        _fetch_eval_results.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Fehler: {e}")

        # Compare two results
        st.divider()
        st.subheader("Ergebnisse vergleichen")

        filenames = [r["filename"] for r in results_list]
        if len(filenames) < 2:
            st.caption("Mindestens 2 Ergebnisse nötig für einen Vergleich.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                sel_a = st.selectbox("Ergebnis A", filenames, index=0, key="cmp_a")
            with c2:
                sel_b = st.selectbox("Ergebnis B", filenames,
                                     index=min(1, len(filenames) - 1), key="cmp_b")

            if sel_a and sel_b and sel_a != sel_b:
                if st.button("Vergleichen", key="cmp_run"):
                    try:
                        res_a = requests.get(f"{API_URL}/eval/results/{sel_a}", timeout=10).json()
                        res_b = requests.get(f"{API_URL}/eval/results/{sel_b}", timeout=10).json()

                        # Metric comparison
                        st.markdown("**Metriken**")
                        mc1, mc2, mc3 = st.columns(3)
                        hr_a = res_a.get("hit_rate", 0)
                        hr_b = res_b.get("hit_rate", 0)
                        mc1.metric("Hit Rate A", f"{hr_a:.1%}")
                        mc2.metric("Hit Rate B", f"{hr_b:.1%}")
                        diff = hr_b - hr_a
                        mc3.metric("Differenz", f"{diff:+.1%}")

                        # Per-question diff
                        details_a = {d.get("id", d.get("question", "")): d for d in res_a.get("details", [])}
                        details_b = {d.get("id", d.get("question", "")): d for d in res_b.get("details", [])}

                        flipped = []
                        all_ids = sorted(set(details_a.keys()) | set(details_b.keys()))
                        for qid in all_ids:
                            da = details_a.get(qid)
                            db = details_b.get(qid)
                            if da and db and da.get("hit") != db.get("hit"):
                                flipped.append({
                                    "ID": qid,
                                    "Frage": (da.get("question") or "")[:60],
                                    "A": "✅" if da.get("hit") else "❌",
                                    "B": "✅" if db.get("hit") else "❌",
                                })

                        if flipped:
                            st.markdown(f"**{len(flipped)} Fragen mit geändertem Ergebnis:**")
                            st.dataframe(pd.DataFrame(flipped), hide_index=True,
                                         use_container_width=True)
                        else:
                            st.success("Keine Unterschiede bei Hit/Miss.")

                    except Exception as e:
                        st.error(f"Fehler: {e}")
            elif sel_a == sel_b:
                st.caption("Wähle zwei verschiedene Ergebnisse.")
