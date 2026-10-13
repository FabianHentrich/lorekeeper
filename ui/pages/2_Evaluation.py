"""Evaluation page.

Golden Set management, single-question retrieval preview, retrieval eval,
end-to-end eval, and result comparison — all from the UI.
"""

import hashlib
import math
import statistics
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from admin_gate import api_url, require_admin
from backend import BackendError, get_json, raise_for_status
from feedback import pairs_from_feedback
from shutdown import render_stop_button

st.set_page_config(page_title="Evaluation — LoreKeeper", page_icon="🎯", layout="wide")

require_admin()
with st.sidebar:
    render_stop_button()

API_URL = api_url()


# ─── Helpers ──────────────────────────────────────────────────────────

_TOP_K_MAX = 50
# Only when /config can't be read — the sliders normally start at the values from settings.yaml
_FALLBACK_RETRIEVAL = {"top_k": 20, "reranking": {"top_k_rerank": 8, "max_per_source": 3}}


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_retrieval_config(api_url: str) -> dict:
    """The retrieval part of /config — the sliders' start values. Errors are not cached."""
    r = requests.get(f"{api_url}/config", timeout=5)
    raise_for_status(r)
    return r.json()["retrieval"]


def _render_tuning_sliders(prefix: str) -> tuple[int, int, int]:
    """Shared retrieval-parameter sliders, reused across tabs; they start at the configured values,
    so a run with untouched sliders measures what the chat uses."""
    try:
        cfg = _fetch_retrieval_config(API_URL)
    except Exception:
        cfg = _FALLBACK_RETRIEVAL
    top_k = st.slider(
        "Kandidaten (Top-K)",
        min_value=1, max_value=_TOP_K_MAX, value=min(cfg["top_k"], _TOP_K_MAX),
        help="Wie viele Chunks der Vectorstore liefert (Bi-Encoder Recall).",
        key=f"{prefix}_top_k",
    )
    top_k_rerank = st.slider(
        "Finale Chunks (nach Reranking)",
        min_value=1, max_value=top_k, value=min(cfg["reranking"]["top_k_rerank"], top_k),
        help="Wie viele Chunks der Cross-Encoder am Ende auswählt.",
        key=f"{prefix}_top_k_rerank",
    )
    max_per_source = st.slider(
        "Max. Chunks pro Quelle (Soft-Cap)",
        min_value=0, max_value=top_k_rerank, value=min(cfg["reranking"]["max_per_source"], top_k_rerank),
        help="Soft-Cap pro Datei. 0 = kein Cap.",
        key=f"{prefix}_max_per_source",
    )
    return top_k, top_k_rerank, max_per_source


_HYBRID_LABELS = ["Config-Default", "Vektor only", "Hybrid (BM25 + Vektor)"]


def _render_hybrid_selector(prefix: str) -> bool | None:
    """Retrieval-mode selector: None = config default, False = pure vector, True = hybrid."""
    mode = st.radio(
        "Retrieval-Modus",
        _HYBRID_LABELS,
        index=0,
        horizontal=True,
        help="Per-Request Override. 'Config-Default' nutzt retrieval.hybrid.enabled aus settings.yaml.",
        key=f"{prefix}_hybrid_mode",
    )
    if mode == "Vektor only":
        return False
    if mode.startswith("Hybrid"):
        return True
    return None


def _hybrid_label(value) -> str:
    """Format the hybrid config flag for display in result summaries."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    return "default"


def _csv(value) -> list[str]:
    """A comma-separated editor cell as a list (empty entries dropped)."""
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _show_eval_error(e: Exception):
    """A 409 means another eval job or an ingest is running — a warning with the backend's reason."""
    if isinstance(e, BackendError) and e.status_code == 409:
        st.warning(e.detail)
    else:
        st.error(f"Fehler: {e}")


def _flipped(res_a: dict, res_b: dict, label_a: str, label_b: str) -> list[dict]:
    """Questions whose hit/miss differs between two eval results (matched by id, else question)."""
    details_a = {d.get("id", d.get("question", "")): d for d in res_a.get("details", [])}
    details_b = {d.get("id", d.get("question", "")): d for d in res_b.get("details", [])}
    return [
        {
            "ID": qid,
            "Frage": (details_a[qid].get("question") or "")[:60],
            label_a: "✅" if details_a[qid].get("hit") else "❌",
            label_b: "✅" if details_b[qid].get("hit") else "❌",
        }
        for qid in sorted(details_a.keys() & details_b.keys())
        if details_a[qid].get("hit") != details_b[qid].get("hit")
    ]


_POLL_MAX_FAILURES = 45  # ~45 s of unreachable backend before giving up (like the Sources page)


def _poll_eval_job(job_id: str, label: str = "Evaluation"):
    """
    Poll an active background evaluation job.
    Monitors progress percentage out of the total question pool and blocks
    the UI until the final results artifact is ready. Gives up after
    ``_POLL_MAX_FAILURES`` failed status requests in a row instead of spinning forever.
    """
    with st.status(f"{label} läuft...", expanded=True) as status:
        progress_line = st.empty()
        failures = 0
        while True:
            try:
                info = get_json(f"{API_URL}/eval/status/{job_id}", timeout=5)
                failures = 0
            except BackendError as e:
                # e.g. 404: job registry gone after a backend restart — final, not "unreachable"
                progress_line.empty()
                status.update(label=f"{label}: Job nicht mehr bekannt", state="error", expanded=True)
                st.error(f"{e} — wurde das Backend neu gestartet?")
                return None
            except Exception as e:
                failures += 1
                if failures >= _POLL_MAX_FAILURES:
                    progress_line.empty()
                    status.update(label=f"{label}: Backend nicht erreichbar", state="error", expanded=True)
                    st.error(f"Status konnte nicht abgefragt werden ({e}). Läuft das Backend noch, "
                             "erscheint das Ergebnis später unter „Ergebnisse“.")
                    return None
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


def _load_stats(details: list[dict]) -> str:
    """p50/p95 latency and Ø/max chunks per question — what `pytest -m perf` watches against its baseline,
    computed the same way (median; p95 as nearest rank), so both numbers can be compared directly.

    Computed from the per-question details (also works for reports saved before these numbers existed).
    Chunks = everything handed to the LLM, section expansion included — the number roadmap 5.6 is about.
    """
    answered = [d for d in details if "error" not in d]
    if not answered:
        return ""
    latencies = sorted(d.get("latency_ms", 0) for d in answered)
    p50 = statistics.median(latencies)
    p95 = latencies[math.ceil(len(latencies) * 0.95) - 1]   # nearest rank — same as tests/perf
    chunks = [len(d.get("retrieved_files", [])) for d in answered]
    return (f"Latenz p50 {p50:.0f} ms · p95 {p95} ms · Chunks pro Frage Ø {sum(chunks) / len(chunks):.1f} / "
            f"max {max(chunks)}")


def _render_retrieval_result(result: dict):
    """
    Render a retrieval eval result containing metrics, taxonomic breakdowns,
    and per-question hit/miss deep dives.
    """
    details = result.get("details", [])
    errors = result.get("errors", [])
    breakdown = result.get("breakdown_by_source_type", {})
    config = result.get("config", {})

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hit Rate", f"{result.get('hit_rate', 0):.1%}")
    c2.metric("Fragen", result.get("total_questions", 0))
    avg_lat = sum(d.get("latency_ms", 0) for d in details) / max(len(details), 1)
    c3.metric("Ø Latenz", f"{avg_lat:.0f} ms")
    c4.metric(
        "Config",
        f"K={config.get('top_k', '?')} R={config.get('top_k_rerank', '?')} "
        f"C={config.get('max_per_source', '?')} H={_hybrid_label(config.get('hybrid'))}",
    )
    if stats := _load_stats(details):
        st.caption(stats)
    if result.get("section_total"):
        sec_rate = result.get("section_hit_rate") or 0
        st.caption(
            f"Abschnitt-Treffer (expected_section): {sec_rate:.1%} "
            f"({round(sec_rate * result['section_total'])}/{result['section_total']})"
        )

    # Breakdown
    if breakdown:
        st.markdown("**Breakdown nach Source-Type**")
        bd_rows = [
            {"Type": st_name, "Hit Rate": f"{v['hit_rate']:.1%}", "Hits": v["hits"], "Total": v["total"]}
            for st_name, v in breakdown.items()
        ]
        st.dataframe(pd.DataFrame(bd_rows), hide_index=True, width="stretch")

    # Details table
    if details:
        st.markdown("**Details**")
        for d in details:
            hit = d.get("hit", False)
            icon = "✅" if hit else "❌"
            # Right file, wrong section (typical for the rulebook PDF)
            if hit and d.get("section_hit") is False:
                icon = "⚠️"
            qid = d.get("id", "?")
            question = d.get("question", "")
            score = d.get("top_score", 0)

            with st.expander(f"{icon} [{qid}] {question[:80]} — Score: {score:.3f}"):
                col_exp, col_ret = st.columns(2)
                with col_exp:
                    st.markdown("**Erwartet:**")
                    for s in d.get("expected_sources", []):
                        st.code(s, language=None)
                    if d.get("expected_source_ids"):
                        st.caption("aus: " + ", ".join(d["expected_source_ids"]))
                with col_ret:
                    st.markdown("**Gefunden:**")
                    sids = d.get("retrieved_source_ids") or [""] * len(d.get("retrieved_files", []))
                    for s, sid in list(zip(d.get("retrieved_files", []), sids, strict=False))[:5]:
                        st.code(f"{s}  ({sid})" if sid else s, language=None)
                if d.get("expected_section"):
                    sec_icon = "✅" if d.get("section_hit") else "❌"
                    st.markdown(f"**Abschnitt {d['expected_section']}:** {sec_icon}")
                    for h in d.get("retrieved_headings", [])[:8]:
                        st.code(h or "—", language=None)

    if errors:
        st.warning(f"{len(errors)} Fehler aufgetreten")
        for e in errors:
            st.code(f"[{e.get('id', '?')}] {e.get('error', '')}", language=None)


# Short in-app guide, kept as Markdown next to the pages (tables can't be wrapped in code);
# the full reference is docs/evaluation.md in the repo
_GUIDE = (Path(__file__).resolve().parents[1] / "evaluation_guide.md").read_text(encoding="utf-8")


# ─── Page ─────────────────────────────────────────────────────────────

st.title("🎯 Evaluation")
st.caption("Golden Set verwalten, Retrieval testen und tunen, Ergebnisse vergleichen.")

with st.expander("ℹ️ Anleitung: Was die Tabs prüfen und wann man sie braucht"):
    st.markdown(_GUIDE)

tab_gs, tab_test, tab_ret, tab_e2e, tab_results, tab_fb, tab_chunks = st.tabs([
    "Golden Set", "Frage testen", "Retrieval Eval", "End-to-End", "Ergebnisse", "Feedback", "Chunks",
])


# ═══ Tab 1: Golden Set ═══════════════════════════════════════════════

@st.cache_data(ttl=10, show_spinner=False)
def _fetch_qa_pairs(api_url: str):
    """Fetch the active pool of Golden Set test questions targeting expected metadata."""
    return get_json(f"{api_url}/eval/qa-pairs", timeout=5)


@st.cache_data(ttl=15, show_spinner=False)
def _fetch_eval_results(api_url: str):
    """Fetch history references for saved evaluation runs."""
    return get_json(f"{api_url}/eval/results", timeout=5)


with tab_gs:
    try:
        qa_data = _fetch_qa_pairs(API_URL)
        pairs = qa_data.get("pairs", [])
    except Exception as e:
        st.error(f"Backend nicht erreichbar: {e}")
        pairs = []

    rows = []
    for p in pairs:
        rows.append({
            "id": p.get("id", ""),
            "question": p.get("question", ""),
            "source_type": p.get("source_type", "markdown"),
            "category": p.get("category", ""),
            "expected_sources": ", ".join(p.get("expected_sources", [])),
            "expected_answer_contains": ", ".join(p.get("expected_answer_contains", [])),
            "expected_section": p.get("expected_section", ""),
            "expected_source_ids": ", ".join(p.get("expected_source_ids", [])),
            "notes": p.get("notes", ""),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["id", "question", "source_type", "category",
                 "expected_sources", "expected_answer_contains", "expected_section",
                 "expected_source_ids", "notes"]
    )

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "question": st.column_config.TextColumn("Frage"),
            "source_type": st.column_config.SelectboxColumn(
                "Source Type", options=["markdown", "pdf", "image", "mixed", "simulation"], width="small",
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
            "expected_section": st.column_config.TextColumn(
                "Abschnitt",
                help="Abschnitt, in dem ein Treffer liegen muss: Nummer (z.B. 4.10) oder Überschrift",
                width="small",
            ),
            "expected_source_ids": st.column_config.TextColumn(
                "Source-IDs",
                help="Optional, komma-getrennt: die erwartete Datei muss aus einer dieser Quellen kommen "
                     "(für gleichnamige Dateien, z.B. zwei llm_report.md)",
                width="small",
            ),
            "notes": st.column_config.TextColumn("Notizen"),
        },
        hide_index=True,
        width="stretch",
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
                "expected_sources": _csv(row.get("expected_sources")),
                "expected_answer_contains": _csv(row.get("expected_answer_contains")),
                "expected_section": (row.get("expected_section") or "").strip(),
                "expected_source_ids": _csv(row.get("expected_source_ids")),
                "notes": (row.get("notes") or "").strip(),
            })
        try:
            resp = requests.put(
                f"{API_URL}/eval/qa-pairs",
                json={"pairs": save_pairs},
                timeout=10,
            )
            raise_for_status(resp)
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
        t_hybrid = _render_hybrid_selector("test")

    # The preview lives in session_state: the "➕ Hinzufügen" click below reruns the page, and a
    # result shown only inside `if st.button("🔍 Testen")` would be gone by then (the add never ran).
    if st.button("🔍 Testen", key="test_run", disabled=not test_question.strip()):
        try:
            resp = requests.post(
                f"{API_URL}/eval/preview",
                json={
                    "question": test_question.strip(),
                    "top_k": t_top_k,
                    "top_k_rerank": t_top_k_rerank,
                    "max_per_source": t_max_per_source,
                    "hybrid": t_hybrid,
                },
                timeout=30,
            )
            raise_for_status(resp)
            st.session_state["_preview"] = {
                "question": test_question.strip(),
                "result": resp.json(),
                "run": st.session_state.get("_preview", {}).get("run", 0) + 1,  # fresh form per test
            }
        except Exception as e:
            st.session_state.pop("_preview", None)
            st.error(f"Fehler: {e}")

    if preview := st.session_state.get("_preview"):
        chunks = preview["result"].get("chunks", [])
        latency = preview["result"].get("latency_ms", 0)
        st.caption(f"„{preview['question']}“ — {len(chunks)} Chunks in {latency:.0f} ms")

        if chunks:
            st.dataframe(pd.DataFrame([{
                "Score": round(c["score"], 4),
                "Source": c["source_file"],
                "Heading": c.get("heading") or "",
                "Preview": c["content_preview"][:100],
            } for c in chunks]), hide_index=True, width="stretch")

            # Convenience: add the tested question to the Golden Set
            st.divider()
            st.markdown("**Zum Golden Set hinzufügen:**")
            run = preview["run"]
            c1, c2, c3 = st.columns(3)
            with c1:
                add_id = st.text_input("ID", key=f"test_add_id_{run}", placeholder="m099")
            with c2:
                add_type = st.selectbox("Source Type", ["markdown", "pdf", "image", "simulation"],
                                        key=f"test_add_type_{run}")
            with c3:
                add_cat = st.text_input("Kategorie", key=f"test_add_cat_{run}")

            add_sources = st.text_input(
                "Expected Sources (komma-getrennt)", key=f"test_add_sources_{run}",
                value=", ".join(dict.fromkeys(c["source_file"] for c in chunks[:3])),
            )

            if st.button("➕ Hinzufügen", key=f"test_add_btn_{run}"):
                if not add_id.strip():
                    st.error("ID ist Pflichtfeld.")
                else:
                    try:
                        existing_pairs = get_json(f"{API_URL}/eval/qa-pairs", timeout=5).get("pairs", [])
                        if any(p.get("id") == add_id.strip() for p in existing_pairs):
                            st.error(f"ID '{add_id.strip()}' gibt es im Golden Set schon.")
                        else:
                            existing_pairs.append({
                                "id": add_id.strip(),
                                "question": preview["question"],
                                "source_type": add_type,
                                "category": (add_cat or "").strip(),
                                "expected_sources": _csv(add_sources),
                                "expected_answer_contains": [],
                                "notes": "",
                            })
                            resp = requests.put(f"{API_URL}/eval/qa-pairs",
                                                json={"pairs": existing_pairs}, timeout=10)
                            raise_for_status(resp)
                            _fetch_qa_pairs.clear()
                            st.success(f"'{add_id.strip()}' zum Golden Set hinzugefügt.")
                    except Exception as e:
                        st.error(f"Fehler: {e}")
        else:
            st.info("Keine Chunks gefunden.")


# ═══ Tab 3: Retrieval Evaluation ═════════════════════════════════════

def _start_retrieval_eval(top_k, top_k_rerank, max_per_source, hybrid, label):
    """POST a retrieval-eval job, poll it to completion, and return the saved result file."""
    resp = requests.post(
        f"{API_URL}/eval/run",
        json={
            "top_k": top_k,
            "top_k_rerank": top_k_rerank,
            "max_per_source": max_per_source,
            "eval_type": "retrieval",
            "hybrid": hybrid,
        },
        timeout=10,
    )
    raise_for_status(resp)
    job_id = resp.json()["job_id"]
    return _poll_eval_job(job_id, label=label)


with tab_ret:
    st.markdown("Führt das komplette Golden Set durch den Retriever (kein LLM).")

    with st.expander("Retrieval-Parameter", expanded=False):
        r_top_k, r_top_k_rerank, r_max_per_source = _render_tuning_sliders("ret")
        r_hybrid = _render_hybrid_selector("ret")

    col_single, col_ab = st.columns([1, 1])

    with col_single:
        run_single = st.button("▶ Evaluation starten", type="primary", key="ret_run")

    with col_ab:
        run_ab = st.button(
            "⚖ A/B starten (Vektor vs. Hybrid)",
            key="ret_ab",
            help="Führt die Evaluation zweimal aus — einmal ohne, einmal mit Hybrid — "
                 "und zeigt den Vergleich direkt an.",
        )

    if run_single:
        try:
            result_file = _start_retrieval_eval(
                r_top_k, r_top_k_rerank, r_max_per_source, r_hybrid,
                label="Retrieval Evaluation",
            )
            _fetch_eval_results.clear()
            if result_file:
                result = get_json(f"{API_URL}/eval/results/{result_file}", timeout=10)
                _render_retrieval_result(result)
        except Exception as e:
            _show_eval_error(e)

    if run_ab:
        try:
            # Sequential runs — backend enforces a single running job at a time.
            file_vec = _start_retrieval_eval(
                r_top_k, r_top_k_rerank, r_max_per_source, hybrid=False,
                label="A/B — Vektor only",
            )
            file_hyb = None
            if file_vec:
                file_hyb = _start_retrieval_eval(
                    r_top_k, r_top_k_rerank, r_max_per_source, hybrid=True,
                    label="A/B — Hybrid",
                )
            _fetch_eval_results.clear()

            if file_vec and file_hyb:
                res_a = get_json(f"{API_URL}/eval/results/{file_vec}", timeout=10)
                res_b = get_json(f"{API_URL}/eval/results/{file_hyb}", timeout=10)

                hr_a = res_a.get("hit_rate", 0)
                hr_b = res_b.get("hit_rate", 0)

                st.subheader("A/B-Vergleich")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Vektor only", f"{hr_a:.1%}")
                mc2.metric("Hybrid", f"{hr_b:.1%}", delta=f"{(hr_b - hr_a):+.1%}")
                mc3.metric("Fragen", res_a.get("total_questions", 0))

                flipped = _flipped(res_a, res_b, "Vektor", "Hybrid")

                if flipped:
                    wins = sum(1 for f in flipped if f["Hybrid"] == "✅")
                    losses = len(flipped) - wins
                    st.markdown(
                        f"**{len(flipped)} Fragen geändert** — "
                        f"Hybrid gewinnt: **{wins}**, verliert: **{losses}**"
                    )
                    st.dataframe(pd.DataFrame(flipped), hide_index=True, width="stretch")
                else:
                    st.info("Keine Unterschiede bei Hit/Miss zwischen Vektor und Hybrid.")

                with st.expander("Einzelergebnisse einsehen"):
                    st.markdown("**Vektor only**")
                    _render_retrieval_result(res_a)
                    st.divider()
                    st.markdown("**Hybrid**")
                    _render_retrieval_result(res_b)

        except Exception as e:
            _show_eval_error(e)

    # Search exactness: is hnsw_ef_search still large enough for the index? (roadmap 5.5)
    st.divider()
    st.markdown("**🎯 Suchgenauigkeit** — findet die Vektorsuche dieselben Kandidaten wie eine exakte Suche?")
    st.caption("Nach größeren Index-Änderungen (neue Quellen, viele neue Dokumente) einmal prüfen. "
               "Nur lesend, dauert einige Sekunden.")
    if st.button("🎯 Suchgenauigkeit prüfen", key="ret_exact"):
        try:
            resp = requests.post(f"{API_URL}/eval/search-exactness", timeout=300)
            raise_for_status(resp)
            ex = resp.json()
            summary = (f"{ex['exact']}/{ex['questions']} Golden-Fragen mit exakten Top-{ex['top_k']} · "
                       f"{ex['chunks']} Chunks · `hnsw_ef_search` {ex['ef_search']} · {ex['seconds']:.1f} s")
            if not ex["deviations"]:
                st.success(f"Exakt: {summary}")
            else:
                st.warning(f"Nicht exakt: {summary}. Die Suche verpasst bei diesen Fragen Chunks, die eine "
                           "exakte Suche finden würde — `vectorstore.hnsw_ef_search` in `settings.yaml` "
                           "erhöhen (z.B. verdoppeln), Backend neu starten, erneut prüfen.")
                st.dataframe(pd.DataFrame([{"ID": d["id"], "Frage": d["question"][:80], "verpasst": d["missed"]}
                                           for d in ex["deviations"]]), hide_index=True, width="stretch")
        except Exception as e:
            _show_eval_error(e)


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
            raise_for_status(resp)
            job_id = resp.json()["job_id"]
            result_file = _poll_eval_job(job_id, label="End-to-End Evaluation")
            _fetch_eval_results.clear()

            if result_file:
                result = get_json(f"{API_URL}/eval/results/{result_file}", timeout=10)

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

        except Exception as e:
            _show_eval_error(e)


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
                        resp = requests.delete(f"{API_URL}/eval/results/{r['filename']}", timeout=5)
                        raise_for_status(resp)
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
                        res_a = get_json(f"{API_URL}/eval/results/{sel_a}", timeout=10)
                        res_b = get_json(f"{API_URL}/eval/results/{sel_b}", timeout=10)

                        # Metric comparison
                        st.markdown("**Metriken**")
                        mc1, mc2, mc3 = st.columns(3)
                        hr_a = res_a.get("hit_rate", 0)
                        hr_b = res_b.get("hit_rate", 0)
                        mc1.metric("Hit Rate A", f"{hr_a:.1%}")
                        mc2.metric("Hit Rate B", f"{hr_b:.1%}")
                        diff = hr_b - hr_a
                        mc3.metric("Differenz", f"{diff:+.1%}")

                        flipped = _flipped(res_a, res_b, "A", "B")

                        if flipped:
                            st.markdown(f"**{len(flipped)} Fragen mit geändertem Ergebnis:**")
                            st.dataframe(pd.DataFrame(flipped), hide_index=True,
                                         width="stretch")
                        else:
                            st.success("Keine Unterschiede bei Hit/Miss.")

                    except Exception as e:
                        st.error(f"Fehler: {e}")
            elif sel_a == sel_b:
                st.caption("Wähle zwei verschiedene Ergebnisse.")


# ═══ Tab 6: Feedback ═════════════════════════════════════════════════

@st.cache_data(ttl=10, show_spinner=False)
def _fetch_feedback(api_url: str):
    """Current 👍/👎 per chat answer, newest first."""
    return get_json(f"{api_url}/feedback", timeout=5)


with tab_fb:
    st.markdown("Bewertungen aus dem Chat (👍/👎, letzte Bewertung je Antwort). "
                "Ausgewählte Fragen lassen sich ins Golden Set übernehmen.")
    try:
        feedback = _fetch_feedback(API_URL)
    except Exception as e:
        st.error(f"Feedback konnte nicht geladen werden: {e}")
        feedback = []

    show_up = st.toggle("Auch 👍 anzeigen", key="fb_show_up")
    shown = [e for e in feedback if e.get("rating") == "down" or show_up]
    if not shown:
        st.caption("Noch keine Bewertungen." if not feedback else "Keine 👎-Bewertungen — 👍 über den Schalter.")
    else:
        fb_df = pd.DataFrame([{
            "add": False,
            "rating": "👍" if e.get("rating") == "up" else "👎",
            "ts": e.get("ts", ""),
            "role": "Spieler" if e.get("role") == "player" else "Spielleiter",
            "question": e.get("question", ""),
            "answer": (e.get("answer", "") or "")[:300],
            "sources": ", ".join(e.get("sources", [])),
            "model": e.get("model", ""),
        } for e in shown])
        fb_edited = st.data_editor(
            fb_df,
            column_config={
                "add": st.column_config.CheckboxColumn("➕", help="Ins Golden Set übernehmen", width="small"),
                "rating": st.column_config.TextColumn("", width="small"),
                "ts": st.column_config.TextColumn("Zeit", width="small"),
                "role": st.column_config.TextColumn("Rolle", width="small"),
                "question": st.column_config.TextColumn("Frage"),
                "answer": st.column_config.TextColumn("Antwort (Anfang)"),
                "sources": st.column_config.TextColumn("Quellen"),
                "model": st.column_config.TextColumn("Modell", width="small"),
            },
            disabled=["rating", "ts", "role", "question", "answer", "sources", "model"],
            hide_index=True,
            width="stretch",
            # Ticks are kept by row position — a different list (👍 toggle, new ratings) must not
            # carry them over to other answers, so the editor gets a fresh key per shown list
            key="fb_editor_" + hashlib.sha1("|".join(e.get("message_id", "") for e in shown).encode()).hexdigest()[:12],
        )
        if st.button("➕ Ausgewählte ins Golden Set übernehmen", key="fb_add"):
            chosen = [e for e, add in zip(shown, fb_edited["add"], strict=True) if add]
            if not chosen:
                st.warning("Keine Zeile ausgewählt.")
            else:
                try:
                    existing = get_json(f"{API_URL}/eval/qa-pairs", timeout=5).get("pairs", [])
                    new_pairs = pairs_from_feedback(chosen, existing)
                    if not new_pairs:
                        st.info("Alle ausgewählten Fragen stehen schon im Golden Set.")
                    else:
                        resp = requests.put(f"{API_URL}/eval/qa-pairs",
                                            json={"pairs": existing + new_pairs}, timeout=10)
                        raise_for_status(resp)
                        _fetch_qa_pairs.clear()
                        down = sum(1 for p in new_pairs if not p["expected_sources"])
                        st.success(f"{len(new_pairs)} Fragen übernommen ({', '.join(p['id'] for p in new_pairs)})."
                                   + (f" Bei {down} 👎-Fragen im Tab „Golden Set“ die erwartete Quelle eintragen."
                                      if down else ""))
                except Exception as e:
                    st.error(f"Übernehmen fehlgeschlagen: {e}")


# ═══ Tab 7: Chunk-Inspector ══════════════════════════════════════════

_SHORT_CHUNK_CHARS = 100  # below this a chunk carries little meaning on its own


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_sources(api_url: str):
    """Configured sources for the inspector's source picker."""
    return get_json(f"{api_url}/sources", timeout=5).get("sources", [])


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_chunk_files(api_url: str, source_id: str):
    """Indexed files of one source; a ChromaDB error raises (shown as an error, not as an empty list)."""
    resp = requests.get(f"{api_url}/eval/chunks/files", params={"source_id": source_id}, timeout=10)
    raise_for_status(resp)
    return resp.json().get("files", [])


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_file_chunks(api_url: str, source_id: str, source_file: str):
    """All chunks of one file in document order."""
    resp = requests.get(f"{api_url}/eval/chunks", params={"source_id": source_id, "source_file": source_file},
                        timeout=10)
    raise_for_status(resp)
    return resp.json()["chunks"]


with tab_chunks:
    st.markdown("Alle gespeicherten Chunks einer Datei — Text und Metadaten so, wie der Retriever sie sieht.")
    try:
        ci_sources = _fetch_sources(API_URL)
    except Exception as e:
        st.error(f"Quellen konnten nicht geladen werden: {e}")
        ci_sources = []

    c1, c2 = st.columns([1, 2])
    with c1:
        ci_sid = st.selectbox("Quelle", [s["id"] for s in ci_sources], index=None,
                              placeholder="Quelle wählen", key="ci_source")
    ci_files: list[str] = []
    ci_failed = False
    if ci_sid:
        try:
            ci_files = _fetch_chunk_files(API_URL, ci_sid)
        except Exception as e:
            ci_failed = True   # an error is not "no files" — don't show both
            st.error(f"Dateien konnten nicht geladen werden: {e}")
    with c2:
        # Key per source: a file picked in one source is no valid option in the next
        ci_file = st.selectbox("Datei", ci_files, index=None, placeholder="Datei wählen",
                               key=f"ci_file_{ci_sid}", disabled=not ci_files)
    if ci_sid and not ci_files and not ci_failed:
        st.caption("Keine indizierten Dateien in dieser Quelle.")

    if ci_sid and ci_file:
        try:
            ci_chunks = _fetch_file_chunks(API_URL, ci_sid, ci_file)
        except Exception as e:
            st.error(f"Chunks konnten nicht geladen werden: {e}")
            ci_chunks = []

        if ci_chunks:
            sizes = [c["chars"] for c in ci_chunks]
            short = sum(1 for n in sizes if n < _SHORT_CHUNK_CHARS)
            st.caption(f"{len(ci_chunks)} Chunks · Zeichen min {min(sizes)} / Ø {sum(sizes) // len(sizes)} / "
                       f"max {max(sizes)}"
                       + (f" · ⚠️ {short} unter {_SHORT_CHUNK_CHARS} Zeichen" if short else ""))
            st.dataframe(pd.DataFrame([{
                "#": c["chunk_index"],
                "Überschrift": c["heading"],
                "Zeichen": c["chars"],
                "Anfang": c["content"][:120],
            } for c in ci_chunks]), hide_index=True, width="stretch")

            ci_pick = st.selectbox(
                "Chunk anzeigen", range(len(ci_chunks)), key=f"ci_chunk_{ci_sid}_{ci_file}",
                format_func=lambda i: f"#{ci_chunks[i]['chunk_index']} — "
                                      f"{ci_chunks[i]['heading'] or '(ohne Überschrift)'}",
            )
            st.code(ci_chunks[ci_pick]["content"], language=None, wrap_lines=True)
            with st.expander("Metadaten"):
                st.json(ci_chunks[ci_pick]["metadata"])
