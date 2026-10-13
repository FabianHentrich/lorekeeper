"""Behaviour that the clean-code simplification (roadmap 7.4, package C) merged or moved."""
from tests.ui import conftest
from tests.ui.conftest import open_page


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


def _sidebar_with_gemini(monkeypatch, source: str):
    state = {**conftest.PAYLOADS["/sidebar-state"],
             "gemini_status": {"has_key": source != "none", "source": source}}
    monkeypatch.setitem(conftest.PAYLOADS, "/sidebar-state", state)


def test_empty_gemini_key_is_rejected_without_a_request(fake_backend):
    at = _ok(open_page("chat").run())
    _ok(at.sidebar.button(key="_gemini_set").click().run())

    assert "Kein Key eingegeben." in [e.value for e in at.sidebar.error]
    assert "POST /provider/gemini/key" not in fake_backend.calls


def test_gemini_key_is_sent_and_confirmed_after_the_rerun(fake_backend):
    at = _ok(open_page("chat").run())
    at.sidebar.text_input(key="_gemini_key").input("  abc  ")
    _ok(at.sidebar.button(key="_gemini_set").click().run())

    assert fake_backend.bodies["POST /provider/gemini/key"] == [{"api_key": "abc"}]
    assert "Gemini-Key gesetzt." in [s.value for s in at.sidebar.success]


def test_only_a_ui_key_can_be_removed(fake_backend, monkeypatch):
    _sidebar_with_gemini(monkeypatch, "env")
    at = _ok(open_page("chat").run())
    assert not [b for b in at.sidebar.button if b.key == "_gemini_clear"]

    conftest.PAYLOADS["/sidebar-state"]["gemini_status"]["source"] = "runtime"
    import streamlit as st
    st.cache_data.clear()
    at = _ok(open_page("chat").run())
    _ok(at.sidebar.button(key="_gemini_clear").click().run())
    assert fake_backend.bodies["POST /provider/gemini/key"] == [{"api_key": None}]


def test_new_session_resets_chat_and_token_counter(fake_backend):
    at = open_page("chat")
    at.session_state["messages"] = [{"role": "user", "content": "x"}]
    at.session_state["session_id"] = "s1"
    at.session_state["session_usage"] = {"tokens_in": 5, "tokens_out": 5, "tokens_thinking": 0}
    _ok(at.run())
    _ok(next(b for b in at.sidebar.button if b.label == "🗑️ Neue Session").click().run())

    assert at.session_state["messages"] == []
    assert at.session_state["session_id"] is None
    assert at.session_state["session_usage"] == {"tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0}


def test_folder_mapping_saves_only_entries_that_differ_from_the_source_defaults(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/sources/vault/folders",
                        {"folders": [{"name": "Regeln", "type": "folder"}, {"name": "NPCs", "type": "folder"}]})
    at = _ok(open_page("sources").run())
    _ok(at.button(key="savemap_vault").click().run())

    saved = fake_backend.bodies["PUT /sources"][0]["sources"][0]
    # NPCs keeps the defaults (misc, lore) — falls back to them without an entry
    assert saved["category_map"] == {"Regeln": {"category": "misc", "group": "rules"}}


def test_recategorize_is_one_button_for_all_sources(fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/sources/recategorize", {"updated": 3})
    at = _ok(open_page("sources").run())
    buttons = [b for b in at.button if "Recategorize" in b.label]
    assert len(buttons) == 1

    _ok(buttons[0].click().run())
    assert fake_backend.calls.count("POST /sources/recategorize") == 1
    assert any("updated" in s.value for s in at.success)


def test_compare_lists_only_questions_whose_hit_changed(fake_backend, monkeypatch):
    def result(hits):
        return {"hit_rate": sum(hits.values()) / len(hits),
                "details": [{"id": q, "question": f"Frage {q}", "hit": h} for q, h in hits.items()]}
    entry = {"eval_type": "retrieval", "hit_rate": 0.5, "total_questions": 3, "config": {}}
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results",
                        [{**entry, "filename": "a.json"}, {**entry, "filename": "b.json"}])
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results/a.json", result({"q1": True, "q2": False, "q3": True}))
    monkeypatch.setitem(conftest.PAYLOADS, "/eval/results/b.json", result({"q1": True, "q2": True, "q3": False}))

    at = _ok(open_page("evaluation").run())
    _ok(at.button(key="cmp_run").click().run())

    table = at.dataframe[-1].value
    assert list(table["ID"]) == ["q2", "q3"]
    assert list(table["A"]) == ["❌", "✅"] and list(table["B"]) == ["✅", "❌"]
