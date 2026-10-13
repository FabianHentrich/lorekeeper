"""Sources page: saving the source table and adding a source (roadmap 9.0 — page had 69 % coverage).

The source table edits category_map as text ("Ordner→kategorie:gruppe"); saving it unchanged must give
back exactly the stored config — this conversion had two drifted copies before (roadmap 7.4 C).
"""
import pytest

from tests.ui import conftest
from tests.ui.conftest import open_page


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


def _click(at, label):
    return _ok(next(b for b in at.button if b.label == label).click().run())


@pytest.fixture
def sources(fake_backend, monkeypatch):
    stored = [{"id": "vault", "path": "C:/Vault", "group": "lore", "default_category": "misc",
               "category_map": {"Regeln": {"category": "regel", "group": "rules"}, "NPCs": "npc"},
               "exclude_patterns": ["*.tmp"], "player_visible": True}]
    monkeypatch.setitem(conftest.PAYLOADS, "/sources", {"sources": stored})
    return stored


def test_saving_the_unchanged_table_gives_back_the_stored_config(sources, fake_backend):
    at = _click(open_page("sources").run(), "💾 Änderungen speichern")

    [body] = fake_backend.bodies["PUT /sources"]
    assert body["sources"] == sources          # category_map text round trip, exclude_patterns kept
    assert not at.error


def test_adding_a_scanned_folder_source(sources, fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/sources/scan", {"folders": [{"name": "Orte", "type": "folder"}],
                                                             "is_file": False})
    at = _ok(open_page("sources").run())
    at.text_input(key="_new_source_path").input("D:/Abenteuer/Kampagne 2")
    _ok(at.button(key="_scan_btn").click().run())

    assert at.text_input(key="_new_source_id").value == "kampagne-2"    # suggested from the folder name
    at.selectbox(key="_new_source_group").select("adventure")
    _ok(at.button(key="_add_source_final").click().run())

    [body] = fake_backend.bodies["PUT /sources"]
    assert body["sources"][:-1] == sources                                # existing ones untouched
    assert body["sources"][-1] == {"id": "kampagne-2", "path": "D:/Abenteuer/Kampagne 2", "group": "adventure",
                                   "default_category": "misc", "category_map": {}, "exclude_patterns": []}


def test_an_existing_id_is_refused_without_a_request(sources, fake_backend, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/sources/scan", {"folders": [], "is_file": True})
    at = _ok(open_page("sources").run())
    at.text_input(key="_new_source_path").input("C:/vault.md")
    _ok(at.button(key="_scan_btn").click().run())
    _ok(at.button(key="_add_source_final").click().run())

    assert "ID 'vault' existiert bereits." in [e.value for e in at.error]
    assert "PUT /sources" not in fake_backend.calls


def test_scan_error_is_shown(sources, monkeypatch):
    monkeypatch.setitem(conftest.PAYLOADS, "/sources/scan",
                        {"folders": [], "is_file": False, "error": "Pfad existiert nicht: X:/weg"})
    at = _ok(open_page("sources").run())
    at.text_input(key="_new_source_path").input("X:/weg")
    _ok(at.button(key="_scan_btn").click().run())

    assert "Pfad existiert nicht: X:/weg" in [e.value for e in at.error]
