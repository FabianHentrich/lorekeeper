"""Evaluation → Chunks tab (roadmap 3.3): pick a source and a file, see every stored chunk."""
from tests.ui.conftest import open_page


def _open(fake_backend):
    at = open_page("evaluation")
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_nothing_loaded_before_a_source_is_picked(fake_backend):
    """Only the cached source list — files and chunks are fetched on demand."""
    _open(fake_backend)
    assert not any(c.startswith("GET /eval/chunks") for c in fake_backend.calls)


def test_picking_source_and_file_shows_chunks(fake_backend):
    at = _open(fake_backend)
    at.selectbox(key="ci_source").set_value("vault").run()
    assert "GET /eval/chunks/files" in fake_backend.calls
    assert fake_backend.bodies["GET /eval/chunks/files"] == [None]  # source_id travels as query param

    at.selectbox(key="ci_file_vault").set_value("Arkenfeld.md").run()
    assert not at.exception, [e.message for e in at.exception]
    assert "GET /eval/chunks" in fake_backend.calls
    assert any("2 Chunks" in c.value and "1 unter 100 Zeichen" in c.value for c in at.caption)
    assert list(at.dataframe[-1].value["Überschrift"]) == ["Arkenfeld", "Arkenfeld > Familie"]
    assert at.code[-1].value == "Arkenfeld ist ein Zeitmagier."

    at.selectbox(key="ci_chunk_vault_Arkenfeld.md").set_value(1).run()
    assert at.code[-1].value == "Seine Schwester lebt in der Akademie."
