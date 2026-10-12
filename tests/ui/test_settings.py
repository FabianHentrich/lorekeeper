"""Settings page: saving shows a visible confirmation, a rejected value shows the backend's reason."""
from tests.ui.conftest import open_page


def _open(fake_backend):
    at = open_page("settings")
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_save_sends_only_the_change_and_shows_saved_after_the_rerun(fake_backend):
    at = _open(fake_backend)
    at.slider(key="ret_top_k").set_value(20).run()
    at.button(key="save_retrieval").click().run()

    assert not at.exception, [e.message for e in at.exception]
    assert fake_backend.bodies["PUT /config"] == [{"retrieval": {"top_k": 20}}]
    assert any("Gespeichert." in s.value for s in at.success)   # survived the st.rerun()


def test_unchanged_section_makes_no_request(fake_backend):
    at = _open(fake_backend)
    at.button(key="save_conv").click().run()

    assert "PUT /config" not in fake_backend.calls
    assert any("Keine Änderungen" in i.value for i in at.info)


def test_rejected_value_shows_the_reason(fake_backend):
    fake_backend.errors["PUT /config"] = (422, {"detail": [
        {"msg": "Value error, chunk_overlap (60) muss kleiner als max_chunk_size (50) sein"}]})
    at = _open(fake_backend)
    at.number_input(key="chunk_max").set_value(50).run()
    at.number_input(key="chunk_overlap").set_value(60).run()
    at.button(key="save_chunking").click().run()

    assert not at.exception, [e.message for e in at.exception]
    assert any("muss kleiner als max_chunk_size" in e.value for e in at.error)
    assert not any("Gespeichert" in s.value for s in at.success)
