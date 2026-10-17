"""Chat: how an answer's sources, tokens and timings are shown (roadmap 9.0 — chat page had 78 %)."""
from tests.ui import conftest
from tests.ui.conftest import open_page

SOURCES = [
    {"file": "Arkenfeld.md", "source_path": r"C:\Vault\Orte\Arkenfeld Stadt.md", "document_type": "markdown",
     "heading": "Arkenfeld > Lage", "chunk_preview": "Liegt im Norden.",
     "content": "Arkenfeld > Lage\n\nLiegt im Norden, am Fluss Veren.", "score": 0.9},
    {"file": "Arkenfeld.md", "source_path": r"C:\Vault\Orte\Arkenfeld Stadt.md", "document_type": "markdown",
     "heading": "Arkenfeld > Rat", "chunk_preview": "Sieben Magier.", "score": 0.7},  # no content: older message
    {"file": "regeln.pdf", "source_path": "", "document_type": "pdf", "heading": None,
     "chunk_preview": "", "content": "|**Dolch**<br>**leicht**|STÄ + 2|", "score": 0.5},
    {"file": "karte.png", "source_path": r"C:\gibt-es-nicht\karte.png", "document_type": "image",
     "heading": None, "chunk_preview": "", "score": 0.4},
]


def _ask(monkeypatch):
    done = {**conftest._STREAM_EVENTS[-1], "sources": SOURCES,
            "timings": {"first_token_ms": 1100, "total_ms": 3000, "retrieval_ms": 500, "condense_ms": 800,
                        "chunks": 8},
            "usage": {"tokens_in": 10, "tokens_out": 5, "tokens_thinking": 3}}
    monkeypatch.setattr(conftest, "_STREAM_EVENTS", [*conftest._STREAM_EVENTS[:-1], done])
    at = open_page("chat").run()
    at.chat_input[0].set_value("Wer regiert Arkenfeld?").run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_sources_are_grouped_per_file_with_a_link_and_best_score(fake_backend, monkeypatch):
    at = _ask(monkeypatch)
    texts = [m.value for m in at.markdown]

    # one header per file; Windows path → file:/// link with the space encoded
    assert ("📄 [**Arkenfeld.md**](file:///C:/Vault/Orte/Arkenfeld%20Stadt.md) — Best Score: 0.90 · 2 Chunks"
            in texts)
    assert "📄 **regeln.pdf** — Best Score: 0.50" in texts               # no path → no link
    labels = [e.label for e in at.expander]
    assert "↳ Arkenfeld > Rat · Score 0.70" in labels
    assert "↳ — · Score 0.50" in labels                                    # chunk without heading
    assert "Bild nicht gefunden: karte.png" in [w.value for w in at.warning]


def test_each_chunk_can_be_read_under_its_source(fake_backend, monkeypatch):
    """Roadmap 3.16: the text the LLM got, under the source — heading line cut, PDF <br> as a line break."""
    texts = [m.value for m in _ask(monkeypatch).markdown]
    assert "Liegt im Norden, am Fluss Veren." in texts                    # heading prefix removed
    assert "Sieben Magier." in texts                                      # no content → the preview
    assert "|**Dolch**  \n**leicht**|STÄ + 2|" in texts


def test_images_of_the_sources_show_under_the_answer(fake_backend, monkeypatch, tmp_path):
    """Roadmap 2: each existing image once, in source order; a missing file is skipped, and only notes
    named in the question or answer count."""
    import base64
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    karte, malek = tmp_path / "Karte.png", tmp_path / "Malek.png"
    karte.write_bytes(png)
    malek.write_bytes(png)
    liora = tmp_path / "Liora Veylan_1.png"
    liora.write_bytes(png)
    # 15.10.: a note named neither in the question nor the answer brings no images
    with_images = [{**SOURCES[0], "images": [str(karte), str(tmp_path / "fehlt.png")]},
                   {**SOURCES[1], "images": [str(karte), str(malek)]},
                   {**SOURCES[1], "file": r"Organisationen\Chronos\Liora Veylan.md", "images": [str(liora)]},
                   *SOURCES[2:]]
    monkeypatch.setattr(conftest, "_STREAM_EVENTS",
                        [*conftest._STREAM_EVENTS[:-1], {**conftest._STREAM_EVENTS[-1], "sources": with_images}])
    at = open_page("chat").run()
    at.chat_input[0].set_value("Wie sieht Arkenfeld aus?").run()
    assert not at.exception, [e.message for e in at.exception]
    assert "🖼️ Bilder (2)" in [e.label for e in at.expander]


def test_no_image_box_without_images(fake_backend, monkeypatch):
    assert not any(e.label.startswith("🖼️") for e in _ask(monkeypatch).expander)


def test_token_and_timing_caption_under_the_answer(fake_backend, monkeypatch):
    captions = " | ".join(c.value for c in _ask(monkeypatch).caption)

    assert "⬇ 10 in · ⬆ 5 out · 🧠 3 think" in captions
    assert "⏱ 1. Token 1.1 s · gesamt 3.0 s · Umformulieren 0.8 s · Suche 0.5 s · 8 Chunks" in captions


def test_sidebar_shows_where_the_reranker_runs(fake_backend, monkeypatch):
    state = {**conftest.PAYLOADS["/sidebar-state"]}
    state["health"] = {**state["health"], "reranker": "gpu"}
    monkeypatch.setitem(conftest.PAYLOADS, "/sidebar-state", state)
    at = open_page("chat").run()
    assert "Reranker: GPU ✅" in [m.value for m in at.sidebar.markdown]


def test_sidebar_warns_when_the_reranker_fell_back_to_the_cpu(fake_backend, monkeypatch):
    """DirectML configured, reranker on the CPU: answers still right but slower — must not stay unnoticed."""
    state = {**conftest.PAYLOADS["/sidebar-state"]}
    state["health"] = {**state["health"], "reranker": "cpu_fallback"}
    monkeypatch.setitem(conftest.PAYLOADS, "/sidebar-state", state)
    at = open_page("chat").run()
    assert any("Reranker läuft auf der CPU" in w.value for w in at.sidebar.warning)


def test_chat_input_is_bounded_like_the_api(fake_backend):
    """The field stops at the API's MAX_QUESTION_CHARS — a pasted document never reaches the backend (13.0)."""
    from src.api.schemas import MAX_QUESTION_CHARS
    at = open_page("chat").run()
    assert at.chat_input[0].proto.max_chars == MAX_QUESTION_CHARS
