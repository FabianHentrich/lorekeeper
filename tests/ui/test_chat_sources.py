"""Chat: how an answer's sources, tokens and timings are shown (roadmap 9.0 — chat page had 78 %)."""
from tests.ui import conftest
from tests.ui.conftest import open_page

SOURCES = [
    {"file": "Arkenfeld.md", "source_path": r"C:\Vault\Orte\Arkenfeld Stadt.md", "document_type": "markdown",
     "heading": "Arkenfeld > Lage", "chunk_preview": "Liegt im Norden.", "score": 0.9},
    {"file": "Arkenfeld.md", "source_path": r"C:\Vault\Orte\Arkenfeld Stadt.md", "document_type": "markdown",
     "heading": "Arkenfeld > Rat", "chunk_preview": "Sieben Magier.", "score": 0.7},
    {"file": "regeln.pdf", "source_path": "", "document_type": "pdf", "heading": None,
     "chunk_preview": "", "score": 0.5},
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
    assert any("↳ *Arkenfeld > Rat* (Score: 0.70)" in t for t in texts)
    assert any("↳ *—* (Score: 0.50)" in t for t in texts)                # chunk without heading
    assert "Bild nicht gefunden: karte.png" in [w.value for w in at.warning]


def test_token_and_timing_caption_under_the_answer(fake_backend, monkeypatch):
    captions = " | ".join(c.value for c in _ask(monkeypatch).caption)

    assert "⬇ 10 in · ⬆ 5 out · 🧠 3 think" in captions
    assert "⏱ 1. Token 1.1 s · gesamt 3.0 s · Umformulieren 0.8 s · Suche 0.5 s · 8 Chunks" in captions
