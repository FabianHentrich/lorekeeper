"""Fake backend for rendering the Streamlit pages with AppTest.

The pages talk to the API via ``requests``; the fixture swaps ``requests.get/post/put``
for a stub that answers from canned payloads and records every call, so tests can
count backend round-trips per render without a running server.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

UI_DIR = Path(__file__).resolve().parents[2] / "ui"

# Pages import the shared ui/admin_gate.py like `streamlit run ui/LoreKeeper.py` does
sys.path.insert(0, str(UI_DIR))
import admin_gate  # noqa: E402,F401 — load .env once now, so the fixture below wins over it


@pytest.fixture(autouse=True)
def _no_admin_password(monkeypatch):
    """Pages render ungated unless a test sets LOREKEEPER_ADMIN_PASSWORD (the real .env may set one)."""
    monkeypatch.delenv("LOREKEEPER_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("LOREKEEPER_API_URL", raising=False)


@pytest.fixture(autouse=True)
def _fresh_login_limiter(monkeypatch):
    """Login rate limiting is process-wide state — reset it, and skip the per-failure delay."""
    monkeypatch.setattr(admin_gate, "_failures", [])
    monkeypatch.setattr(admin_gate, "FAILURE_DELAY_S", 0.0)


@pytest.fixture(autouse=True)
def _fresh_streamlit_cache():
    """st.cache_data lives for the whole process — without clearing, a page would reuse
    another test's fake responses (e.g. the cached /sidebar-state)."""
    import streamlit as st
    st.cache_data.clear()

PAGES = {
    "chat": UI_DIR / "LoreKeeper.py",
    "sources": UI_DIR / "pages" / "1_Sources.py",
    "evaluation": UI_DIR / "pages" / "2_Evaluation.py",
    "prompts": UI_DIR / "pages" / "3_Prompts.py",
    "settings": UI_DIR / "pages" / "4_Settings.py",
}

_PROMPTS = {"system": "S", "qa": "Q", "condense": "C", "no_context": "N"}

PAYLOADS: dict[str, object] = {
    "/sidebar-state": {
        "health": {"status": "healthy", "chromadb": True, "llm": True, "sources_configured": True},
        "provider": {"provider": "ollama", "model": "qwen3:14b"},
        "gemini_status": {"has_key": False, "source": "none"},
        "available_categories": ["misc"],
        "player_source_ids": ["regelbuch"],
        "llm_timeout": 300,
    },
    "/sources": {"sources": [{
        "id": "vault", "path": "C:/Vault", "group": "lore", "default_category": "misc",
        "category_map": {"Regeln": {"category": "misc", "group": "rules"}}, "exclude_patterns": [],
    }]},
    "/sources/vault/folders": {"folders": [{"name": "Regeln", "type": "folder"}]},
    "/stats": {"chunk_count": 100},
    "/config": {
        "retrieval": {"top_k": 15, "score_threshold": 0.5,
                      "reranking": {"enabled": True, "top_k_rerank": 8, "max_per_source": 3},
                      "hybrid": {"enabled": False, "bm25_weight": 0.3, "bm25_top_k": 15}},
        "llm": {"provider": "ollama", "fallback_enabled": False,
                "ollama": {"model": "qwen3:14b", "temperature": 0.3, "top_p": 0.9, "max_tokens": 2048, "timeout": 300},
                "gemini": {"model": "gemini-2.5-flash", "temperature": 0.3, "top_p": 0.9, "max_tokens": 8192,
                           "timeout": 30}},
        "conversation": {"window_size": 8, "condense_question": True, "session_timeout_minutes": 60},
        "chunking": {"strategy": "heading_aware", "max_chunk_size": 256, "chunk_overlap": 30, "min_chunk_size": 20},
        "embeddings": {"model": "intfloat/multilingual-e5-base", "device": "auto"},
        "vectorstore": {"mode": "embedded", "collection_name": "lorekeeper"},
    },
    "/prompts/active": {"name": "active", "description": "", "prompts": _PROMPTS},
    "/prompts/variants": [{"name": "Variante", "description": "", "is_active": False}],
    "/prompts/variants/Variante": {"name": "Variante", "description": "", "prompts": _PROMPTS},
    "/eval/results": [],
    "/feedback": [],
    "/eval/chunks/files": {"files": ["Arkenfeld.md", "Regeln.md"]},
    "/eval/chunks": {"source_id": "vault", "source_file": "Arkenfeld.md", "chunks": [
        {"id": "c0", "chunk_index": 0, "heading": "Arkenfeld", "chars": 29,
         "content": "Arkenfeld ist ein Zeitmagier.", "metadata": {"source_id": "vault", "chunk_index": 0}},
        {"id": "c1", "chunk_index": 1, "heading": "Arkenfeld > Familie", "chars": 300,
         "content": "Seine Schwester lebt in der Akademie.", "metadata": {"source_id": "vault", "chunk_index": 1}},
    ]},
    "/eval/qa-pairs": {"pairs": [{
        "id": "p001", "question": "Was ist eine Marke?", "source_type": "pdf", "category": "mechanic",
        "expected_sources": ["regeln.pdf"], "expected_answer_contains": ["Marke"],
        "expected_section": "2.5", "notes": "",
    }]},
}

_STREAM_EVENTS = [
    {"type": "token", "content": "Eine Marke "},
    {"type": "token", "content": "ist eine Ressource."},
    {"type": "done", "session_id": "s1", "sources": [], "model_used": "qwen3:14b", "search_query": "Was ist eine Marke?",
     "timings": {"retrieval_ms": 500, "first_token_ms": 1100, "total_ms": 3000, "chunks": 8},
     "usage": {"tokens_in": 10, "tokens_out": 5, "tokens_thinking": 0},
     "session_usage": {"tokens_in": 10, "tokens_out": 5, "tokens_thinking": 0}},
]


class _Response:
    """Minimal stand-in for ``requests.Response`` (JSON body or an SSE stream)."""

    def __init__(self, data, stream_events=None, status_code=200):
        self._data = data
        self._stream_events = stream_events or []
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        for event in self._stream_events:
            yield f"data: {json.dumps(event)}"


class FakeBackend:
    """Records every backend call as "METHOD /path" and answers from PAYLOADS."""

    def __init__(self):
        self.calls: list[str] = []
        self.unknown: list[str] = []
        self.bodies: dict[str, list] = {}   # "METHOD /path" -> JSON bodies sent, in order
        self.fail_paths: set[str] = set()   # paths that raise ConnectionError (backend down)
        self.errors: dict[str, tuple[int, object]] = {}  # "METHOD /path" -> (status code, JSON body)

    def _handler(self, method: str):
        def handle(url, *args, **kwargs):
            path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
            self.calls.append(f"{method} {path}")
            self.bodies.setdefault(f"{method} {path}", []).append(kwargs.get("json"))
            if path in self.fail_paths:
                raise requests.ConnectionError(f"fake backend down: {path}")
            if f"{method} {path}" in self.errors:
                code, body = self.errors[f"{method} {path}"]
                return _Response(body, status_code=code)
            if path == "/query/stream":
                return _Response({}, stream_events=_STREAM_EVENTS)
            if path not in PAYLOADS:
                self.unknown.append(f"{method} {path}")
            return _Response(PAYLOADS.get(path, {}))
        return handle


def open_page(page: str, role: str | None = "game_master"):
    """AppTest for a page; the chat starts with a role already picked unless role=None."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PAGES[page]), default_timeout=60)
    if page == "chat" and role is not None:
        at.session_state["_role"] = role
    return at


@pytest.fixture
def fake_backend():
    """Patch ``requests`` so the Streamlit pages hit the fake backend instead of the API."""
    backend = FakeBackend()
    with patch.object(requests, "get", backend._handler("GET")), \
            patch.object(requests, "post", backend._handler("POST")), \
            patch.object(requests, "put", backend._handler("PUT")):
        yield backend
