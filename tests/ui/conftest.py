"""Fake backend for rendering the Streamlit pages with AppTest.

The pages talk to the API via ``requests``; the fixture swaps ``requests.get/post/put``
for a stub that answers from canned payloads and records every call, so tests can
count backend round-trips per render without a running server.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

UI_DIR = Path(__file__).resolve().parents[2] / "ui"

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

    def __init__(self, data, stream_events=None):
        self._data = data
        self._stream_events = stream_events or []
        self.status_code = 200
        self.ok = True
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

    def _handler(self, method: str):
        def handle(url, *args, **kwargs):
            path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
            self.calls.append(f"{method} {path}")
            if path == "/query/stream":
                return _Response({}, stream_events=_STREAM_EVENTS)
            if path not in PAYLOADS:
                self.unknown.append(f"{method} {path}")
            return _Response(PAYLOADS.get(path, {}))
        return handle


@pytest.fixture
def fake_backend():
    """Patch ``requests`` so the Streamlit pages hit the fake backend instead of the API."""
    backend = FakeBackend()
    with patch.object(requests, "get", backend._handler("GET")), \
            patch.object(requests, "post", backend._handler("POST")), \
            patch.object(requests, "put", backend._handler("PUT")):
        yield backend
