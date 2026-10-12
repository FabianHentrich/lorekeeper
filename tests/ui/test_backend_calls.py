"""Performance guard (roadmap 5.2): backend round-trips per UI render.

Every Streamlit interaction reruns the whole page script, so an uncached request
costs its full latency on *every click*. The budgets below are the current counts;
a test fails when a change adds requests to a page. If the extra call is intended,
raise the budget in the same change.
"""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import open_page

ROOT = Path(__file__).resolve().parents[2]

# page: (max calls on first render, max calls on a rerun without changes)
BUDGETS = {
    "chat": (1, 0),         # /sidebar-state, cached afterwards
    "sources": (3, 0),      # /sources, /stats, 1x /folders per source — all cached afterwards
    "evaluation": (4, 0),   # /eval/qa-pairs, /eval/results, /feedback, /sources (chunk inspector)
    "prompts": (3, 0),      # /prompts/active, /prompts/variants, /prompts/variants/{name}
    "settings": (1, 0),     # /config
}


def _run(page: str) -> AppTest:
    at = open_page(page)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize("page", sorted(BUDGETS))
def test_backend_calls_per_render_stay_within_budget(page, fake_backend):
    """First render and an unchanged rerun must not exceed the page's request budget."""
    first_max, rerun_max = BUDGETS[page]

    at = _run(page)
    first = list(fake_backend.calls)
    fake_backend.calls.clear()
    at.run()

    assert not fake_backend.unknown, f"unexpected endpoints (add to PAYLOADS): {fake_backend.unknown}"
    assert len(first) <= first_max, f"{page} first render: {first}"
    assert len(fake_backend.calls) <= rerun_max, f"{page} rerun: {fake_backend.calls}"


def test_chat_message_costs_one_request(fake_backend):
    """Sending a question: only the streamed answer — the LLM timeout comes from the cached /sidebar-state."""
    at = _run("chat")
    fake_backend.calls.clear()

    at.chat_input[0].set_value("Was ist eine Marke?").run()

    assert not at.exception, [e.message for e in at.exception]
    assert "POST /query/stream" in fake_backend.calls
    assert fake_backend.calls == ["POST /query/stream"], fake_backend.calls
    assert any("Eine Marke ist eine Ressource." in m.value for m in at.markdown)


def test_local_service_urls_avoid_localhost():
    """Default URLs to the backend and Ollama must use 127.0.0.1, not localhost.

    On Windows "localhost" resolves to IPv6 ::1 first; uvicorn and Ollama listen on
    IPv4 only, so every new connection waits ~2 s for the fallback (measured: 2000 ms
    vs. 3 ms per request) — that was added to every chat message and page render.
    """
    files = [*sorted((ROOT / "ui").rglob("*.py")), ROOT / "config" / "settings.yaml",
             ROOT / "src" / "config" / "manager.py", ROOT / "evaluation" / "evaluate.py"]
    offenders = [
        f"{f.relative_to(ROOT)}:{n}"
        for f in files
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"https?://localhost\b", line)
    ]
    assert not offenders, offenders
