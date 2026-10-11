"""Timing check (roadmap 5.2): render time per Streamlit page against the fake backend.

Measures only the page script itself (backend latency is excluded by the fake), so a
slow rerun here means heavy UI code — what you feel on every click and page switch.
"""
import time

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import PAGES, fake_backend  # noqa: F401 — fixture import

pytestmark = pytest.mark.perf


@pytest.mark.parametrize("page", sorted(PAGES))
def test_page_render_time(page, fake_backend, baseline):  # noqa: F811
    """First render and a rerun, each compared to the baseline (×1.5 + 0.2 s slack for noise)."""
    at = AppTest.from_file(str(PAGES[page]), default_timeout=60)
    t0 = time.perf_counter()
    at.run()
    first = time.perf_counter() - t0
    assert not at.exception, [e.message for e in at.exception]

    t0 = time.perf_counter()
    at.run()
    rerun = time.perf_counter() - t0

    baseline.check("ui_render_s", f"{page}_first", first, slack=0.2)
    baseline.check("ui_render_s", f"{page}_rerun", rerun, slack=0.2)
