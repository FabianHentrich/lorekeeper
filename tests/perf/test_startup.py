"""Backend startup time (roadmap 5.2): from process start until /health answers.

The FastAPI lifespan loads everything before uvicorn serves — Ollama autostart, embedding
warm-up, ChromaDB, reranker — so the first /health response marks a finished startup.

Starts a second backend on a free port against the real config and ``chroma_data/``. The
embedded ChromaDB can't be opened twice, so the check is skipped while LoreKeeper runs.
Measure on an otherwise idle machine — a busy CPU (e.g. a simulation run) inflates the
model loading time and would end up in the baseline.
"""
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.perf

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_S = 180


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop(proc: subprocess.Popen) -> None:
    """Graceful shutdown (lifespan cleanup runs), hard kill as fallback."""
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        proc.wait(timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        proc.kill()
        proc.wait(timeout=10)


def test_startup_until_health(baseline, tmp_path):
    """Startup compared to the baseline (×1.5 + 3 s slack — model loading varies with disk cache)."""
    if _port_open(8000):
        pytest.skip("LoreKeeper backend is running on :8000 — chroma_data/ is locked, stop it first")

    port = _free_port()
    log = (tmp_path / "backend.log").open("w", encoding="utf-8")
    start = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    try:
        elapsed = None
        while time.perf_counter() - start < TIMEOUT_S:
            if proc.poll() is not None:
                break
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                    elapsed = time.perf_counter() - start
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
    finally:
        _stop(proc)
        log.close()

    output = (tmp_path / "backend.log").read_text(encoding="utf-8", errors="replace")[-2000:]
    assert elapsed is not None, f"backend did not answer /health within {TIMEOUT_S}s:\n{output}"
    baseline.check("startup_s", "until_health", elapsed, slack=3)
