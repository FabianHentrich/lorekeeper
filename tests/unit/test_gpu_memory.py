"""Shared-GPU-memory check of the local LLM (roadmap 7b) — PowerShell is mocked; every failure reads as unknown."""
import subprocess
from types import SimpleNamespace

import pytest

from src import gpu_memory


def _run(stdout="", returncode=0, stderr=""):
    return lambda *a, **kw: SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


def test_reads_the_byte_count(monkeypatch):
    monkeypatch.setattr(gpu_memory.subprocess, "run", _run("2845415424\r\n"))
    assert gpu_memory.llm_shared_bytes() == 2845415424


def test_asks_the_shared_usage_counter_without_a_console_window(monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return SimpleNamespace(stdout="0", returncode=0, stderr="")
    monkeypatch.setattr(gpu_memory.subprocess, "run", run)

    assert gpu_memory.llm_shared_bytes() == 0
    assert "\\GPU Process Memory(*)\\Shared Usage" in seen["cmd"][-1]
    assert "llama-server" in seen["cmd"][-1]
    assert seen["creationflags"] == gpu_memory._CREATE_NO_WINDOW and seen["timeout"] == gpu_memory._TIMEOUT_S


@pytest.mark.parametrize("result", [_run(""), _run("", returncode=1, stderr="Get-Counter: …"), _run("n/a")])
def test_no_ollama_process_failed_counter_or_garbage_is_unknown(monkeypatch, result):
    monkeypatch.setattr(gpu_memory.subprocess, "run", result)
    assert gpu_memory.llm_shared_bytes() is None


@pytest.mark.parametrize("error", [OSError("powershell missing"), subprocess.TimeoutExpired("powershell", 20)])
def test_powershell_missing_or_hanging_is_unknown(monkeypatch, error):
    def run(*a, **kw):
        raise error
    monkeypatch.setattr(gpu_memory.subprocess, "run", run)
    assert gpu_memory.llm_shared_bytes() is None
