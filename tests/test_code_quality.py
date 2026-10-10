"""Lint and type checks as part of the test suite — config lives in pyproject.toml."""
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args], cwd=_REPO, capture_output=True, text=True, encoding="utf-8",
    )


def test_ruff():
    """Verify `ruff check .` reports no findings."""
    result = _run("ruff", "check", ".", "--output-format", "concise")
    assert result.returncode == 0, result.stdout + result.stderr


def test_mypy():
    """Verify `mypy` (files configured in pyproject.toml) reports no errors."""
    result = _run("mypy")
    assert result.returncode == 0, result.stdout + result.stderr
