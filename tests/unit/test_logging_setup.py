import io
import logging
import sys

import pytest

from src.config.manager import LoggingConfig
from src.logging_setup import setup_logging


@pytest.fixture
def restore_root_logger():
    """setup_logging replaces the root handlers — put the originals back afterwards."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for h in root.handlers:
        if h not in handlers:
            h.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_console_handles_non_cp1252_chars(monkeypatch, restore_root_logger):
    """Verify a cp1252 stdout (redirected Windows console) logs '→' without a logging error."""
    raw = io.BytesIO()
    fake_stdout = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    errors = []
    monkeypatch.setattr(logging.Handler, "handleError", lambda self, record: errors.append(record))

    setup_logging(LoggingConfig(file=None))
    logging.getLogger("src.test").info("DELETE /sources/x → 200")
    fake_stdout.flush()

    assert errors == []
    assert "DELETE /sources/x → 200" in raw.getvalue().decode("utf-8")
