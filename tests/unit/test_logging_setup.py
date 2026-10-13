import io
import logging
import sys

import pytest

from src.config.manager import LoggingConfig
from src.logging_setup import setup_logging


@pytest.fixture
def restore_root_logger():
    """setup_logging replaces the root handlers (and touches uvicorn's loggers) — restore them afterwards."""
    root = logging.getLogger()
    uv, uv_err = logging.getLogger("uvicorn"), logging.getLogger("uvicorn.error")
    handlers, level = list(root.handlers), root.level
    uv_err_handlers, uv_propagate = list(uv_err.handlers), uv.propagate
    yield
    for h in root.handlers:
        if h not in handlers:
            h.close()
    root.handlers[:] = handlers
    root.setLevel(level)
    uv_err.handlers[:] = uv_err_handlers
    uv.propagate = uv_propagate


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


def _file_lines(path):
    for h in logging.getLogger().handlers:
        h.flush()
    return path.read_text(encoding="utf-8").splitlines()


def test_uvicorn_errors_reach_the_log_file(tmp_path, restore_root_logger):
    """Under uvicorn, "uvicorn" doesn't propagate — 500 tracebacks and startup crashes would miss the file."""
    logging.getLogger("uvicorn").propagate = False   # as uvicorn's own logging config sets it
    log = tmp_path / "lk.log"

    setup_logging(LoggingConfig(file=str(log)))
    setup_logging(LoggingConfig(file=str(log)))       # a second setup (reload) must not duplicate lines
    logging.getLogger("uvicorn.error").error("Exception in ASGI application")

    assert sum("Exception in ASGI application" in line for line in _file_lines(log)) == 1


def test_no_extra_handler_when_uvicorn_errors_reach_the_root(tmp_path, restore_root_logger):
    """Without uvicorn (tests, CLI) the records already reach the root file handler — no duplicate."""
    logging.getLogger("uvicorn").propagate = True
    log = tmp_path / "lk.log"

    setup_logging(LoggingConfig(file=str(log)))
    logging.getLogger("uvicorn.error").error("only once")

    assert sum("only once" in line for line in _file_lines(log)) == 1


def test_suppress_list_in_code_matches_settings_yaml():
    """The YAML list replaces the code default — two lists that drift apart left loggers unsuppressed."""
    from pathlib import Path

    import yaml

    shipped = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "settings.yaml").read_text(
        encoding="utf-8"))["logging"]["suppress"]
    assert shipped == LoggingConfig().suppress


def test_one_file_per_day_named_by_date_and_old_days_deleted(tmp_path, restore_root_logger):
    """At the day change the current file becomes lorekeeper.<day>.log; only backup_count days stay."""
    for day in ("2026-09-01", "2026-09-02", "2026-09-03"):
        (tmp_path / f"lorekeeper.{day}.log").write_text("old", encoding="utf-8")
    (tmp_path / "feedback.jsonl").write_text("{}", encoding="utf-8")   # other files in logs/ stay untouched
    log = tmp_path / "lorekeeper.log"

    setup_logging(LoggingConfig(file=str(log), backup_count=2))
    handler = next(h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler))
    logging.getLogger("src.test").info("Spielabend")
    handler.doRollover()

    names = sorted(p.name for p in tmp_path.iterdir())
    dated = [n for n in names if n.startswith("lorekeeper.") and n != "lorekeeper.log"]
    assert len(dated) == 2 and "lorekeeper.2026-09-01.log" not in dated
    assert "feedback.jsonl" in names and "lorekeeper.log" in names
    newest = tmp_path / dated[-1]
    assert "Spielabend" in newest.read_text(encoding="utf-8")


def test_the_day_changes_at_rotate_hour_not_midnight(tmp_path, restore_root_logger):
    """A game night past midnight must stay in one file."""
    setup_logging(LoggingConfig(file=str(tmp_path / "lorekeeper.log"), rotate_hour=6))
    handler = next(h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler))

    from datetime import datetime
    assert datetime.fromtimestamp(handler.rolloverAt).hour == 6
