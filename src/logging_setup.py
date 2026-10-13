"""Logging for backend and CLI: console plus one log file per day, noisy third-party loggers set to ERROR,
uvicorn's access log to WARNING.

The day changes at ``logging.rotate_hour`` (06:00), not at midnight, so a game night stays in one file.
The current day is written to ``lorekeeper.log``; at the change it becomes ``lorekeeper.<YYYY-MM-DD>.log``
(the day it started) and only the newest ``backup_count`` days are kept. A backend that isn't running at
06:00 switches files with its first line after that.
"""
import io
import logging
import sys
from datetime import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from src.config.manager import LoggingConfig

# uvicorn logs unhandled route exceptions and lifespan crashes here
_UVICORN_ERROR = "uvicorn.error"


def _reaches_root(logger: logging.Logger) -> bool:
    """True if records of ``logger`` propagate up to the root handlers."""
    current: logging.Logger | None = logger
    while current is not None and current.parent is not None:
        if not current.propagate:
            return False
        current = current.parent
    return True


def _dated_name(default_name: str) -> str:
    """``lorekeeper.log.2026-10-13`` → ``lorekeeper.2026-10-13.log``: sorts by day and keeps the ``.log``
    extension, so the file still opens in an editor by double-click."""
    path = Path(default_name)
    base, _, day = path.name.rpartition(".")
    return str(path.with_name(f"{Path(base).stem}.{day}{Path(base).suffix}"))


def _daily_file_handler(path: Path, config: LoggingConfig) -> TimedRotatingFileHandler:
    """File handler that starts a new file every day at ``config.rotate_hour``."""
    handler = TimedRotatingFileHandler(
        path,
        when="midnight",                          # daily — atTime moves the change away from midnight
        atTime=time(hour=config.rotate_hour),
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    handler.namer = _dated_name
    return handler


def setup_logging(config: LoggingConfig):
    """Wire up console + daily file handlers and silence noisy third-party loggers.

    Under uvicorn, ``uvicorn.error`` stops at the ``uvicorn`` logger (``propagate=False``, console only),
    so tracebacks of 500s and startup crashes would never reach the file — the file handler is attached
    to it directly. Without uvicorn (tests, CLI) the records already reach the root and nothing is added.
    """
    root = logging.getLogger()
    root.setLevel(config.level.upper())

    # Clear existing handlers (avoid duplicates on reload)
    root.handlers.clear()

    formatter = logging.Formatter(config.format)

    # Console handler. Redirected stdout on Windows defaults to cp1252, which can't encode
    # characters like "→" in our log lines — every such line would raise a logging error.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (one file per day)
    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _daily_file_handler(log_path, config)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        uvicorn_error = logging.getLogger(_UVICORN_ERROR)
        for old in [h for h in uvicorn_error.handlers if isinstance(h, logging.FileHandler)]:
            uvicorn_error.removeHandler(old)  # repeated setup (reload) must not write every line twice
            old.close()
        if not _reaches_root(uvicorn_error):
            uvicorn_error.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in config.suppress:
        logging.getLogger(name).setLevel(logging.ERROR)

    # uvicorn access logs are noisy, keep only errors
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger("src").info(
        f"Logging initialized (level={config.level}, file={config.file})"
    )
