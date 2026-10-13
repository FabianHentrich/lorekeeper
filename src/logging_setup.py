"""Logging for backend and CLI: console plus rotating file, noisy third-party loggers set to ERROR,
uvicorn's access log to WARNING.
"""
import io
import logging
import sys
from logging.handlers import RotatingFileHandler
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


def setup_logging(config: LoggingConfig):
    """Wire up console + rotating file handlers and silence noisy third-party loggers.

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

    # File handler (rotating)
    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        uvicorn_error = logging.getLogger(_UVICORN_ERROR)
        for old in [h for h in uvicorn_error.handlers if isinstance(h, RotatingFileHandler)]:
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
