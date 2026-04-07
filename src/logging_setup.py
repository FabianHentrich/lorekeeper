import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.manager import LoggingConfig


def setup_logging(config: LoggingConfig):
    root = logging.getLogger()
    root.setLevel(config.level.upper())

    # Clear existing handlers (avoid duplicates on reload)
    root.handlers.clear()

    formatter = logging.Formatter(config.format)

    # Console handler
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

    # Suppress noisy third-party loggers
    for name in config.suppress:
        logging.getLogger(name).setLevel(logging.WARNING)

    # uvicorn access logs are noisy, keep only errors
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger("src").info(
        f"Logging initialized (level={config.level}, file={config.file})"
    )
