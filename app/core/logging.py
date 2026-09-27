"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import PROJECT_ROOT, settings

_CONFIGURED = False


def setup_logging(
    level: str | int | None = None,
    log_file: str = "msn-ai.log",
    max_bytes: int = 2 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure the root logger once and return it.

    Writes to both stdout and a rotating file under the project root.
    """
    global _CONFIGURED

    root = logging.getLogger()
    if _CONFIGURED:
        return root

    level = (level or settings.log_level or "INFO").upper()
    numeric_level = getattr(logging, level, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.setLevel(numeric_level)
    _CONFIGURED = True

    # Silence noisy third-party loggers at DEBUG.
    for name in ("uvicorn.access", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return root


def get_logger(name: str = "msn_ai") -> logging.Logger:
    """Return a child logger, ensuring base configuration exists."""
    setup_logging()
    return logging.getLogger(name)