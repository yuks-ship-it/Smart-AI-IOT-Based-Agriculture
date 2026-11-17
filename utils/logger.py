"""
Centralized logging setup for the AI layer.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = LOG_DIR / "app.log"

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_configured_loggers = set()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with console + rotating file handlers."""
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger
