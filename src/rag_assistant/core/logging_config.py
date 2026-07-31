"""Single place that configures logging for the app and CLI."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-38s | %(message)s"


def configure_logging(level: str | int = "INFO", *, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level if isinstance(level, int) else level.upper())

    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
