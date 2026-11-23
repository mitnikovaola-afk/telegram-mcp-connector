"""Logging: stderr (stdout belongs to the MCP stdio protocol) + a rotating file.

The formatter masks every known secret value in the whole formatted record (message, traceback,
stack), so a token that leaks into an exception or a library debug line is never written anywhere.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from .config import Settings


class RedactingFormatter(logging.Formatter):
    """Masks secrets AFTER formatting: the traceback of `log.exception` is part of the output too."""

    def __init__(self, fmt: str, datefmt: str, secrets: list[str]):
        super().__init__(fmt, datefmt)
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


def setup_logging(settings: Settings, level: int = logging.INFO) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S",
                             settings.secret_values())

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)

    file_handler = RotatingFileHandler(settings.log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    os.chmod(settings.log_path, 0o600)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stderr_handler)
    root.addHandler(file_handler)
    root.setLevel(level)
    # Third-party libraries are chatty and may echo request details.
    for noisy in ("telethon", "httpx", "httpx2", "httpcore", "google_genai", "notion_client", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
