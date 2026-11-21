"""Logging: stderr (stdout belongs to the MCP stdio protocol) + a rotating file.

A filter masks every known secret value, so a token that leaks into an exception
message or a library debug line is still never written anywhere.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from .config import Settings


class RedactSecretsFilter(logging.Filter):
    def __init__(self, secrets: list[str]):
        super().__init__()
        self._secrets = sorted(set(secrets), key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self._secrets:
            if secret in redacted:
                redacted = redacted.replace(secret, "***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(settings: Settings, level: int = logging.INFO) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    redact = RedactSecretsFilter(settings.secret_values())

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    stderr_handler.addFilter(redact)

    file_handler = RotatingFileHandler(settings.log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(redact)
    os.chmod(settings.log_path, 0o600)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stderr_handler)
    root.addHandler(file_handler)
    root.setLevel(level)
    # Third-party libraries are chatty and may echo request details.
    for noisy in ("telethon", "httpx", "httpx2", "httpcore", "google_genai", "notion_client", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
