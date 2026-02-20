"""
logging.py
----------
Structured logging setup for the biometric ML package.

Configures a root logger that emits JSON-structured records when
``json_format=True`` (recommended for production / cloud log aggregators
such as Azure Monitor) and human-readable records for local development.

Usage::

    from biometric_ml.utils.logging import setup_logging
    setup_logging(level="INFO", json_format=False)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """
    Configure the root logger.

    Args:
        level:       Logging level string: DEBUG / INFO / WARNING / ERROR.
        json_format: If True, emit JSON logs (for cloud log aggregators).
    """
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
