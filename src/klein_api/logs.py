"""Structured logging: one JSON object per line.

Events carry their data as fields (``event(log, "job.finished", job_id=..., total_seconds=...)``)
so job ids, state transitions, timings and failure categories can be filtered with ``jq``.
Never pass prompts, image data, tokens or file contents as fields: log sizes instead
(``prompt_chars=``).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_FIELDS = "klein_fields"


def event(logger: logging.Logger, name: str, *, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, name, extra={_FIELDS: fields})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, _FIELDS, None)
        if isinstance(fields, dict):
            entry.update(fields)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """JSON lines on stderr for klein-api and uvicorn (started with ``log_config=None``)."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
