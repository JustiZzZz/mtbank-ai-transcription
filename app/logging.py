"""Минимальная настройка логов для локального запуска и Docker."""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from logging import LogRecord
from typing import Any

from app.config import Settings

RESERVED_LOG_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {
    "asctime",
    "message",
}


def json_default(value: Any) -> str:
    """Fallback serializer for structured log extras."""
    return str(value)


class JsonFormatter(logging.Formatter):
    """Одна строка лога = один JSON-объект."""

    def format(self, record: LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        for key, value in record.__dict__.items():
            if key not in RESERVED_LOG_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=json_default)


class TextFormatter(logging.Formatter):
    """Простой текстовый формат, когда JSON мешает читать консоль."""

    def format(self, record: LogRecord) -> str:
        timestamp = self.formatTime(record)
        return f"{timestamp} {record.levelname:<8} [{record.name}] {record.getMessage()}"


def configure_logging(settings: Settings) -> None:
    """Настраивает корневой логгер по значениям из .env."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(JsonFormatter() if settings.log_format == "json" else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(handler)
