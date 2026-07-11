"""Unit tests for structured JSON logging."""

import json
import logging

from app.logging import JsonFormatter


def test_json_formatter_includes_structured_extra_fields() -> None:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Agent output",
        args=(),
        exc_info=None,
    )
    record.event = "agent_output"
    record.agent = "classifier"
    record.duration_ms = 12.34
    record.agent_output = {"topic": "кредиты"}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "Agent output"
    assert payload["event"] == "agent_output"
    assert payload["agent"] == "classifier"
    assert payload["duration_ms"] == 12.34
    assert payload["agent_output"] == {"topic": "кредиты"}
