"""Structured logging helpers for agent input/output telemetry."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from app.schemas import TranscriptSegment

AgentResult = TypeVar("AgentResult")

MAX_LOG_TEXT_CHARS = 500


def truncate_text(value: str, *, limit: int = MAX_LOG_TEXT_CHARS) -> str:
    """Keep logs readable while preserving enough text for debugging."""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def transcript_log_payload(transcript: Sequence[TranscriptSegment]) -> dict[str, Any]:
    """Serialize transcript input for agent logs."""
    segments = [
        {
            "speaker": segment.speaker,
            "start": segment.start,
            "end": segment.end,
            "text": truncate_text(segment.text),
        }
        for segment in transcript
    ]
    return {
        "segment_count": len(segments),
        "total_text_chars": sum(len(segment.text) for segment in transcript),
        "segments": segments,
    }


def result_log_payload(result: Any) -> Any:
    """Serialize agent output for JSON logs."""
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, tuple) and len(result) == 2:
        payload, mode = result
        return {
            "mode": mode,
            "result": result_log_payload(payload),
        }
    if isinstance(result, list):
        return [result_log_payload(item) for item in result]
    if isinstance(result, dict):
        return {key: result_log_payload(value) for key, value in result.items()}
    return result


async def run_logged_agent(
    *,
    logger: logging.Logger,
    agent_name: str,
    transcript: Sequence[TranscriptSegment],
    call: Callable[[], Awaitable[AgentResult]],
) -> AgentResult:
    """Run one agent and emit structured input/output/error JSON log records."""
    started_at = time.perf_counter()
    input_payload = transcript_log_payload(transcript)
    logger.info(
        "Agent input",
        extra={
            "event": "agent_input",
            "agent": agent_name,
            "agent_input": input_payload,
        },
    )
    try:
        result = await call()
    except Exception:
        logger.exception(
            "Agent failed",
            extra={
                "event": "agent_error",
                "agent": agent_name,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "agent_input": input_payload,
            },
        )
        raise

    logger.info(
        "Agent output",
        extra={
            "event": "agent_output",
            "agent": agent_name,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "agent_output": result_log_payload(result),
        },
    )
    return result
