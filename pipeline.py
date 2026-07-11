"""OpenWebUI Pipeline для анализа звонков МТБанка."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.runtime import AnalysisRuntime, get_analysis_runtime
from app.schemas import AnalysisResult, TranscriptSegment
from app.service import AgentAnalysis

URL_RE = re.compile(r"https?://\S+")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".m4a", ".flac", ".mp4")
OPENWEBUI_UPLOAD_DIR = Path("/app/backend/data/uploads")
MAX_TRANSCRIPT_SEGMENT_CHARS = 220

logger = logging.getLogger(__name__)


class Pipeline:
    """OpenWebUI Pipeline: аудио на входе, markdown-анализ на выходе."""

    class Valves(BaseModel):
        WHISPER_MODEL: str = "medium"
        WHISPER_DEVICE: str = "cuda"
        WHISPER_COMPUTE_TYPE: str = "int8_float16"
        WHISPER_LANGUAGE: str = "ru"
        PREFER_OPENWEBUI_TRANSCRIPT: bool = True
        ALLOW_LATEST_UPLOAD_FALLBACK: bool = True
        LATEST_UPLOAD_MAX_AGE_SECONDS: int = 86400

    def __init__(self, runtime: AnalysisRuntime | None = None) -> None:
        self.valves = self.Valves()
        self.runtime = runtime
        self._cache: dict[str, str] = {}
        self._analysis_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="mtbank-pipeline-event-loop",
            daemon=True,
        )
        self._loop_thread.start()

    async def on_startup(self) -> None:
        if self.runtime is None:
            self.runtime = get_analysis_runtime()

    def pipe(
        self,
        user_message: str | dict[str, Any] | None = None,
        model_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        body: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
    ) -> str:
        request_body = self._coerce_body(user_message=user_message, messages=messages, body=body)
        try:
            with self._analysis_lock:
                return self._run_async(self._pipe_async(request_body))
        except Exception as exc:
            logger.exception("OpenWebUI Pipeline failed")
            return f"Ошибка анализа аудио: {exc}"

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coroutine) -> str:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    async def _pipe_async(self, body: dict[str, Any]) -> str:
        runtime = self.runtime or get_analysis_runtime()
        selected = self._select_input(body)
        if selected is None:
            return "Не найден аудиофайл или URL. Загрузите WAV, MP3 или OGG."

        cache_key, input_type, value = selected
        if cached := self._cache.get(cache_key):
            return cached

        if input_type == "transcript":
            result = await self._analyze_transcript_text(runtime, value, cache_key=cache_key)
        else:
            reference = value
            if reference.startswith("http://") or reference.startswith("https://"):
                result = await runtime.analyze_url(reference)
            else:
                result = await runtime.analyze_path(Path(reference))

        response = self._format_response(result)
        self._cache[cache_key] = response
        return response

    def _select_input(self, body: dict[str, Any]) -> tuple[str, str, str] | None:
        reference = self._extract_audio_reference(body)
        if self.valves.PREFER_OPENWEBUI_TRANSCRIPT:
            transcript = self._extract_transcript_text(body)
            if transcript:
                return self._transcript_selection(transcript, source="body")
            if reference:
                transcript = self._transcript_for_audio_reference(reference)
                if transcript:
                    return self._transcript_selection(transcript, source=reference)

        if not reference:
            return None
        if reference.startswith("http://") or reference.startswith("https://"):
            return (f"url:{reference}", "audio", reference)
        path = Path(reference)
        key = f"audio:{path}:{path.stat().st_mtime_ns if path.exists() else 0}"
        return (key, "audio", reference)

    def _transcript_selection(self, transcript: str, *, source: str) -> tuple[str, str, str]:
        key = f"transcript:{source}:{hash(transcript)}"
        return (key, "transcript", transcript)

    def _coerce_body(
        self,
        *,
        user_message: str | dict[str, Any] | None,
        messages: list[dict[str, Any]] | None,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if isinstance(user_message, dict) and body is None:
            return user_message
        request_body = dict(body or {})
        if messages is not None:
            request_body["messages"] = messages
        elif isinstance(user_message, str) and user_message:
            request_body["messages"] = [{"role": "user", "content": user_message}]
        return request_body

    def _extract_transcript_text(self, body: dict[str, Any]) -> str | None:
        for item in self._walk_dicts(body):
            data = item.get("data")
            if isinstance(data, dict):
                transcript = (
                    data.get("transcript")
                    or data.get("transcription")
                    or data.get("text")
                    or data.get("content")
                )
                if isinstance(transcript, str) and self._looks_like_raw_transcript(transcript):
                    return transcript.strip()
        return None

    def _looks_like_raw_transcript(self, value: str) -> bool:
        text = value.strip()
        if len(text) < 80:
            return False
        lower = text.lower()
        analysis_markers = (
            "## анализ звонка",
            "### резюме",
            "### action items",
            "### compliance issues",
            "| спикер |",
        )
        if any(marker in lower for marker in analysis_markers):
            return False
        return any(marker in lower for marker in ("мтбанк", "мт банк", "клиент", "оператор"))

    def _extract_audio_reference(self, body: dict[str, Any]) -> str | None:
        for item in self._walk_dicts(body):
            resolved = self._reference_from_mapping(item)
            if resolved:
                return resolved
        for key in ("path", "url", "audio_url"):
            value = body.get(key)
            resolved = self._reference_from_value(value)
            if resolved:
                return resolved
        for item in self._walk_dicts(body):
            content = item.get("content")
            if isinstance(content, str):
                match = URL_RE.search(content)
                if match:
                    return match.group(0)
        return self._latest_openwebui_upload()

    def _walk_dicts(self, value: Any):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from self._walk_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._walk_dicts(child)

    def _reference_from_mapping(self, item: dict[str, Any]) -> str | None:
        for key in ("path", "url", "file", "filename", "name"):
            resolved = self._reference_from_value(item.get(key))
            if resolved:
                return resolved
        file_id = item.get("id")
        filename = item.get("filename") or item.get("name")
        if isinstance(file_id, str) and isinstance(filename, str):
            return self._resolve_openwebui_upload(file_id=file_id, filename=filename)
        return None

    def _reference_from_value(self, value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        if value.startswith("http://") or value.startswith("https://"):
            return value
        if not value.lower().endswith(AUDIO_EXTENSIONS):
            return None
        path = Path(value)
        if path.exists():
            return str(path)
        return self._resolve_openwebui_upload(filename=path.name)

    def _resolve_openwebui_upload(
        self,
        *,
        filename: str,
        file_id: str | None = None,
    ) -> str | None:
        if file_id:
            candidate = OPENWEBUI_UPLOAD_DIR / f"{file_id}_{filename}"
            if candidate.exists():
                return str(candidate)
            matches = sorted(OPENWEBUI_UPLOAD_DIR.glob(f"{file_id}_*"))
            for match in matches:
                if match.suffix.lower() in AUDIO_EXTENSIONS:
                    return str(match)
        matches = sorted(OPENWEBUI_UPLOAD_DIR.glob(f"*_{filename}"))
        for match in matches:
            if match.suffix.lower() in AUDIO_EXTENSIONS:
                return str(match)
        return None

    def _latest_openwebui_upload(self) -> str | None:
        if not OPENWEBUI_UPLOAD_DIR.exists():
            return None
        candidates = [
            path
            for path in OPENWEBUI_UPLOAD_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        age = max(0, int(time.time() - latest.stat().st_mtime))
        if (
            not self.valves.ALLOW_LATEST_UPLOAD_FALLBACK
            or age > self.valves.LATEST_UPLOAD_MAX_AGE_SECONDS
        ):
            return None
        return str(latest)

    def _transcript_for_audio_reference(self, reference: str) -> str | None:
        if reference.startswith("http://") or reference.startswith("https://"):
            return None
        audio_path = Path(reference)
        candidates = [audio_path.with_suffix(".json")]
        if audio_path.name.startswith(tuple("0123456789abcdef")):
            candidates.extend(audio_path.parent.glob(f"{audio_path.stem}*.json"))
        for candidate in candidates:
            transcript = self._transcript_from_json(candidate)
            if transcript:
                return transcript
        return None

    def _transcript_from_json(self, path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        transcript = payload.get("text") or payload.get("content")
        if isinstance(transcript, str) and transcript.strip():
            return transcript.strip()
        return None

    async def _analyze_transcript_text(
        self,
        runtime: AnalysisRuntime,
        text: str,
        *,
        cache_key: str,
    ) -> AnalysisResult:
        raw_segments = self._segments_from_transcript_text(text)
        transcript = await runtime.service.diarizer.diarize(raw_segments)
        agent_analysis = await runtime.service.supervisor.analyze(transcript)
        return self._result_from_agent_analysis(
            transcript=transcript,
            agent_analysis=agent_analysis,
            metadata={
                "openwebui_input_mode": "preprocessed_transcript",
                "pipeline_cache_key": cache_key,
            },
        )

    def _segments_from_transcript_text(self, text: str) -> list[TranscriptSegment]:
        pieces = [
            piece.strip()
            for piece in re.split(r"(?<=[.!?])\s+", text)
            if piece.strip()
        ]
        if not pieces:
            pieces = [text.strip()]

        segments: list[TranscriptSegment] = []
        buffer = ""
        for piece in pieces:
            candidate = f"{buffer} {piece}".strip() if buffer else piece
            if len(candidate) <= MAX_TRANSCRIPT_SEGMENT_CHARS:
                buffer = candidate
                continue
            if buffer:
                segments.append(self._segment(buffer, len(segments)))
            buffer = piece
        if buffer:
            segments.append(self._segment(buffer, len(segments)))
        return segments

    def _segment(self, text: str, index: int) -> TranscriptSegment:
        return TranscriptSegment(
            speaker="UNKNOWN",
            start=float(index),
            end=float(index + 1),
            text=text,
        )

    def _result_from_agent_analysis(
        self,
        *,
        transcript: list[TranscriptSegment],
        agent_analysis: AgentAnalysis,
        metadata: dict[str, str],
    ) -> AnalysisResult:
        result_metadata = {
            **dict(agent_analysis.metadata),
            **metadata,
        }
        summary = agent_analysis.summary
        return AnalysisResult(
            transcript=transcript,
            classification=agent_analysis.classification,
            quality_score=agent_analysis.quality_score,
            compliance=agent_analysis.compliance,
            summary=summary.summary,
            action_items=summary.action_items,
            metadata=result_metadata,
        )

    def _format_response(self, result: AnalysisResult) -> str:
        transcript = "\n".join(self._format_segment(segment) for segment in result.transcript)
        actions = "\n".join(f"- {item}" for item in result.action_items) or "- Нет"
        issues = (
            "\n".join(f"- {issue.severity}: {issue.phrase}" for issue in result.compliance.issues)
            or "- Нет"
        )
        return f"""## Анализ звонка

**Тема:** {result.classification.topic}
**Приоритет:** {result.classification.priority}
**Уверенность:** {result.classification.confidence_label} ({result.classification.confidence:.2f})

**Оценка качества:** {result.quality_score.total}/100
**Compliance:** {"пройден" if result.compliance.passed else "есть замечания"}

### Резюме
{result.summary}

### Action items
{actions}

### Compliance issues
{issues}

### Транскрипт
| Спикер | Старт | Конец | Текст |
|---|---:|---:|---|
{transcript}
"""

    def _format_segment(self, segment: TranscriptSegment) -> str:
        text = segment.text.replace("|", "\\|")
        return f"| {segment.speaker} | {segment.start:.1f} | {segment.end:.1f} | {text} |"
