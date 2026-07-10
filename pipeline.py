"""OpenWebUI Pipeline для анализа звонков МТБанка."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.runtime import AnalysisRuntime, get_analysis_runtime
from app.schemas import AnalysisResult, TranscriptSegment

URL_RE = re.compile(r"https?://\S+")


class Pipeline:
    """OpenWebUI Pipeline: аудио на входе, markdown-анализ на выходе."""

    class Valves(BaseModel):
        WHISPER_MODEL: str = "medium"
        WHISPER_DEVICE: str = "cuda"
        WHISPER_COMPUTE_TYPE: str = "int8_float16"
        WHISPER_LANGUAGE: str = "ru"

    def __init__(self, runtime: AnalysisRuntime | None = None) -> None:
        self.valves = self.Valves()
        self.runtime = runtime

    async def on_startup(self) -> None:
        if self.runtime is None:
            self.runtime = get_analysis_runtime()

    async def pipe(self, body: dict[str, Any], __user__: dict[str, Any] | None = None) -> str:
        runtime = self.runtime or get_analysis_runtime()
        reference = self._extract_audio_reference(body)
        if not reference:
            return "Не найден аудиофайл или URL. Загрузите WAV, MP3 или OGG."
        if reference.startswith("http://") or reference.startswith("https://"):
            result = await runtime.analyze_url(reference)
        else:
            result = await runtime.analyze_path(Path(reference))
        return self._format_response(result)

    def _extract_audio_reference(self, body: dict[str, Any]) -> str | None:
        for file_info in body.get("files") or []:
            if isinstance(file_info, dict):
                for key in ("path", "url", "file", "filename"):
                    value = file_info.get(key)
                    if isinstance(value, str) and value:
                        return value
        for key in ("path", "url", "audio_url"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
        messages = body.get("messages") or []
        if messages:
            content = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
            match = URL_RE.search(content)
            if match:
                return match.group(0)
        return None

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
