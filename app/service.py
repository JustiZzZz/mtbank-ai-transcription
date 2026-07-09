"""Общий сервис анализа для FastAPI и OpenWebUI Pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.schemas import (
    AnalysisResult,
    Classification,
    ComplianceResult,
    QualityScore,
    SummaryResult,
    TranscriptSegment,
)

MetadataValue = str | int | float | bool | None
Metadata = Mapping[str, MetadataValue]


class Transcriber(Protocol):
    """ASR-компонент, который превращает аудио в черновые сегменты."""

    async def transcribe(self, audio_path: Path) -> Sequence[TranscriptSegment]:
        """Вернуть сегменты с текстом и таймкодами."""


class Diarizer(Protocol):
    """Компонент, который назначает speaker для сегментов."""

    async def diarize(self, segments: Sequence[TranscriptSegment]) -> Sequence[TranscriptSegment]:
        """Вернуть сегменты с ролями Оператор/Клиент/UNKNOWN."""


class Supervisor(Protocol):
    """Оркестратор агентов анализа."""

    async def analyze(self, transcript: Sequence[TranscriptSegment]) -> AgentAnalysis:
        """Вернуть валидированные результаты агентов."""


@dataclass(frozen=True, slots=True)
class AgentAnalysis:
    """Валидированные части анализа, которые возвращает supervisor."""

    classification: Classification
    quality_score: QualityScore
    compliance: ComplianceResult
    summary: SummaryResult
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalysisService:
    """Единая точка бизнес-логики для API и OpenWebUI Pipeline."""

    transcriber: Transcriber
    diarizer: Diarizer
    supervisor: Supervisor

    async def analyze_audio_path(
        self,
        audio_path: str | Path,
        *,
        metadata: Metadata | None = None,
    ) -> AnalysisResult:
        """Проанализировать локальный аудиофайл и собрать общий JSON-контракт."""
        raw_transcript = await self.transcriber.transcribe(Path(audio_path))
        transcript = await self.diarizer.diarize(raw_transcript)
        agent_analysis = await self.supervisor.analyze(transcript)

        result_metadata = {
            **dict(agent_analysis.metadata),
            **dict(metadata or {}),
        }
        summary = agent_analysis.summary

        return AnalysisResult(
            transcript=list(transcript),
            classification=agent_analysis.classification,
            quality_score=agent_analysis.quality_score,
            compliance=agent_analysis.compliance,
            summary=summary.summary,
            action_items=summary.action_items,
            metadata=result_metadata,
        )
