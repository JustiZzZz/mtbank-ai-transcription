"""Supervisor для запуска fallback-агентов."""

import asyncio
from collections.abc import Sequence

from app.agents.classifier import FallbackClassifierAgent
from app.agents.compliance import FallbackComplianceAgent
from app.agents.quality import FallbackQualityAgent
from app.agents.summarizer import FallbackSummarizerAgent
from app.schemas import TranscriptSegment
from app.service import AgentAnalysis


class FallbackSupervisor:
    """Запускает четыре deterministic агента поверх готового транскрипта."""

    def __init__(self) -> None:
        self.classifier = FallbackClassifierAgent()
        self.quality = FallbackQualityAgent()
        self.compliance = FallbackComplianceAgent()
        self.summarizer = FallbackSummarizerAgent()

    async def analyze(self, transcript: Sequence[TranscriptSegment]) -> AgentAnalysis:
        classification, quality_score, compliance, summary = await asyncio.gather(
            self.classifier.analyze(transcript),
            self.quality.analyze(transcript),
            self.compliance.analyze(transcript),
            self.summarizer.analyze(transcript),
        )
        return AgentAnalysis(
            classification=classification,
            quality_score=quality_score,
            compliance=compliance,
            summary=summary,
            metadata={"agents_mode": "fallback"},
        )
