"""Supervisor для запуска fallback-агентов."""

import asyncio
from collections.abc import Sequence

from app.agents.classifier import FallbackClassifierAgent
from app.agents.compliance import FallbackComplianceAgent
from app.agents.llm import (
    LLMClassifierAgent,
    LLMComplianceAgent,
    LLMQualityAgent,
    LLMSummarizerAgent,
)
from app.agents.quality import FallbackQualityAgent
from app.agents.summarizer import FallbackSummarizerAgent
from app.config import Settings
from app.llm.client import OpenAICompatibleClient
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


class LLMSupervisor:
    """Запускает четыре LLM-агента, каждый со своим deterministic fallback."""

    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client
        self.classifier = LLMClassifierAgent(client)
        self.quality = LLMQualityAgent(client)
        self.compliance = LLMComplianceAgent(client)
        self.summarizer = LLMSummarizerAgent(client)

    async def analyze(self, transcript: Sequence[TranscriptSegment]) -> AgentAnalysis:
        (
            (classification, classifier_mode),
            (quality_score, quality_mode),
            (compliance, compliance_mode),
            (summary, summarizer_mode),
        ) = await asyncio.gather(
            self.classifier.analyze_with_mode(transcript),
            self.quality.analyze_with_mode(transcript),
            self.compliance.analyze_with_mode(transcript),
            self.summarizer.analyze_with_mode(transcript),
        )
        return AgentAnalysis(
            classification=classification,
            quality_score=quality_score,
            compliance=compliance,
            summary=summary,
            metadata={
                "agents_mode": "llm",
                "llm_model": self.client.settings.openai_model,
                "llm_classifier_mode": classifier_mode,
                "llm_quality_mode": quality_mode,
                "llm_compliance_mode": compliance_mode,
                "llm_summarizer_mode": summarizer_mode,
            },
        )


def build_supervisor(settings: Settings) -> FallbackSupervisor | LLMSupervisor:
    """Выбрать LLM supervisor только при полной конфигурации провайдера."""
    if settings.llm_enabled and settings.openai_api_key and settings.openai_model:
        return LLMSupervisor(OpenAICompatibleClient(settings))
    return FallbackSupervisor()
