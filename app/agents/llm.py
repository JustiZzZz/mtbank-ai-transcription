"""LLM-агенты с deterministic fallback на текущую реализацию."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.classifier import FallbackClassifierAgent
from app.agents.compliance import FallbackComplianceAgent
from app.agents.prompts import (
    BASE_SYSTEM_PROMPT,
    CLASSIFIER_PROMPT,
    COMPLIANCE_PROMPT,
    QUALITY_PROMPT,
    SUMMARIZER_PROMPT,
)
from app.agents.quality import FallbackQualityAgent
from app.agents.summarizer import FallbackSummarizerAgent
from app.llm.client import LLMClientError, OpenAICompatibleClient
from app.schemas import (
    Classification,
    ComplianceResult,
    QualityScore,
    SummaryResult,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

ResultModel = TypeVar("ResultModel", bound=BaseModel)
FallbackCallable = Callable[[Sequence[TranscriptSegment]], Awaitable[ResultModel]]


class LLMValidatedAgent:
    """Общий слой: LLM JSON -> Pydantic -> fallback при любой ошибке."""

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        agent_name: str,
        task_prompt: str,
        response_model: type[ResultModel],
        fallback: FallbackCallable[ResultModel],
    ) -> None:
        self.client = client
        self.agent_name = agent_name
        self.task_prompt = task_prompt
        self.response_model = response_model
        self.fallback = fallback

    async def analyze(self, transcript: Sequence[TranscriptSegment]) -> ResultModel:
        try:
            payload = await self.client.complete_json(
                system_prompt=BASE_SYSTEM_PROMPT,
                user_prompt=self._build_user_prompt(transcript),
            )
            return self.response_model.model_validate(payload)
        except (LLMClientError, ValidationError, ValueError, TypeError) as exc:
            logger.warning(
                "LLM agent %s failed with %s: %s; using fallback",
                self.agent_name,
                exc.__class__.__name__,
                exc,
            )
            return await self.fallback(transcript)

    def _build_user_prompt(self, transcript: Sequence[TranscriptSegment]) -> str:
        schema = json.dumps(self.response_model.model_json_schema(), ensure_ascii=False)
        transcript_json = json.dumps(
            [segment.model_dump() for segment in transcript],
            ensure_ascii=False,
        )
        return (
            f"{self.task_prompt}\n\n"
            f"Верни только JSON-объект по этой JSON Schema:\n{schema}\n\n"
            f"Транскрипт:\n{transcript_json}"
        )


class LLMClassifierAgent(LLMValidatedAgent):
    def __init__(self, client: OpenAICompatibleClient) -> None:
        fallback_agent = FallbackClassifierAgent()
        super().__init__(
            client=client,
            agent_name="classifier",
            task_prompt=CLASSIFIER_PROMPT,
            response_model=Classification,
            fallback=fallback_agent.analyze,
        )


class LLMQualityAgent(LLMValidatedAgent):
    def __init__(self, client: OpenAICompatibleClient) -> None:
        fallback_agent = FallbackQualityAgent()
        super().__init__(
            client=client,
            agent_name="quality",
            task_prompt=QUALITY_PROMPT,
            response_model=QualityScore,
            fallback=fallback_agent.analyze,
        )


class LLMComplianceAgent(LLMValidatedAgent):
    def __init__(self, client: OpenAICompatibleClient) -> None:
        fallback_agent = FallbackComplianceAgent()
        super().__init__(
            client=client,
            agent_name="compliance",
            task_prompt=COMPLIANCE_PROMPT,
            response_model=ComplianceResult,
            fallback=fallback_agent.analyze,
        )


class LLMSummarizerAgent(LLMValidatedAgent):
    def __init__(self, client: OpenAICompatibleClient) -> None:
        fallback_agent = FallbackSummarizerAgent()
        super().__init__(
            client=client,
            agent_name="summarizer",
            task_prompt=SUMMARIZER_PROMPT,
            response_model=SummaryResult,
            fallback=fallback_agent.analyze,
        )
