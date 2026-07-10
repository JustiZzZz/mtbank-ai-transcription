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
AgentMode = str


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
        result, _mode = await self.analyze_with_mode(transcript)
        return result

    async def analyze_with_mode(
        self,
        transcript: Sequence[TranscriptSegment],
    ) -> tuple[ResultModel, AgentMode]:
        user_prompt = self._build_user_prompt(transcript)
        last_error: Exception | None = None
        attempts = max(1, self.client.settings.llm_validation_retries + 1)
        for attempt in range(attempts):
            try:
                payload = await self.client.complete_json(
                    system_prompt=BASE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                result = self.response_model.model_validate(payload)
                return result, "llm"
            except ValidationError as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                user_prompt = self._build_repair_prompt(transcript, payload, exc)
            except (LLMClientError, ValueError, TypeError) as exc:
                last_error = exc
                break

        reason = last_error or "unknown error"
        logger.warning(
            "LLM agent %s failed with %s: %s; using fallback",
            self.agent_name,
            reason.__class__.__name__,
            reason,
        )
        return await self.fallback(transcript), "fallback"

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

    def _build_repair_prompt(
        self,
        transcript: Sequence[TranscriptSegment],
        invalid_payload: dict[str, object],
        error: ValidationError,
    ) -> str:
        return (
            f"{self._build_user_prompt(transcript)}\n\n"
            "Предыдущий JSON не прошел Pydantic validation. Исправь только JSON, "
            "не добавляй markdown или пояснения.\n"
            f"Ошибки validation:\n{error.errors()}\n"
            f"Предыдущий JSON:\n{json.dumps(invalid_payload, ensure_ascii=False)}"
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
