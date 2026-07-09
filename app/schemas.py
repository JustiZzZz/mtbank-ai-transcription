"""Общие Pydantic-схемы для транскрипта и анализа звонка."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Speaker = Literal["Оператор", "Клиент", "UNKNOWN"]
Topic = Literal["кредиты", "карты", "переводы", "жалобы", "другое"]
Priority = Literal["low", "medium", "high"]
Severity = Literal["low", "medium", "high"]


class TranscriptSegment(BaseModel):
    """Одна реплика после ASR и диаризации."""

    speaker: Speaker
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_time_order(self) -> "TranscriptSegment":
        if self.end < self.start:
            msg = "end должен быть больше или равен start"
            raise ValueError(msg)
        return self


class Classification(BaseModel):
    """Тематика обращения и приоритет для супервайзера."""

    topic: Topic
    priority: Priority
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class QualityChecklist(BaseModel):
    """Чеклист качества разговора оператора."""

    greeting: bool = False
    identification_or_intro: bool = False
    need_detection: bool = False
    solution_provided: bool = False
    objection_handling: bool = False
    farewell: bool = False


class QualityScore(BaseModel):
    """Итоговая оценка качества и пояснения."""

    total: int = Field(ge=0, le=100)
    checklist: QualityChecklist
    comments: list[str] = Field(default_factory=list)


class ComplianceIssue(BaseModel):
    """Одна найденная compliance-проблема."""

    severity: Severity
    phrase: str = Field(min_length=1)
    description: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)


class ComplianceResult(BaseModel):
    """Результат compliance-проверки."""

    passed: bool
    issues: list[ComplianceIssue] = Field(default_factory=list)


class SummaryResult(BaseModel):
    """Краткое резюме и действия после звонка."""

    summary: str = Field(min_length=1)
    action_items: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Полный результат анализа, общий для FastAPI и OpenWebUI Pipeline."""

    transcript: list[TranscriptSegment]
    classification: Classification
    quality_score: QualityScore
    compliance: ComplianceResult
    summary: str = Field(min_length=1)
    action_items: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
