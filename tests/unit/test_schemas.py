"""Проверяем базовые контракты JSON-ответов."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnalysisResult,
    Classification,
    ComplianceIssue,
    ComplianceResult,
    QualityChecklist,
    QualityScore,
    SummaryResult,
    TranscriptSegment,
)


def test_analysis_result_serializes_contract() -> None:
    segment = TranscriptSegment(speaker="Оператор", start=0.0, end=2.4, text="Добрый день")
    result = AnalysisResult(
        transcript=[segment],
        classification=Classification(
            topic="кредиты",
            priority="medium",
            confidence=0.82,
            rationale="Клиент спрашивает про кредит.",
        ),
        quality_score=QualityScore(
            total=70,
            checklist=QualityChecklist(greeting=True, need_detection=True, solution_provided=True),
            comments=["Нет прощания."],
        ),
        compliance=ComplianceResult(passed=True, issues=[]),
        summary="Клиент уточнил условия кредита.",
        action_items=[],
        metadata={"diarization_backend": "fallback"},
    )

    payload = result.model_dump()

    assert payload["transcript"][0]["speaker"] == "Оператор"
    assert payload["classification"]["topic"] == "кредиты"
    assert payload["quality_score"]["total"] == 70
    assert payload["compliance"]["passed"] is True
    assert payload["summary"] == "Клиент уточнил условия кредита."
    assert payload["action_items"] == []


def test_transcript_segment_rejects_wrong_time_order() -> None:
    with pytest.raises(ValidationError):
        TranscriptSegment(speaker="Клиент", start=5.0, end=3.0, text="Здравствуйте")


def test_enums_and_ranges_are_validated() -> None:
    with pytest.raises(ValidationError):
        Classification(topic="ипотека", priority="urgent", confidence=1.7, rationale="")

    with pytest.raises(ValidationError):
        QualityScore(total=120, checklist=QualityChecklist(), comments=[])


def test_compliance_issue_requires_clear_fields() -> None:
    issue = ComplianceIssue(
        severity="high",
        phrase="гарантируем одобрение",
        description="Нельзя обещать гарантированное одобрение кредита.",
        recommendation="Заменить на нейтральную формулировку.",
    )

    assert issue.severity == "high"


def test_summary_result_is_agent_output_contract() -> None:
    summary = SummaryResult(
        summary="Клиент попросил отправить условия по карте.",
        action_items=["Отправить условия по карте."],
    )

    assert summary.action_items == ["Отправить условия по карте."]
