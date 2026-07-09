"""Unit-тесты deterministic fallback-агентов."""

import pytest

from app.agents.classifier import FallbackClassifierAgent
from app.agents.compliance import FallbackComplianceAgent
from app.agents.quality import FallbackQualityAgent
from app.agents.summarizer import FallbackSummarizerAgent
from app.orchestration.supervisor import FallbackSupervisor
from app.schemas import TranscriptSegment


def make_transcript(*texts: str) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            speaker="Оператор" if index % 2 == 0 else "Клиент",
            start=float(index * 3),
            end=float(index * 3 + 2),
            text=text,
        )
        for index, text in enumerate(texts)
    ]


@pytest.mark.parametrize(
    ("text", "topic"),
    [
        ("Хочу узнать условия кредита наличными", "кредиты"),
        ("Не проходит перевод по реквизитам", "переводы"),
        ("У меня проблема с картой и пин-кодом", "карты"),
        ("Не могу погасить Халву, какой минимальный платеж?", "карты"),
        ("Хочу оставить жалобу на обслуживание", "жалобы"),
    ],
)
async def test_classifier_detects_topic(text: str, topic: str) -> None:
    result = await FallbackClassifierAgent().analyze(make_transcript(text))

    assert result.topic == topic


async def test_classifier_uses_weighted_relevance_for_mixed_dialog() -> None:
    result = await FallbackClassifierAgent().analyze(
        make_transcript("Карта заблокирована, не могу снять наличные в банкомате")
    )

    assert result.topic == "карты"
    assert result.confidence >= 0.75


async def test_classifier_marks_complaint_as_high_priority() -> None:
    result = await FallbackClassifierAgent().analyze(
        make_transcript("Это срочная жалоба, прошу разобраться сегодня")
    )

    assert result.priority == "high"


async def test_classifier_marks_fraud_risk_as_high_priority() -> None:
    result = await FallbackClassifierAgent().analyze(
        make_transcript("Подозреваю мошенничество, списали деньги без моего согласия")
    )

    assert result.priority == "high"


async def test_compliance_flags_forbidden_credit_promise() -> None:
    result = await FallbackComplianceAgent().analyze(
        make_transcript("Мы гарантируем одобрение кредита каждому клиенту")
    )

    assert result.passed is False
    assert result.issues[0].severity == "high"
    assert result.issues[0].phrase == "гарантируем одобрение"


async def test_compliance_flags_missing_decision_disclaimer() -> None:
    result = await FallbackComplianceAgent().analyze(
        make_transcript("Кредит вам точно одобрен, заявку проверять не нужно")
    )

    assert result.passed is False
    assert result.issues[0].severity == "high"


async def test_quality_checklist_affects_score() -> None:
    result = await FallbackQualityAgent().analyze(
        [
            TranscriptSegment(
                speaker="Оператор",
                start=0,
                end=2,
                text="Добрый день, МТБанк, меня зовут Анна",
            ),
            TranscriptSegment(speaker="Клиент", start=3, end=5, text="Хочу узнать про кредит"),
            TranscriptSegment(
                speaker="Оператор",
                start=6,
                end=8,
                text="Подскажите, какая сумма и срок вам нужны?",
            ),
            TranscriptSegment(
                speaker="Оператор",
                start=9,
                end=11,
                text="Я могу предложить кредит на 24 месяца",
            ),
            TranscriptSegment(
                speaker="Оператор",
                start=12,
                end=14,
                text="Спасибо за обращение, до свидания",
            ),
        ]
    )

    assert result.checklist.greeting is True
    assert result.checklist.identification_or_intro is True
    assert result.checklist.need_detection is True
    assert result.checklist.solution_provided is True
    assert result.checklist.farewell is True
    assert result.total >= 80


async def test_quality_prefers_operator_phrases_for_checklist() -> None:
    transcript = [
        TranscriptSegment(
            speaker="Клиент",
            start=0,
            end=2,
            text="Здравствуйте, меня зовут Иван, хочу кредит",
        ),
        TranscriptSegment(
            speaker="Оператор",
            start=3,
            end=5,
            text="Подскажите сумму и срок, я проверю условия",
        ),
    ]

    result = await FallbackQualityAgent().analyze(transcript)

    assert result.checklist.greeting is False
    assert result.checklist.identification_or_intro is False
    assert result.checklist.need_detection is True


async def test_summarizer_uses_transcript_and_extracts_action_items() -> None:
    result = await FallbackSummarizerAgent().analyze(
        make_transcript(
            "Клиент просит условия по карте",
            "Отправлю информацию на email клиента сегодня",
        )
    )

    assert "Клиент просит условия по карте" in result.summary
    assert result.action_items == ["Отправлю информацию на email клиента сегодня"]


async def test_summarizer_deduplicates_action_items() -> None:
    result = await FallbackSummarizerAgent().analyze(
        make_transcript(
            "Отправлю информацию на email клиента сегодня",
            "Отправлю информацию на email клиента сегодня",
            "Перезвоню завтра после проверки заявки",
        )
    )

    assert result.action_items == [
        "Отправлю информацию на email клиента сегодня",
        "Перезвоню завтра после проверки заявки",
    ]


async def test_supervisor_returns_agent_analysis_contract() -> None:
    result = await FallbackSupervisor().analyze(
        make_transcript(
            "Добрый день, МТБанк, меня зовут Анна",
            "У меня жалоба по кредиту",
            "Отправлю обращение специалисту",
        )
    )

    assert result.classification.topic == "жалобы"
    assert result.classification.priority == "high"
    assert result.quality_score.checklist.greeting is True
    assert result.compliance.passed is True
    assert result.summary.summary
    assert result.metadata == {"agents_mode": "fallback"}
