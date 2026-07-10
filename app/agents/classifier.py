"""Fallback-классификатор тематики обращения."""

from collections.abc import Sequence

from app.agents.base import contains_any, transcript_text
from app.schemas import Classification, TranscriptSegment

TOPIC_KEYWORDS: dict[str, tuple[tuple[str, int], ...]] = {
    "жалобы": (
        ("жалоб", 6),
        ("претензи", 6),
        ("недовол", 4),
        ("наруш", 4),
        ("обман", 5),
        ("ошибк", 3),
        ("некачествен", 4),
        ("разберитесь", 4),
        ("оставить отзыв", 3),
        ("плохое обслуживание", 5),
        ("не решили вопрос", 4),
    ),
    "карты": (
        ("карт", 5),
        ("карточ", 5),
        ("халв", 7),
        ("овердрафт", 5),
        ("грейс", 4),
        ("рассроч", 4),
        ("пин", 4),
        ("cvv", 4),
        ("3-d secure", 4),
        ("смс-код", 3),
        ("банкомат", 4),
        ("cash-in", 4),
        ("снятие налич", 4),
        ("лимит", 3),
        ("блокиров", 4),
        ("перевыпуск", 4),
        ("дополнительная карта", 4),
        ("бесконтакт", 3),
        ("apple pay", 3),
        ("google pay", 3),
    ),
    "кредиты": (
        ("кредит", 6),
        ("займ", 5),
        ("ставк", 4),
        ("потребительск", 4),
        ("рефинанс", 5),
        ("погаш", 3),
        ("задолж", 4),
        ("ежемесячный платеж", 4),
        ("график платеж", 4),
        ("полная стоимость кредита", 5),
        ("пск", 4),
        ("договор кредит", 4),
        ("справк", 3),
        ("поручител", 3),
        ("досроч", 3),
        ("отсроч", 3),
        ("кредитная заявка", 5),
    ),
    "переводы": (
        ("перевод", 6),
        ("платеж", 4),
        ("ерип", 5),
        ("iban", 5),
        ("swift", 5),
        ("реквизит", 5),
        ("международн", 4),
        ("денежный перевод", 6),
        ("с карты на карту", 5),
        ("p2p", 4),
        ("moby", 3),
        ("интернет-банк", 3),
        ("mybank", 3),
        ("denegram", 4),
        ("комисси", 3),
        ("зачислен", 3),
        ("не дошли деньги", 5),
    ),
}
HIGH_PRIORITY_KEYWORDS = (
    "срочно",
    "жалоб",
    "претензи",
    "мошен",
    "списали деньги",
    "без моего согласия",
    "украли",
    "компрометац",
    "заблок",
    "карта потеряна",
    "не могу пользоваться",
    "угроза",
)
MEDIUM_PRIORITY_KEYWORDS = (
    "не работает",
    "не проходит",
    "не пришли",
    "ошибка",
    "проблем",
    "задерж",
)


def score_topic(text: str, keywords: tuple[tuple[str, int], ...]) -> int:
    """Считает взвешенную релевантность темы."""
    return sum(weight for keyword, weight in keywords if keyword in text)


def confidence_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


class FallbackClassifierAgent:
    """Детерминированный классификатор для fallback-режима."""

    async def analyze(self, transcript: Sequence[TranscriptSegment]) -> Classification:
        text = transcript_text(transcript)
        scores = {topic: score_topic(text, keywords) for topic, keywords in TOPIC_KEYWORDS.items()}
        topic, score = max(scores.items(), key=lambda item: item[1])
        if score == 0:
            topic = "другое"
            confidence = 0.35
        elif score >= 8:
            confidence = 0.88
        elif score >= 5:
            confidence = 0.76
        else:
            confidence = 0.58

        priority = "high" if contains_any(text, HIGH_PRIORITY_KEYWORDS) else "medium"
        if priority != "high" and not contains_any(text, MEDIUM_PRIORITY_KEYWORDS):
            priority = "medium" if topic != "другое" else "low"
        if topic == "другое":
            priority = "high" if priority == "high" else "low"

        return Classification(
            topic=topic,
            priority=priority,
            confidence=confidence,
            confidence_label=confidence_label(confidence),
            rationale=f"Fallback keyword analysis detected topic '{topic}'.",
        )
