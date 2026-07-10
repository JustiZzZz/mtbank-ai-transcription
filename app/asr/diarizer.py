"""Базовая fallback-диаризация без внешних моделей."""

from __future__ import annotations

from collections.abc import Sequence

from app.agents.base import contains_any, normalize_text
from app.schemas import TranscriptSegment

OPERATOR_OPENERS = (
    "мтбанк",
    "мт банк",
    "меня зовут",
    "чем могу помочь",
    "контакт-центр",
    "оператор",
    "добрый день",
    "здравствуйте",
)


class FallbackDiarizer:
    """Назначает роли по простому правилу opener + alternation."""

    async def diarize(self, segments: Sequence[TranscriptSegment]) -> list[TranscriptSegment]:
        transcript = list(segments)
        if not transcript:
            return []
        if any(segment.speaker != "UNKNOWN" for segment in transcript):
            return transcript
        if not self._starts_with_operator(transcript[0]):
            return transcript
        return [
            segment.model_copy(update={"speaker": "Оператор" if index % 2 == 0 else "Клиент"})
            for index, segment in enumerate(transcript)
        ]

    def _starts_with_operator(self, first_segment: TranscriptSegment) -> bool:
        text = normalize_text(first_segment.text)
        has_bank_context = contains_any(text, ("мтбанк", "мт банк", "контакт-центр"))
        has_intro = contains_any(text, ("меня зовут", "оператор", "чем могу помочь"))
        has_greeting = contains_any(text, ("добрый день", "здравствуйте"))
        return has_bank_context or (has_greeting and has_intro)
