"""Fallback-суммаризатор транскрипта."""

from collections.abc import Sequence

from app.agents.base import contains_any, unique_preserve_order
from app.schemas import SummaryResult, TranscriptSegment

ACTION_KEYWORDS = (
    "отправ",
    "передам",
    "перезвон",
    "создам",
    "оформ",
    "проверю",
    "уточню",
    "подготов",
    "зарегистр",
    "свяж",
)
CLIENT_NEED_KEYWORDS = (
    "хочу",
    "нужно",
    "проблем",
    "жалоб",
    "не могу",
    "не совершал",
    "списание",
    "не дош",
    "заблок",
)
SOLUTION_KEYWORDS = (
    "можно",
    "предлагаю",
    "могу",
    "нужно",
    "важно",
    "провер",
    "оформ",
    "создам",
    "заблок",
    "передам",
)


class FallbackSummarizerAgent:
    """Делает короткое резюме только из текста звонка."""

    async def analyze(self, transcript: Sequence[TranscriptSegment]) -> SummaryResult:
        texts = [segment.text.strip() for segment in transcript if segment.text.strip()]
        summary = self._build_summary(transcript, texts)
        action_items = unique_preserve_order(
            text for text in texts if any(keyword in text.lower() for keyword in ACTION_KEYWORDS)
        )
        return SummaryResult(summary=summary, action_items=action_items[:5])

    def _build_summary(self, transcript: Sequence[TranscriptSegment], texts: list[str]) -> str:
        if not texts:
            return "Транскрипт пуст."

        client_need = self._first_matching(transcript, "Клиент", CLIENT_NEED_KEYWORDS)
        operator_solution = self._first_matching(transcript, "Оператор", SOLUTION_KEYWORDS)
        action_item = self._first_text_matching(texts, ACTION_KEYWORDS, exclude=operator_solution)

        sentences: list[str] = []
        if client_need:
            sentences.append(f"Клиент сообщил: {client_need}.")
        else:
            sentences.append(f"В разговоре зафиксировано: {texts[0]}.")

        if operator_solution:
            sentences.append(f"Оператор предложил: {operator_solution}.")
        elif len(texts) > 1:
            sentences.append(f"Дополнительный контекст: {texts[1]}.")

        if action_item and action_item != operator_solution:
            sentences.append(f"Следующий шаг: {action_item}.")

        return " ".join(sentences[:4])

    def _first_matching(
        self,
        transcript: Sequence[TranscriptSegment],
        speaker: str,
        keywords: Sequence[str],
    ) -> str | None:
        for segment in transcript:
            text = segment.text.strip()
            if segment.speaker == speaker and contains_any(text.lower(), keywords):
                return text
        return None

    def _first_text_matching(
        self,
        texts: Sequence[str],
        keywords: Sequence[str],
        *,
        exclude: str | None = None,
    ) -> str | None:
        for text in texts:
            if text == exclude:
                continue
            if contains_any(text.lower(), keywords):
                return text
        return None
