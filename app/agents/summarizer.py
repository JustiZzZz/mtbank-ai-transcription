"""Fallback-суммаризатор транскрипта."""

from collections.abc import Sequence

from app.agents.base import unique_preserve_order
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


class FallbackSummarizerAgent:
    """Делает короткое резюме только из текста звонка."""

    async def analyze(self, transcript: Sequence[TranscriptSegment]) -> SummaryResult:
        texts = [segment.text.strip() for segment in transcript if segment.text.strip()]
        summary = " ".join(texts[:2]) if texts else "Транскрипт пуст."
        action_items = unique_preserve_order(
            text for text in texts if any(keyword in text.lower() for keyword in ACTION_KEYWORDS)
        )
        return SummaryResult(summary=summary, action_items=action_items[:5])
