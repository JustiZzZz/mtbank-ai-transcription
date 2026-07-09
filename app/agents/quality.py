"""Fallback-оценка качества разговора оператора."""

from collections.abc import Sequence

from app.agents.base import contains_any, speaker_text, transcript_text
from app.schemas import QualityChecklist, QualityScore, TranscriptSegment

CHECK_WEIGHTS = {
    "greeting": 15,
    "identification_or_intro": 10,
    "need_detection": 20,
    "solution_provided": 25,
    "objection_handling": 10,
    "farewell": 20,
}
GREETINGS = ("добрый день", "доброе утро", "добрый вечер", "здравствуйте", "приветствую")
INTRO = ("меня зовут", "мтбанк", "контакт-центр", "оператор", "специалист")
NEED_DETECTION = (
    "подскажите",
    "уточните",
    "какая сумма",
    "какой срок",
    "что вас интересует",
    "чем могу помочь",
    "правильно понимаю",
    "для какой цели",
    "какой вопрос",
)
SOLUTION = (
    "могу предложить",
    "предлагаю",
    "можно оформить",
    "отправлю",
    "передам",
    "создам обращение",
    "проверю",
    "рассчитаю",
    "объясню",
    "вам необходимо",
)
OBJECTION = (
    "понимаю",
    "давайте проверим",
    "разберемся",
    "сейчас уточню",
    "могу предложить альтернативу",
    "проверим другой вариант",
)
FAREWELL = (
    "до свидания",
    "спасибо за обращение",
    "хорошего дня",
    "всего доброго",
    "были рады помочь",
)


class FallbackQualityAgent:
    """Считает чеклист по очевидным фразам оператора."""

    async def analyze(self, transcript: Sequence[TranscriptSegment]) -> QualityScore:
        text = speaker_text(transcript, "Оператор") or transcript_text(transcript)
        checklist = QualityChecklist(
            greeting=contains_any(text, GREETINGS),
            identification_or_intro=contains_any(text, INTRO),
            need_detection=contains_any(text, NEED_DETECTION),
            solution_provided=contains_any(text, SOLUTION),
            objection_handling=contains_any(text, OBJECTION),
            farewell=contains_any(text, FAREWELL),
        )
        total = sum(weight for field, weight in CHECK_WEIGHTS.items() if getattr(checklist, field))
        comments = [
            label
            for field, label in (
                ("greeting", "Нет явного приветствия."),
                ("identification_or_intro", "Оператор не представился или не назвал банк."),
                ("need_detection", "Не зафиксировано выявление потребности."),
                ("solution_provided", "Не найдено предложенное решение."),
                ("farewell", "Нет явного прощания."),
            )
            if not getattr(checklist, field)
        ]
        return QualityScore(total=total, checklist=checklist, comments=comments)
