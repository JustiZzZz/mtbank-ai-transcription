"""Unit-тесты fallback diarizer."""

from app.asr.diarizer import FallbackDiarizer
from app.schemas import TranscriptSegment


def segment(index: int, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        speaker="UNKNOWN",
        start=float(index * 3),
        end=float(index * 3 + 2),
        text=text,
    )


async def test_diarizer_labels_operator_opener_and_alternates() -> None:
    result = await FallbackDiarizer().diarize(
        [
            segment(0, "Добрый день, МТБанк, меня зовут Анна"),
            segment(1, "Здравствуйте, хочу узнать про кредит"),
            segment(2, "Подскажите сумму и срок"),
        ]
    )

    assert [item.speaker for item in result] == ["Оператор", "Клиент", "Оператор"]


async def test_diarizer_keeps_unknown_when_confidence_is_low() -> None:
    result = await FallbackDiarizer().diarize(
        [
            segment(0, "Алло"),
            segment(1, "Да"),
        ]
    )

    assert [item.speaker for item in result] == ["UNKNOWN", "UNKNOWN"]


async def test_diarizer_preserves_existing_speaker_labels() -> None:
    transcript = [
        TranscriptSegment(speaker="Клиент", start=0, end=1, text="Здравствуйте"),
        TranscriptSegment(speaker="Оператор", start=2, end=3, text="Добрый день"),
    ]

    result = await FallbackDiarizer().diarize(transcript)

    assert result == transcript
