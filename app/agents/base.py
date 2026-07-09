"""Общие утилиты fallback-агентов."""

from collections.abc import Iterable, Sequence

from app.schemas import TranscriptSegment


def normalize_text(text: str) -> str:
    """Нормализует русский текст для простого keyword matching."""
    return " ".join(text.lower().replace("ё", "е").split())


def transcript_text(transcript: Sequence[TranscriptSegment]) -> str:
    """Склеивает текст транскрипта для keyword fallback-анализа."""
    return normalize_text(" ".join(segment.text for segment in transcript))


def speaker_text(transcript: Sequence[TranscriptSegment], speaker: str) -> str:
    """Склеивает реплики конкретного speaker; если их нет, вернет пустую строку."""
    return normalize_text(
        " ".join(segment.text for segment in transcript if segment.speaker == speaker)
    )


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    """Проверяет наличие любого ключевого слова или фразы."""
    return any(normalize_text(keyword) in text for keyword in keywords)


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Удаляет дубли, сохраняя исходный порядок."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result
