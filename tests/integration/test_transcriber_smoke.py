"""Ручной smoke-тест реального faster-whisper.

Запускать только локально, когда модель и тестовый файл готовы:
RUN_ASR_SMOKE=1 py -m pytest tests/integration/test_transcriber_smoke.py -m asr
"""

import os
from pathlib import Path

import pytest

from app.asr.transcriber import FasterWhisperTranscriber


@pytest.mark.asr
@pytest.mark.skipif(os.getenv("RUN_ASR_SMOKE") != "1", reason="ASR smoke disabled")
async def test_faster_whisper_smoke() -> None:
    audio_path = Path(os.environ["ASR_SMOKE_AUDIO"])

    result = await FasterWhisperTranscriber().transcribe(audio_path)

    assert result
    assert all(segment.text for segment in result)
