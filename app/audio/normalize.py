"""Нормализация аудио через ffmpeg перед ASR."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Запускает ffmpeg и возвращает результат для проверки ошибок."""
    return subprocess.run(command, capture_output=True, check=False, text=True)


class AudioNormalizer:
    """Конвертирует входное аудио в mono WAV для Whisper."""

    def __init__(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        sample_rate: int = 16000,
        channels: int = 1,
        runner: Runner = default_runner,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.runner = runner

    def normalize(self, source_path: str | Path, output_dir: str | Path) -> Path:
        source = Path(source_path)
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{source.stem}.wav"
        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-f",
            "wav",
            str(target),
        ]
        result = self.runner(command)
        if result.returncode != 0:
            msg = f"ffmpeg failed: {result.stderr.strip() or 'unknown error'}"
            raise RuntimeError(msg)
        return target
