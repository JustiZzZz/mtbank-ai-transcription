"""Unit-тесты ffmpeg-нормализации аудио."""

import subprocess
from pathlib import Path

from app.audio.normalize import AudioNormalizer


def test_normalizer_builds_ffmpeg_command(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    source = tmp_path / "call.ogg"
    source.write_bytes(b"audio")

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    normalizer = AudioNormalizer(ffmpeg_path="ffmpeg-test", runner=fake_runner)

    output = normalizer.normalize(source, tmp_path / "normalized")

    assert output == tmp_path / "normalized" / "call.wav"
    assert calls == [
        [
            "ffmpeg-test",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output),
        ]
    ]


def test_normalizer_raises_on_ffmpeg_failure(tmp_path: Path) -> None:
    source = tmp_path / "call.mp3"
    source.write_bytes(b"audio")

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "broken")

    normalizer = AudioNormalizer(runner=fake_runner)

    try:
        normalizer.normalize(source, tmp_path / "normalized")
    except RuntimeError as exc:
        assert "ffmpeg failed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
