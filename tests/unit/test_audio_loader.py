"""Unit-тесты безопасной загрузки аудио."""

from pathlib import Path

import pytest

from app.audio.loader import AudioLoader, AudioValidationError, LocalAudioStorage
from app.config import Settings


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        max_audio_mb=1,
        allowed_audio_extensions=[".wav", ".mp3", ".ogg", ".mp4"],
        audio_storage_dir=str(tmp_path / "stored"),
    )


@pytest.mark.parametrize("suffix", [".wav", ".mp3", ".ogg", ".mp4", ".WAV"])
def test_loader_accepts_supported_media_formats(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"call{suffix}"
    source.write_bytes(b"audio")
    loader = AudioLoader(make_settings(tmp_path))

    stored = loader.store_local_file(source)

    assert stored.path.exists()
    assert stored.extension == suffix.lower()
    assert stored.size_bytes == 5


def test_loader_rejects_disallowed_extension(tmp_path: Path) -> None:
    source = tmp_path / "call.exe"
    source.write_bytes(b"audio")
    loader = AudioLoader(make_settings(tmp_path))

    with pytest.raises(AudioValidationError):
        loader.store_local_file(source)


def test_loader_rejects_file_that_is_too_large(tmp_path: Path) -> None:
    source = tmp_path / "large.wav"
    source.write_bytes(b"0" * (1024 * 1024 + 1))
    loader = AudioLoader(make_settings(tmp_path))

    with pytest.raises(AudioValidationError):
        loader.store_local_file(source)


def test_storage_uses_safe_unique_filename(tmp_path: Path) -> None:
    source = tmp_path / ".. weird name .mp3"
    source.write_bytes(b"audio")
    storage = LocalAudioStorage(tmp_path / "stored")

    stored = storage.save(source, ".mp3")

    assert stored.name.endswith(".mp3")
    assert ".." not in stored.name
    assert " " not in stored.name
    assert stored.parent == tmp_path / "stored"


def test_loader_requires_custom_storage_for_s3_backend(tmp_path: Path) -> None:
    settings = Settings(audio_storage_backend="s3", audio_storage_dir=str(tmp_path))

    with pytest.raises(NotImplementedError):
        AudioLoader(settings)
