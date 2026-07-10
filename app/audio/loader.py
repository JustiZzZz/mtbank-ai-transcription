"""Безопасная загрузка локальных аудиофайлов перед ASR."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.config import Settings, get_settings

REQUIRED_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg"})
DEFAULT_AUDIO_EXTENSIONS = REQUIRED_AUDIO_EXTENSIONS | frozenset({".m4a", ".flac", ".mp4"})
SAFE_STEM_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


class AudioValidationError(ValueError):
    """Ошибка проверки аудиофайла."""


@dataclass(frozen=True, slots=True)
class StoredAudio:
    """Метаданные сохраненной записи."""

    path: Path
    original_name: str
    extension: str
    size_bytes: int
    storage_backend: str = "local"
    storage_uri: str | None = None


class AudioStorage(Protocol):
    """Интерфейс хранения записи; S3/MinIO можно добавить без изменения loader."""

    backend: str

    def save(self, source_path: Path, extension: str) -> Path:
        """Сохранить source_path и вернуть путь или mount-path к объекту."""


class LocalAudioStorage:
    """Локальное хранилище для Docker volume или временной директории."""

    backend = "local"

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)

    def save(self, source_path: Path, extension: str) -> Path:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = SAFE_STEM_RE.sub("_", source_path.stem).strip("._") or "audio"
        target = self.storage_dir / f"{safe_stem}-{uuid4().hex}{extension}"
        shutil.copy2(source_path, target)
        return target


class AudioLoader:
    """Проверяет аудио и сохраняет его в выбранное storage."""

    def __init__(
        self,
        settings: Settings | None = None,
        storage: AudioStorage | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if storage is None and self.settings.audio_storage_backend != "local":
            msg = "Подключено только локальное хранилище аудио."
            raise NotImplementedError(msg)
        self.storage = storage or LocalAudioStorage(self.settings.audio_storage_dir)

    def store_local_file(self, source_path: str | Path) -> StoredAudio:
        source = Path(source_path)
        extension = self._validate_path(source)
        size_bytes = self._validate_size(source)
        stored_path = self.storage.save(source, extension)
        return StoredAudio(
            path=stored_path,
            original_name=source.name,
            extension=extension,
            size_bytes=size_bytes,
            storage_backend=self.storage.backend,
            storage_uri=str(stored_path),
        )

    def _validate_path(self, source: Path) -> str:
        if not source.is_file():
            msg = "Аудио-файла не существует"
            raise AudioValidationError(msg)
        extension = source.suffix.lower()
        allowed = {item.lower() for item in self.settings.allowed_audio_extensions}
        if extension not in allowed:
            msg = f"Неподдерживаемое расширение файла: {extension or '<none>'}."
            raise AudioValidationError(msg)
        return extension

    def _validate_size(self, source: Path) -> int:
        size_bytes = source.stat().st_size
        max_bytes = self.settings.max_audio_mb * 1024 * 1024
        if size_bytes > max_bytes:
            msg = f"Аудиофайл слишком велик: {size_bytes} байт, максимально {max_bytes} байт."
            raise AudioValidationError(msg)
        return size_bytes
