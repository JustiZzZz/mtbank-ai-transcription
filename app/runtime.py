"""Сборка и запуск полного анализа аудио."""

from __future__ import annotations

import asyncio
import ipaddress
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.agents.base import normalize_text
from app.asr.diarizer import FallbackDiarizer
from app.asr.transcriber import FasterWhisperTranscriber
from app.audio.loader import AudioLoader, AudioValidationError
from app.audio.normalize import AudioNormalizer
from app.config import Settings, get_settings
from app.orchestration.supervisor import FallbackSupervisor
from app.schemas import AnalysisResult
from app.service import AnalysisService

DOWNLOAD_TIMEOUT_SECONDS = 20
UPLOAD_CHUNK_SIZE = 1024 * 1024


class AudioInputError(ValueError):
    """Ошибка входного файла или URL."""


def create_analysis_service(settings: Settings | None = None) -> AnalysisService:
    """Собирает общий AnalysisService для API и OpenWebUI Pipeline."""
    resolved_settings = settings or get_settings()
    return AnalysisService(
        transcriber=FasterWhisperTranscriber(resolved_settings),
        diarizer=FallbackDiarizer(),
        supervisor=FallbackSupervisor(),
    )


class AnalysisRuntime:
    """Полный runtime: загрузка, нормализация и бизнес-анализ."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        loader: AudioLoader | None = None,
        normalizer: AudioNormalizer | None = None,
        service: AnalysisService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loader = loader or AudioLoader(self.settings)
        self.normalizer = normalizer or AudioNormalizer(
            sample_rate=self.settings.audio_normalized_sample_rate,
            channels=self.settings.audio_normalized_channels,
        )
        self.service = service or create_analysis_service(self.settings)
        self.work_dir = Path(self.settings.audio_storage_dir)
        self.normalized_dir = self.work_dir / "normalized"

    async def preload(self) -> None:
        """Заранее прогреть тяжелые компоненты runtime."""
        preload = getattr(self.service.transcriber, "preload", None)
        if preload is not None:
            await preload()

    async def analyze_path(self, path: str | Path) -> AnalysisResult:
        try:
            stored = self.loader.store_local_file(path)
            normalized = self.normalizer.normalize(stored.path, self.normalized_dir)
        except AudioValidationError as exc:
            raise AudioInputError(str(exc)) from exc
        return await self.service.analyze_audio_path(
            normalized,
            metadata={
                "audio_storage_backend": stored.storage_backend,
                "audio_storage_uri": stored.storage_uri,
                "original_audio_name": stored.original_name,
                "normalized_audio_path": str(normalized),
            },
        )

    async def analyze_upload(self, upload: Any) -> AnalysisResult:
        suffix = Path(upload.filename or "audio").suffix.lower()
        temp_path = await self._write_upload_to_temp(upload, suffix)
        return await self.analyze_path(temp_path)

    async def analyze_url(self, url: str) -> AnalysisResult:
        temp_path = await asyncio.to_thread(self._download_url_to_temp, url)
        return await self.analyze_path(temp_path)

    async def _write_upload_to_temp(self, upload: Any, suffix: str) -> Path:
        temp_path = self._new_temp_path(suffix)
        max_bytes = self.settings.max_audio_mb * 1024 * 1024
        written = 0
        with temp_path.open("wb") as output:
            while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    temp_path.unlink(missing_ok=True)
                    msg = "Audio upload exceeds configured size limit."
                    raise AudioInputError(msg)
                output.write(chunk)
        return temp_path

    def _download_url_to_temp(self, url: str) -> Path:
        parsed = self._validate_url(url)
        suffix = Path(parsed.path).suffix.lower()
        temp_path = self._new_temp_path(suffix)
        max_bytes = self.settings.max_audio_mb * 1024 * 1024
        written = 0
        request = urllib.request.Request(url, headers={"User-Agent": "mtbank-ai-transcription/0.1"})
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with temp_path.open("wb") as output:
                while chunk := response.read(UPLOAD_CHUNK_SIZE):
                    written += len(chunk)
                    if written > max_bytes:
                        temp_path.unlink(missing_ok=True)
                        msg = "Audio URL exceeds configured size limit."
                        raise AudioInputError(msg)
                    output.write(chunk)
        return temp_path

    def _new_temp_path(self, suffix: str) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "mtbank-ai-transcription"
        temp_dir.mkdir(parents=True, exist_ok=True)
        extension = suffix if suffix.startswith(".") else ""
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=extension, dir=temp_dir)
        handle.close()
        return Path(handle.name)

    def _validate_url(self, url: str):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            msg = "Only http/https audio URLs are supported."
            raise AudioInputError(msg)
        host = normalize_text(parsed.hostname)
        if host in {"localhost", "127.0.0.1", "::1"}:
            msg = "Local audio URLs are not allowed."
            raise AudioInputError(msg)
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return parsed
        if address.is_private or address.is_loopback or address.is_link_local:
            msg = "Private network audio URLs are not allowed."
            raise AudioInputError(msg)
        return parsed


@lru_cache(maxsize=1)
def get_analysis_runtime() -> AnalysisRuntime:
    """Кешированный runtime для FastAPI production-запуска."""
    return AnalysisRuntime()
