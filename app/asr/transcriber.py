"""faster-whisper wrapper для ASR."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.schemas import TranscriptSegment

logger = logging.getLogger(__name__)


class WhisperSegment(Protocol):
    """Минимальный контракт сегмента faster-whisper."""

    start: float
    end: float
    text: str


class WhisperModelLike(Protocol):
    """Минимальный контракт модели faster-whisper для тестов."""

    def transcribe(self, audio_path: str, **kwargs: Any) -> tuple[Sequence[WhisperSegment], Any]:
        """Вернуть сегменты и info, как faster-whisper."""


ModelFactory = Callable[..., WhisperModelLike]
BatchPipelineFactory = Callable[[WhisperModelLike], WhisperModelLike]


class ASRRuntimeError(RuntimeError):
    """Понятная ошибка ASR runtime/configuration."""


def default_model_factory(**kwargs: Any) -> WhisperModelLike:
    """Лениво импортирует faster-whisper, чтобы unit-тесты не требовали модель."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        msg = "Install faster-whisper to run real ASR: pip install faster-whisper"
        raise RuntimeError(msg) from exc
    return WhisperModel(**kwargs)


def default_batch_pipeline_factory(model: WhisperModelLike) -> WhisperModelLike:
    """Оборачивает модель в штатный batched runner faster-whisper."""
    try:
        from faster_whisper import BatchedInferencePipeline
    except ImportError as exc:
        msg = "Install faster-whisper to run batched ASR: pip install faster-whisper"
        raise RuntimeError(msg) from exc
    return BatchedInferencePipeline(model=model)


class FasterWhisperTranscriber:
    """Транскрибирует аудио через faster-whisper и возвращает Pydantic-сегменты."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_factory: ModelFactory = default_model_factory,
        batch_pipeline_factory: BatchPipelineFactory = default_batch_pipeline_factory,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_factory = model_factory
        self.batch_pipeline_factory = batch_pipeline_factory
        self._model: WhisperModelLike | None = None
        self._pipeline: WhisperModelLike | None = None

    @property
    def model(self) -> WhisperModelLike:
        if self._model is None:
            started_at = time.perf_counter()
            logger.info(
                "Loading faster-whisper model %s on %s with %s",
                self.settings.whisper_model,
                self.settings.whisper_device,
                self.settings.whisper_compute_type,
            )
            try:
                self._model = self.model_factory(
                    model_size_or_path=self.settings.whisper_model,
                    device=self.settings.whisper_device,
                    compute_type=self.settings.whisper_compute_type,
                    cpu_threads=self.settings.whisper_cpu_threads,
                    num_workers=self.settings.whisper_num_workers,
                    download_root=self.settings.whisper_download_root,
                    local_files_only=self.settings.whisper_local_files_only,
                    use_auth_token=self.settings.hf_token or None,
                )
            except Exception as exc:
                raise ASRRuntimeError(self._format_runtime_error(exc)) from exc
            logger.info(
                "Loaded faster-whisper model in %.2fs",
                time.perf_counter() - started_at,
            )
        return self._model

    @property
    def pipeline(self) -> WhisperModelLike:
        if self.settings.whisper_batch_size <= 1:
            return self.model
        if self._pipeline is None:
            logger.info(
                "Using faster-whisper batched pipeline with batch_size=%s",
                self.settings.whisper_batch_size,
            )
            try:
                self._pipeline = self.batch_pipeline_factory(self.model)
            except Exception as exc:
                raise ASRRuntimeError(self._format_runtime_error(exc)) from exc
        return self._pipeline

    async def preload(self) -> None:
        """Заранее загрузить модель, чтобы первый пользовательский запрос не ждал download."""
        await asyncio.to_thread(lambda: self.pipeline)

    async def transcribe(self, audio_path: str | Path) -> list[TranscriptSegment]:
        return await asyncio.to_thread(self._transcribe_sync, Path(audio_path))

    def _transcribe_sync(self, audio_path: Path) -> list[TranscriptSegment]:
        try:
            kwargs: dict[str, Any] = {
                "beam_size": self.settings.whisper_beam_size,
                "language": self.settings.whisper_language,
                "vad_filter": self.settings.whisper_vad_filter,
                "word_timestamps": self.settings.whisper_word_timestamps,
                "condition_on_previous_text": self.settings.whisper_condition_on_previous_text,
            }
            if self.settings.whisper_batch_size > 1:
                kwargs["batch_size"] = self.settings.whisper_batch_size

            segments, _info = self.pipeline.transcribe(
                str(audio_path),
                **kwargs,
            )
        except RuntimeError as exc:
            raise ASRRuntimeError(self._format_runtime_error(exc)) from exc
        return [
            TranscriptSegment(
                speaker="UNKNOWN",
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text.strip()
        ]

    def _format_runtime_error(self, exc: Exception) -> str:
        text = str(exc)
        if "cublas" in text.lower() or "cudnn" in text.lower():
            return (
                "ASR GPU runtime is not available. Install matching CUDA/cuDNN libraries "
                "or set WHISPER_DEVICE=cpu and WHISPER_COMPUTE_TYPE=int8."
            )
        if "bearer" in text.lower() or "illegal header value" in text.lower():
            return "HF_TOKEN is empty or invalid. Remove it from .env or set a valid token."
        return f"ASR runtime failed: {text}"
