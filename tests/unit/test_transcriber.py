"""Unit-тесты faster-whisper wrapper без загрузки реальной модели."""

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.asr.transcriber import ASRRuntimeError, FasterWhisperTranscriber
from app.config import Settings


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio_path: str, **kwargs: object) -> tuple[list[object], object]:
        self.calls.append({"audio_path": audio_path, **kwargs})
        return (
            [
                SimpleNamespace(start=0.0, end=1.2, text=" Добрый день "),
                SimpleNamespace(start=1.4, end=2.0, text=""),
                SimpleNamespace(start=2.1, end=3.4, text="Хочу кредит"),
            ],
            SimpleNamespace(language="ru"),
        )


async def test_transcriber_maps_whisper_segments() -> None:
    fake_model = FakeWhisperModel()

    def model_factory(**kwargs: object) -> FakeWhisperModel:
        assert kwargs == {
            "model_size_or_path": "medium",
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": 0,
            "num_workers": 1,
            "download_root": "var/models",
            "local_files_only": True,
            "use_auth_token": "hf_test",
        }
        return fake_model

    transcriber = FasterWhisperTranscriber(
        Settings(
            _env_file=None,
            whisper_model="medium",
            whisper_device="cpu",
            whisper_compute_type="int8",
            whisper_cpu_threads=0,
            whisper_beam_size=3,
            whisper_batch_size=1,
            whisper_language="ru",
            whisper_vad_filter=True,
            whisper_word_timestamps=False,
            whisper_condition_on_previous_text=False,
            whisper_download_root="var/models",
            whisper_local_files_only=True,
            hf_token="hf_test",
        ),
        model_factory=model_factory,
    )

    result = await transcriber.transcribe(Path("call.wav"))

    assert [segment.text for segment in result] == ["Добрый день", "Хочу кредит"]
    assert [segment.speaker for segment in result] == ["UNKNOWN", "UNKNOWN"]
    assert fake_model.calls == [
        {
            "audio_path": "call.wav",
            "beam_size": 3,
            "language": "ru",
            "vad_filter": True,
            "word_timestamps": False,
            "condition_on_previous_text": False,
        }
    ]


async def test_transcriber_uses_batched_pipeline_when_enabled() -> None:
    fake_model = FakeWhisperModel()
    fake_pipeline = FakeWhisperModel()
    wrapped_models: list[FakeWhisperModel] = []

    def batch_pipeline_factory(model: FakeWhisperModel) -> FakeWhisperModel:
        wrapped_models.append(model)
        return fake_pipeline

    transcriber = FasterWhisperTranscriber(
        Settings(
            _env_file=None,
            whisper_device="cpu",
            whisper_compute_type="int8",
            whisper_batch_size=8,
        ),
        model_factory=lambda **_: fake_model,
        batch_pipeline_factory=batch_pipeline_factory,
    )

    result = await transcriber.transcribe(Path("call.wav"))

    assert [segment.text for segment in result] == ["Добрый день", "Хочу кредит"]
    assert wrapped_models == [fake_model]
    assert fake_model.calls == []
    assert fake_pipeline.calls == [
        {
            "audio_path": "call.wav",
            "beam_size": 1,
            "language": "ru",
            "vad_filter": True,
            "word_timestamps": False,
            "condition_on_previous_text": False,
            "batch_size": 8,
        }
    ]


async def test_transcriber_preload_loads_model_once() -> None:
    created = 0

    def model_factory(**kwargs: object) -> FakeWhisperModel:
        nonlocal created
        created += 1
        return FakeWhisperModel()

    transcriber = FasterWhisperTranscriber(
        Settings(
            _env_file=None,
            whisper_device="cpu",
            whisper_compute_type="int8",
            whisper_batch_size=1,
        ),
        model_factory=model_factory,
    )

    await transcriber.preload()
    await transcriber.preload()

    assert created == 1


async def test_transcriber_wraps_cuda_runtime_errors() -> None:
    def model_factory(**kwargs: object) -> FakeWhisperModel:
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    transcriber = FasterWhisperTranscriber(
        Settings(_env_file=None, whisper_device="cuda", whisper_compute_type="int8_float16"),
        model_factory=model_factory,
    )

    with pytest.raises(ASRRuntimeError, match="WHISPER_DEVICE=cpu"):
        await transcriber.transcribe(Path("call.wav"))


async def test_transcriber_does_not_pass_empty_hf_token() -> None:
    received_kwargs: dict[str, object] = {}

    def model_factory(**kwargs: object) -> FakeWhisperModel:
        received_kwargs.update(kwargs)
        return FakeWhisperModel()

    transcriber = FasterWhisperTranscriber(
        Settings(
            _env_file=None,
            whisper_device="cpu",
            whisper_compute_type="int8",
            hf_token="",
            whisper_batch_size=1,
        ),
        model_factory=model_factory,
    )

    await transcriber.preload()

    assert received_kwargs["use_auth_token"] is None


async def test_transcriber_limits_cpu_transcribes_to_one_at_a_time() -> None:
    class SlowWhisperModel(FakeWhisperModel):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def transcribe(self, audio_path: str, **kwargs: object) -> tuple[list[object], object]:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.05)
                return super().transcribe(audio_path, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

    fake_model = SlowWhisperModel()
    transcriber = FasterWhisperTranscriber(
        Settings(
            _env_file=None,
            whisper_device="cpu",
            whisper_compute_type="int8",
            whisper_batch_size=1,
        ),
        model_factory=lambda **_: fake_model,
    )

    await asyncio.gather(
        transcriber.transcribe(Path("first.wav")),
        transcriber.transcribe(Path("second.wav")),
    )

    assert fake_model.max_active == 1
    assert [call["audio_path"] for call in fake_model.calls] == ["first.wav", "second.wav"]
