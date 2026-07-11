"""Интеграционный тест OpenWebUI pipeline formatting."""

from pathlib import Path

import pipeline as pipeline_module
from app.schemas import (
    AnalysisResult,
    Classification,
    ComplianceResult,
    QualityChecklist,
    QualityScore,
    SummaryResult,
    TranscriptSegment,
)
from app.service import AgentAnalysis


def sample_result() -> AnalysisResult:
    return AnalysisResult(
        transcript=[
            TranscriptSegment(speaker="Оператор", start=0.0, end=1.0, text="Добрый день"),
            TranscriptSegment(speaker="Клиент", start=1.1, end=2.0, text="Хочу кредит"),
        ],
        classification=Classification(
            topic="кредиты",
            priority="medium",
            confidence=0.9,
            confidence_label="high",
            rationale="Кредитная тематика.",
        ),
        quality_score=QualityScore(
            total=75,
            checklist=QualityChecklist(greeting=True, need_detection=True),
            comments=["Нет прощания."],
        ),
        compliance=ComplianceResult(passed=True),
        summary="Клиент спросил про кредит.",
        action_items=["Отправить условия."],
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.service = FakeService()

    async def analyze_path(self, path: str | Path) -> AnalysisResult:
        self.paths.append(Path(path))
        return sample_result()


class FakeDiarizer:
    async def diarize(self, segments):
        return [
            segment.model_copy(update={"speaker": "Оператор" if index % 2 == 0 else "Клиент"})
            for index, segment in enumerate(segments)
        ]


class FakeSupervisor:
    def __init__(self) -> None:
        self.transcripts: list[list[TranscriptSegment]] = []

    async def analyze(self, transcript):
        self.transcripts.append(list(transcript))
        return AgentAnalysis(
            classification=sample_result().classification,
            quality_score=sample_result().quality_score,
            compliance=sample_result().compliance,
            summary=SummaryResult(
                summary="Клиент спросил про кредит.",
                action_items=["Отправить условия."],
            ),
            metadata={"agents_mode": "fake"},
        )


class FakeService:
    def __init__(self) -> None:
        self.diarizer = FakeDiarizer()
        self.supervisor = FakeSupervisor()


async def test_pipeline_formats_analysis_markdown(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    pipe = pipeline_module.Pipeline(runtime=runtime)
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"audio")

    output = await pipe._pipe_async({"files": [{"path": str(audio_path)}]})

    assert runtime.paths == [audio_path]
    assert "## Анализ звонка" in output
    assert "**Тема:** кредиты" in output
    assert "| Оператор | 0.0 | 1.0 | Добрый день |" in output
    assert "Отправить условия." in output


def test_pipeline_accepts_openwebui_pipelines_call_shape() -> None:
    pipe = pipeline_module.Pipeline(runtime=FakeRuntime())

    body = pipe._coerce_body(
        user_message="Проанализируй https://example.com/call.mp3",
        messages=[{"role": "user", "content": "Проанализируй https://example.com/call.mp3"}],
        body={
            "model": "mtbank_ai_transcription",
            "files": [{"path": "call.wav"}],
        },
    )

    assert body["files"] == [{"path": "call.wav"}]
    assert body["messages"][-1]["content"].endswith("call.mp3")


def test_pipeline_extracts_nested_openwebui_file_path(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "call04_compliance_risky.wav"
    audio_path.write_bytes(b"audio")
    pipe = pipeline_module.Pipeline(runtime=FakeRuntime())

    reference = pipe._extract_audio_reference(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "",
                    "files": [
                        {
                            "type": "file",
                            "file": {
                                "id": "file-id",
                                "filename": audio_path.name,
                                "path": str(audio_path),
                            },
                        }
                    ],
                }
            ]
        }
    )

    assert reference == str(audio_path)


def test_pipeline_falls_back_to_latest_openwebui_upload(tmp_path, monkeypatch) -> None:
    older = tmp_path / "old_call.wav"
    newer = tmp_path / "new_call.mp3"
    ignored = tmp_path / "new_call.json"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    ignored.write_bytes(b"{}")
    older.touch()
    newer.touch()
    monkeypatch.setattr(pipeline_module, "OPENWEBUI_UPLOAD_DIR", tmp_path)
    pipe = pipeline_module.Pipeline(runtime=FakeRuntime())

    assert pipe._extract_audio_reference({"messages": [{"role": "user", "content": ""}]}) == str(
        newer
    )


def test_pipeline_prefers_openwebui_transcript_sidecar(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "file-id_call.wav"
    sidecar = tmp_path / "file-id_call.json"
    audio.write_bytes(b"audio")
    sidecar.write_text(
        '{"text": "Добрый день, МТБанк. Хочу узнать условия кредита наличными."}',
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_module, "OPENWEBUI_UPLOAD_DIR", tmp_path)
    runtime = FakeRuntime()
    pipe = pipeline_module.Pipeline(runtime=runtime)

    output = pipe.pipe({"messages": [{"role": "user", "content": ""}]})

    assert runtime.paths == []
    assert "**Тема:** кредиты" in output
    assert runtime.service.supervisor.transcripts[0][0].speaker == "Оператор"


def test_pipeline_does_not_reuse_previous_analysis_as_transcript(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "file-id_call06_unknown_nonsense.flac"
    sidecar = tmp_path / "file-id_call06_unknown_nonsense.json"
    audio.write_bytes(b"audio")
    sidecar.write_text(
        '{"text": "Алло. У меня на столе зеленая лампа и три пустые коробки."}',
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_module, "OPENWEBUI_UPLOAD_DIR", tmp_path)
    runtime = FakeRuntime()
    pipe = pipeline_module.Pipeline(runtime=runtime)

    output = pipe.pipe(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "## Анализ звонка\n\n"
                        "### Транскрипт\n"
                        "| Спикер | Старт | Конец | Текст |"
                    ),
                },
                {"role": "user", "content": "Проанализируй загруженный звонок"},
            ]
        }
    )

    analyzed_text = " ".join(segment.text for segment in runtime.service.supervisor.transcripts[0])
    assert runtime.paths == []
    assert "## Анализ звонка" in output
    assert "зеленая лампа" in analyzed_text
    assert "### Транскрипт" not in analyzed_text
