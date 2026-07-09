"""Unit-тест общего сервиса анализа на fake-компонентах."""

from pathlib import Path

from app.schemas import (
    Classification,
    ComplianceResult,
    QualityChecklist,
    QualityScore,
    SummaryResult,
    TranscriptSegment,
)
from app.service import AgentAnalysis, AnalysisService


async def test_analysis_service_orchestrates_components() -> None:
    calls: list[str] = []
    audio_path = Path("call.wav")
    raw_segments = [
        TranscriptSegment(speaker="UNKNOWN", start=0.0, end=1.5, text="Добрый день"),
        TranscriptSegment(speaker="UNKNOWN", start=1.8, end=3.2, text="Хочу кредит"),
    ]
    diarized_segments = [
        raw_segments[0].model_copy(update={"speaker": "Оператор"}),
        raw_segments[1].model_copy(update={"speaker": "Клиент"}),
    ]

    class FakeTranscriber:
        async def transcribe(self, path: Path) -> list[TranscriptSegment]:
            calls.append(f"transcribe:{path.name}")
            return raw_segments

    class FakeDiarizer:
        async def diarize(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
            calls.append(f"diarize:{segments[0].speaker}")
            return diarized_segments

    class FakeSupervisor:
        async def analyze(self, transcript: list[TranscriptSegment]) -> AgentAnalysis:
            calls.append(f"supervisor:{transcript[1].speaker}")
            return AgentAnalysis(
                classification=Classification(
                    topic="кредиты",
                    priority="medium",
                    confidence=0.9,
                    rationale="Клиент интересуется кредитом.",
                ),
                quality_score=QualityScore(
                    total=80,
                    checklist=QualityChecklist(
                        greeting=True,
                        need_detection=True,
                        solution_provided=True,
                    ),
                    comments=["Нет прощания."],
                ),
                compliance=ComplianceResult(passed=True),
                summary=SummaryResult(
                    summary="Клиент обратился по вопросу кредита.",
                    action_items=["Подготовить условия кредита."],
                ),
                metadata={"agent_mode": "fake"},
            )

    service = AnalysisService(
        transcriber=FakeTranscriber(),
        diarizer=FakeDiarizer(),
        supervisor=FakeSupervisor(),
    )

    result = await service.analyze_audio_path(audio_path, metadata={"request_id": "unit-test"})

    assert calls == ["transcribe:call.wav", "diarize:UNKNOWN", "supervisor:Клиент"]
    assert result.transcript == diarized_segments
    assert result.classification.topic == "кредиты"
    assert result.quality_score.total == 80
    assert result.compliance.passed is True
    assert result.summary == "Клиент обратился по вопросу кредита."
    assert result.action_items == ["Подготовить условия кредита."]
    assert result.metadata == {"request_id": "unit-test", "agent_mode": "fake"}
