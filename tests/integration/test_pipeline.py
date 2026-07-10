"""Интеграционный тест OpenWebUI pipeline formatting."""

from pathlib import Path

import pipeline as pipeline_module
from app.schemas import (
    AnalysisResult,
    Classification,
    ComplianceResult,
    QualityChecklist,
    QualityScore,
    TranscriptSegment,
)


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

    async def analyze_path(self, path: str | Path) -> AnalysisResult:
        self.paths.append(Path(path))
        return sample_result()


async def test_pipeline_formats_analysis_markdown() -> None:
    runtime = FakeRuntime()
    pipe = pipeline_module.Pipeline(runtime=runtime)

    output = await pipe.pipe({"files": [{"path": "call.wav"}]})

    assert runtime.paths == [Path("call.wav")]
    assert "## Анализ звонка" in output
    assert "**Тема:** кредиты" in output
    assert "| Оператор | 0.0 | 1.0 | Добрый день |" in output
    assert "Отправить условия." in output
