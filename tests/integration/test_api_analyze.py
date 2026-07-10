"""Интеграционные тесты /analyze без реального ASR."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.analyze import get_analysis_runtime
from app.asr.transcriber import ASRRuntimeError
from app.main import create_app
from app.schemas import (
    AnalysisResult,
    Classification,
    ComplianceResult,
    QualityChecklist,
    QualityScore,
    TranscriptSegment,
)


def sample_result(source: str) -> AnalysisResult:
    return AnalysisResult(
        transcript=[
            TranscriptSegment(
                speaker="Оператор",
                start=0.0,
                end=1.0,
                text="Добрый день, МТБанк",
            )
        ],
        classification=Classification(
            topic="кредиты",
            priority="medium",
            confidence=0.85,
            confidence_label="high",
            rationale="Клиент спрашивает про кредит.",
        ),
        quality_score=QualityScore(
            total=80,
            checklist=QualityChecklist(greeting=True, solution_provided=True),
            comments=[],
        ),
        compliance=ComplianceResult(passed=True),
        summary="Клиент обратился по вопросу кредита.",
        action_items=["Отправить условия кредита."],
        metadata={"source": source},
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.upload_names: list[str] = []
        self.urls: list[str] = []

    async def analyze_upload(self, upload) -> AnalysisResult:
        self.upload_names.append(upload.filename)
        return sample_result("upload")

    async def analyze_url(self, url: str) -> AnalysisResult:
        self.urls.append(url)
        return sample_result("url")


class FailingRuntime:
    async def analyze_upload(self, upload) -> AnalysisResult:
        raise ASRRuntimeError("ASR GPU runtime is not available.")

    async def analyze_url(self, url: str) -> AnalysisResult:
        raise ASRRuntimeError("ASR GPU runtime is not available.")


@pytest.fixture
async def analyze_client() -> tuple[AsyncClient, FakeRuntime]:
    runtime = FakeRuntime()
    app = create_app()
    app.dependency_overrides[get_analysis_runtime] = lambda: runtime
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, runtime


async def test_root_analyze_accepts_multipart_file(analyze_client) -> None:
    client, runtime = analyze_client

    response = await client.post(
        "/analyze",
        files={"file": ("call.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["topic"] == "кредиты"
    assert payload["transcript"][0]["speaker"] == "Оператор"
    assert runtime.upload_names == ["call.wav"]


async def test_analyze_accepts_json_url(analyze_client) -> None:
    client, runtime = analyze_client

    response = await client.post(
        "/analyze",
        json={"url": "https://example.com/call.mp3"},
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["source"] == "url"
    assert runtime.urls == ["https://example.com/call.mp3"]


async def test_v1_analyze_is_not_registered(analyze_client) -> None:
    client, _runtime = analyze_client

    response = await client.post(
        "/api/v1/analyze",
        json={"url": "https://example.com/call.mp3"},
    )

    assert response.status_code == 404


async def test_analyze_rejects_missing_input(analyze_client) -> None:
    client, _runtime = analyze_client

    response = await client.post("/analyze", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide multipart file or JSON url."


async def test_analyze_openapi_exposes_file_and_json_inputs(analyze_client) -> None:
    client, _runtime = analyze_client

    response = await client.get("/openapi.json")

    assert response.status_code == 200
    content = response.json()["paths"]["/analyze"]["post"]["requestBody"]["content"]
    assert content["multipart/form-data"]["schema"]["properties"]["file"]["format"] == "binary"
    assert content["application/json"]["schema"]["properties"]["url"]["format"] == "uri"


async def test_analyze_returns_503_for_asr_runtime_error() -> None:
    app = create_app()
    app.dependency_overrides[get_analysis_runtime] = lambda: FailingRuntime()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/analyze",
            files={"file": ("call.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "ASR GPU runtime is not available."
