"""Unit-тесты OpenAI-compatible LLM слоя без реальных сетевых вызовов."""

import json

import httpx

from app.agents.llm import LLMClassifierAgent
from app.config import Settings
from app.llm.client import LLMClientError, OpenAICompatibleClient
from app.orchestration.supervisor import FallbackSupervisor, LLMSupervisor, build_supervisor
from app.schemas import TranscriptSegment


def make_transcript(text: str) -> list[TranscriptSegment]:
    return [TranscriptSegment(speaker="Клиент", start=0.0, end=2.0, text=text)]


def make_client(handler) -> OpenAICompatibleClient:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleClient(
        Settings(
            _env_file=None,
            openai_base_url="https://example.test/compatible-mode/v1",
            openai_api_key="sk-test",
            openai_model="qwen3.7-plus",
            llm_temperature=0,
            llm_timeout_seconds=5,
            llm_max_output_tokens=400,
            llm_enable_thinking=False,
            llm_json_mode=True,
            llm_validation_retries=1,
        ),
        transport=transport,
    )


async def test_openai_compatible_client_posts_chat_completion() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            {
                "url": str(request.url),
                "authorization": request.headers["authorization"],
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"topic":"кредиты","priority":"medium"}',
                        }
                    }
                ]
            },
        )

    client = make_client(handler)

    result = await client.complete_json(
        system_prompt="system",
        user_prompt="user",
    )

    assert result == {"topic": "кредиты", "priority": "medium"}
    assert requests[0]["url"] == "https://example.test/compatible-mode/v1/chat/completions"
    assert requests[0]["authorization"] == "Bearer sk-test"
    assert requests[0]["body"]["model"] == "qwen3.7-plus"
    assert requests[0]["body"]["temperature"] == 0
    assert requests[0]["body"]["enable_thinking"] is False
    assert requests[0]["body"]["response_format"] == {"type": "json_object"}


async def test_openai_compatible_client_reports_sanitized_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"message": "invalid api key sk-ws-secret-value"},
        )

    client = make_client(handler)

    try:
        await client.complete_json(system_prompt="system", user_prompt="user")
    except LLMClientError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected LLMClientError")

    assert "HTTP 401" in message
    assert "sk-ws-secret-value" not in message
    assert "sk-***" in message


async def test_llm_classifier_uses_valid_model_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "topic": "жалобы",
                                    "priority": "high",
                                    "confidence": 0.91,
                                    "confidence_label": "high",
                                    "rationale": "Клиент явно просит зарегистрировать жалобу.",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    }
                ]
            },
        )

    result = await LLMClassifierAgent(make_client(handler)).analyze(
        make_transcript("Хочу оставить жалобу")
    )

    assert result.topic == "жалобы"
    assert result.priority == "high"
    assert result.confidence == 0.91


async def test_llm_classifier_falls_back_on_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "не json"}}]},
        )

    result = await LLMClassifierAgent(make_client(handler)).analyze(
        make_transcript("Хочу узнать условия кредита")
    )

    assert result.topic == "кредиты"
    assert "Fallback" in result.rationale


async def test_llm_classifier_repairs_validation_error_once() -> None:
    responses = [
        {
            "topic": "complaint",
            "priority": "urgent",
            "confidence": 1.2,
            "rationale": "Wrong enum values.",
        },
        {
            "topic": "жалобы",
            "priority": "high",
            "confidence": 0.9,
            "confidence_label": "high",
            "rationale": "Исправлены enum и confidence.",
        },
    ]
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(responses.pop(0), ensure_ascii=False)}}
                ]
            },
        )

    agent = LLMClassifierAgent(make_client(handler))

    result, mode = await agent.analyze_with_mode(make_transcript("Хочу оставить жалобу"))

    assert result.topic == "жалобы"
    assert mode == "llm"
    assert len(requests) == 2
    assert "Pydantic validation" in requests[1]["messages"][1]["content"]


def test_build_supervisor_uses_fallback_without_key() -> None:
    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        openai_api_key=None,
        openai_model="qwen3.7-plus",
    )

    assert isinstance(build_supervisor(settings), FallbackSupervisor)


def test_build_supervisor_uses_llm_when_configured() -> None:
    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        openai_base_url="https://llm-pnv5fw3uqyar1tdu.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        openai_api_key="sk-test",
        openai_model="qwen3.7-plus",
    )

    supervisor = build_supervisor(settings)

    assert isinstance(supervisor, LLMSupervisor)
    assert supervisor.client.settings.openai_model == "qwen3.7-plus"


async def test_llm_supervisor_reports_per_agent_modes() -> None:
    response_by_call = [
        {
            "topic": "жалобы",
            "priority": "high",
            "confidence": 0.9,
            "confidence_label": "high",
            "rationale": "Жалоба клиента.",
        },
        {
            "total": 55,
            "checklist": {
                "greeting": False,
                "identification_or_intro": False,
                "need_detection": True,
                "solution_provided": True,
                "objection_handling": True,
                "farewell": False,
            },
            "comments": ["Нет приветствия."],
        },
        {"passed": True, "issues": []},
        {"summary": "Клиент сообщил о спорной операции.", "action_items": []},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(response_by_call.pop(0), ensure_ascii=False)
                        }
                    }
                ]
            },
        )

    supervisor = LLMSupervisor(make_client(handler))

    result = await supervisor.analyze(make_transcript("Хочу оставить жалобу"))

    assert result.metadata["agents_mode"] == "llm"
    assert result.metadata["llm_classifier_mode"] == "llm"
    assert result.metadata["llm_quality_mode"] == "llm"
    assert result.metadata["llm_compliance_mode"] == "llm"
    assert result.metadata["llm_summarizer_mode"] == "llm"
