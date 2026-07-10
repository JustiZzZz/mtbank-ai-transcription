"""Минимальный OpenAI-compatible client для Qwen/OpenAI/OpenRouter."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import Settings, get_settings


class LLMClientError(RuntimeError):
    """Ошибка вызова или разбора ответа LLM."""


SECRET_RE = re.compile(r"sk-[A-Za-z0-9._-]+")
ERROR_SNIPPET_LIMIT = 300


class OpenAICompatibleClient:
    """Вызывает `/chat/completions` у OpenAI-compatible провайдера."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.transport = transport

    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Вернуть JSON-объект из message.content."""
        if not self.settings.openai_api_key or not self.settings.openai_model:
            msg = "LLM API key or model is not configured."
            raise LLMClientError(msg)

        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_output_tokens,
            "enable_thinking": self.settings.llm_enable_thinking,
        }
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        url = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.llm_timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = self._safe_response_detail(exc.response)
            raise LLMClientError(
                f"LLM request failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"LLM request failed with {exc.__class__.__name__}.") from exc
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM response body is not JSON.") from exc

        content = self._extract_content(data)
        return self._parse_json_object(content)

    def _safe_response_detail(self, response: httpx.Response) -> str:
        text = response.text.strip().replace("\n", " ")
        text = SECRET_RE.sub("sk-***", text)
        if len(text) > ERROR_SNIPPET_LIMIT:
            return f"{text[:ERROR_SNIPPET_LIMIT]}..."
        return text or "<empty response>"

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM response has no message content.") from exc
        if not isinstance(content, str):
            raise LLMClientError("LLM message content is not a string.")
        return content

    def _parse_json_object(self, content: str) -> dict[str, Any]:
        text = self._strip_code_fence(content.strip())
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise LLMClientError("LLM response is not a JSON object.")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM response is invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise LLMClientError("LLM response JSON must be an object.")
        return parsed

    def _strip_code_fence(self, text: str) -> str:
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
