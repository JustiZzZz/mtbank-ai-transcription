"""REST API для анализа аудио."""

from __future__ import annotations

from json import JSONDecodeError
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, HttpUrl, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.asr.transcriber import ASRRuntimeError
from app.runtime import AnalysisRuntime, AudioInputError, get_analysis_runtime
from app.schemas import AnalysisResult

router = APIRouter()


class AnalyzeUrlRequest(BaseModel):
    """JSON-вход для анализа аудио по URL."""

    url: HttpUrl


ANALYZE_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "WAV, MP3, OGG, M4A or FLAC audio file.",
                        }
                    },
                    "required": ["file"],
                }
            },
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "format": "uri",
                            "description": "Public http/https audio URL.",
                        }
                    },
                    "required": ["url"],
                }
            },
        },
    }
}


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    openapi_extra=ANALYZE_OPENAPI_EXTRA,
)
async def analyze_audio(
    request: Request,
    runtime: Annotated[AnalysisRuntime, Depends(get_analysis_runtime)],
) -> AnalysisResult:
    """Принимает multipart file или JSON {"url": "..."} и возвращает анализ."""
    try:
        if "multipart/form-data" in request.headers.get("content-type", ""):
            upload = await _extract_upload(request)
            return await runtime.analyze_upload(upload)
        payload = await _extract_json_url(request)
        return await runtime.analyze_url(str(payload.url))
    except AudioInputError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ASRRuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


async def _extract_upload(request: Request) -> StarletteUploadFile:
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, StarletteUploadFile):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide multipart file or JSON url.",
        )
    return upload


async def _extract_json_url(request: Request) -> AnalyzeUrlRequest:
    try:
        body = await request.json()
        return AnalyzeUrlRequest.model_validate(body)
    except (JSONDecodeError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide multipart file or JSON url.",
        ) from None
