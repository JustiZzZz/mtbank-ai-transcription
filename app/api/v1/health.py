"""Проверка, что сервис жив."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """Самый простой ответ для healthcheck."""
    return {"status": "ok"}
