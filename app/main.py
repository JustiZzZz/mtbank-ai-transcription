"""Создание FastAPI-приложения."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

from app.api.v1.analyze import router as analyze_router
from app.api.v1.router import api_router
from app.config import get_settings
from app.logging import configure_logging
from app.runtime import get_analysis_runtime

settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Инициализация тяжелых компонентов приложения."""
    if settings.preload_asr_model:
        logger.info("Preloading ASR model on startup")
        await get_analysis_runtime().preload()
    yield


def create_app() -> FastAPI:
    """Собирает приложение: middleware, healthcheck и роуты."""
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error for %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    @app.get(settings.health_path, tags=["health"])
    async def root_healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    app.include_router(analyze_router, tags=["analyze"])
    return app


app = create_app()
