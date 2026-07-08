"""Настройки приложения из переменных окружения."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Один небольшой объект настроек для API и будущего Pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "MTBank AI Transcription"
    environment: str = "dev"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    health_path: str = "/health"

    log_format: str = "json"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    whisper_model: str = "medium"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"
    whisper_beam_size: int = 1
    whisper_language: str = "ru"
    whisper_vad_filter: bool = True

    diarization_backend: str = "fallback"
    hf_token: str | None = None

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_model: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Позволяет писать CORS_ORIGINS через запятую в .env."""
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [origin.strip() for origin in value.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Читаем .env один раз, дальше переиспользуем готовые настройки."""
    return Settings()
