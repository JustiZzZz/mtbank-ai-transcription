"""Настройки приложения из переменных окружения."""

from functools import lru_cache
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Один небольшой объект настроек для API и OpenWebUI Pipeline."""

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
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    whisper_model: str = "medium"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"
    whisper_cpu_threads: int = 0
    whisper_num_workers: int = 1
    whisper_beam_size: int = 1
    whisper_batch_size: int = 8
    whisper_language: str = "ru"
    whisper_vad_filter: bool = True
    whisper_word_timestamps: bool = False
    whisper_condition_on_previous_text: bool = False
    whisper_download_root: str | None = None
    whisper_local_files_only: bool = False
    preload_asr_model: bool = False

    max_audio_mb: int = 100
    max_audio_seconds: int = 600
    allowed_audio_extensions: Annotated[list[str], NoDecode] = [
        ".wav",
        ".mp3",
        ".ogg",
        ".m4a",
        ".flac",
        ".mp4",
    ]
    audio_storage_backend: str = "local"
    audio_storage_dir: str = "var/audio"
    audio_normalized_sample_rate: int = 16000
    audio_normalized_channels: int = 1

    diarization_backend: str = "fallback"
    hf_token: str | None = None

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_model: str | None = None
    llm_enabled: bool = False
    llm_temperature: float = 0
    llm_timeout_seconds: float = 60
    llm_max_output_tokens: int = 1200
    llm_enable_thinking: bool = False
    llm_json_mode: bool = True
    llm_validation_retries: int = 1

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Позволяет писать CORS_ORIGINS через запятую в .env."""
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @field_validator("allowed_audio_extensions", mode="before")
    @classmethod
    def parse_audio_extensions(cls, value: str | list[str]) -> list[str]:
        """Позволяет писать ALLOWED_AUDIO_EXTENSIONS через запятую в .env."""
        if isinstance(value, list):
            raw_extensions = value
        else:
            raw_extensions = value.split(",") if value else []
        return [
            extension.strip().lower()
            if extension.strip().startswith(".")
            else f".{extension.strip().lower()}"
            for extension in raw_extensions
            if extension.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Читаем .env один раз, дальше переиспользуем готовые настройки."""
    return Settings()
