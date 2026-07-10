"""Unit-тесты настроек приложения."""

from app.config import Settings


def test_settings_can_ignore_local_env_file(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "WHISPER_BATCH_SIZE=99\nAUDIO_STORAGE_BACKEND=s3\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WHISPER_BATCH_SIZE", raising=False)
    monkeypatch.delenv("AUDIO_STORAGE_BACKEND", raising=False)

    settings = Settings(_env_file=None)

    assert settings.whisper_batch_size == 8
    assert settings.audio_storage_backend == "local"
