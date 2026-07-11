"""Unit tests for WER/CER utility helpers."""

import json
from pathlib import Path

import pytest

from scripts.compute_wer import (
    WerResult,
    compute_metrics,
    format_markdown,
    load_cases,
    normalize_text,
)


def test_normalize_text_prepares_russian_asr_text() -> None:
    assert normalize_text("Ёж, БАНК!  Привет.") == "еж банк привет"


def test_compute_metrics_returns_zero_for_equivalent_text() -> None:
    wer, cer = compute_metrics("Добрый день, МТБанк", "добрый день мтбанк")

    assert wer == 0
    assert cer == 0


def test_load_cases_reads_manifest_and_resolves_audio(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "call.wav").write_bytes(b"audio")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file": "call.wav",
                        "title": "Test call",
                        "duration_seconds": 1.5,
                        "transcript": [{"text": "Добрый день"}, {"text": "Хочу кредит"}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_cases(manifest, [audio_dir])

    assert len(cases) == 1
    assert cases[0].audio_path == audio_dir / "call.wav"
    assert cases[0].reference_text == "Добрый день Хочу кредит"


def test_load_cases_reports_missing_audio(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file": "missing.wav",
                        "duration_seconds": 1,
                        "transcript": [{"text": "Добрый день"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing.wav"):
        load_cases(manifest, [tmp_path / "audio"])


def test_format_markdown_contains_table_and_summary() -> None:
    markdown = format_markdown(
        [
            WerResult(
                file="call.wav",
                title="Test",
                duration_seconds=2.0,
                wer=0.25,
                cer=0.1,
                latency_seconds=1.2,
                reference_words=4,
                hypothesis_words=4,
                hypothesis_text="Добрый день",
            )
        ],
        model="medium",
        device="cpu",
    )

    assert "Model: `medium`" in markdown
    assert "| call.wav | 2.00 | 0.250 | 0.100 | 1.20 | 4 | 4 |" in markdown
    assert "Mean WER: 0.250" in markdown
