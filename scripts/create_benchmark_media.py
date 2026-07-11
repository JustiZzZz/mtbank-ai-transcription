"""Create a 5-minute benchmark video for /analyze latency checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "test_data"
OUTPUT = TEST_DATA / "benchmark_5min_call.mp4"
DURATION_SECONDS = 300
PHONE_FILTER = (
    "highpass=f=300,"
    "lowpass=f=3400,"
    "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=120"
)
INPUT_FILES = (
    "call01_credit_online.wav",
    "call02_halva_fraud.mp3",
    "call03_transfer_stuck.ogg",
    "call04_compliance_risky.wav",
    "call05_poor_quality_complaint.mp3",
    "call06_unknown_nonsense.flac",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def duration(path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def main() -> None:
    inputs = [TEST_DATA / name for name in INPUT_FILES]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Generate test_data first, missing: {missing_names}")

    input_args: list[str] = []
    for path in inputs:
        input_args.extend(["-i", str(path)])

    audio_inputs = "".join(f"[{index + 1}:a]" for index in range(len(inputs)))
    filter_complex = (
        f"{audio_inputs}concat=n={len(inputs)}:v=0:a=1,"
        f"atrim=0:{DURATION_SECONDS},asetpts=N/SR/TB,{PHONE_FILTER}[a]"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=640x360:r=1:d={DURATION_SECONDS}",
            *input_args,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-t",
            str(DURATION_SECONDS),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(OUTPUT),
        ]
    )
    metadata = {
        "file": OUTPUT.name,
        "duration_seconds": round(duration(OUTPUT), 2),
        "purpose": "5-minute /analyze latency benchmark",
        "expected_limit_seconds": 60,
        "source_files": list(INPUT_FILES),
        "notes": "Black video with phone-bandlimited Russian call-center audio.",
    }
    (TEST_DATA / "benchmark_5min_call.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
