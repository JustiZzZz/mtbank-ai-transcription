"""Compute WER/CER for generated MTBank test audio."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jiwer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.asr.transcriber import FasterWhisperTranscriber  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.schemas import TranscriptSegment  # noqa: E402

DEFAULT_MANIFEST = ROOT / "test_data" / "manifest.json"
DEFAULT_OUTPUT_MD = ROOT / "test_data" / "wer_results.md"
DEFAULT_OUTPUT_JSON = ROOT / "test_data" / "wer_results.json"


@dataclass(frozen=True, slots=True)
class WerCase:
    """One reference audio file and its expected text."""

    audio_path: Path
    reference_text: str
    duration_seconds: float
    title: str


@dataclass(frozen=True, slots=True)
class WerResult:
    """Computed ASR quality metrics for one file."""

    file: str
    title: str
    duration_seconds: float
    wer: float
    cer: float
    latency_seconds: float
    reference_words: int
    hypothesis_words: int
    hypothesis_text: str


def normalize_text(text: str) -> str:
    """Normalize Russian ASR text before WER/CER calculation."""
    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.SubstituteRegexes({r"ё": "е"}),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    return transformation(text)


def reference_text_from_item(item: dict[str, Any]) -> str:
    """Join reference transcript turns from a manifest item."""
    transcript = item.get("transcript")
    if not isinstance(transcript, list):
        msg = f"Manifest item {item.get('file', '<unknown>')} has no transcript list."
        raise ValueError(msg)
    texts = [str(turn.get("text", "")).strip() for turn in transcript if isinstance(turn, dict)]
    return " ".join(text for text in texts if text)


def resolve_audio_path(filename: str, audio_dirs: list[Path]) -> Path:
    """Find an audio file in the first existing input directory."""
    for audio_dir in audio_dirs:
        candidate = audio_dir / filename
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in audio_dirs)
    msg = f"Audio file {filename} not found. Searched: {searched}"
    raise FileNotFoundError(msg)


def load_cases(manifest_path: Path, audio_dirs: list[Path]) -> list[WerCase]:
    """Load benchmark cases from generated test_data/manifest.json."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("files")
    if not isinstance(items, list):
        msg = f"Manifest {manifest_path} has no files list."
        raise ValueError(msg)

    cases: list[WerCase] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        filename = str(item["file"])
        cases.append(
            WerCase(
                audio_path=resolve_audio_path(filename, audio_dirs),
                reference_text=reference_text_from_item(item),
                duration_seconds=float(item["duration_seconds"]),
                title=str(item.get("title") or filename),
            )
        )
    return cases


def compute_metrics(reference_text: str, hypothesis_text: str) -> tuple[float, float]:
    """Return normalized WER and CER values."""
    reference = normalize_text(reference_text)
    hypothesis = normalize_text(hypothesis_text)
    return jiwer.wer(reference, hypothesis), jiwer.cer(reference, hypothesis)


def join_segments(segments: list[TranscriptSegment]) -> str:
    """Join ASR segments into plain text for metrics."""
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


async def transcribe_case(
    transcriber: FasterWhisperTranscriber,
    case: WerCase,
) -> WerResult:
    """Run ASR for one case and calculate quality metrics."""
    started_at = time.perf_counter()
    segments = await transcriber.transcribe(case.audio_path)
    latency = time.perf_counter() - started_at
    hypothesis = join_segments(list(segments))
    wer, cer = compute_metrics(case.reference_text, hypothesis)
    return WerResult(
        file=case.audio_path.name,
        title=case.title,
        duration_seconds=case.duration_seconds,
        wer=wer,
        cer=cer,
        latency_seconds=latency,
        reference_words=len(normalize_text(case.reference_text).split()),
        hypothesis_words=len(normalize_text(hypothesis).split()),
        hypothesis_text=hypothesis,
    )


def format_markdown(results: list[WerResult], *, model: str, device: str) -> str:
    """Build the markdown table expected for final README inclusion."""
    summary = summarize_results(results)
    lines = [
        "# WER/CER Results",
        "",
        f"Generated at: `{datetime.now(UTC).isoformat()}`",
        f"Model: `{model}`",
        f"Device: `{device}`",
        f"Files: {summary['files']}",
        f"Total audio duration: {summary['total_duration_seconds']:.2f}s",
        f"Total ASR latency: {summary['total_latency_seconds']:.2f}s",
        f"Realtime factor: {summary['realtime_factor']:.2f}x",
        "",
        "| File | Duration, s | WER | CER | ASR latency, s | Ref words | Hyp words |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.file} | {result.duration_seconds:.2f} | {result.wer:.3f} | "
            f"{result.cer:.3f} | {result.latency_seconds:.2f} | "
            f"{result.reference_words} | {result.hypothesis_words} |"
        )
    lines.extend(
        [
            "",
            f"Mean WER: {summary['mean_wer']:.3f}",
            f"Mean CER: {summary['mean_cer']:.3f}",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_results(results: list[WerResult]) -> dict[str, float | int]:
    """Compute aggregate ASR quality and speed numbers."""
    total_duration = sum(result.duration_seconds for result in results)
    total_latency = sum(result.latency_seconds for result in results)
    return {
        "files": len(results),
        "total_duration_seconds": total_duration,
        "total_latency_seconds": total_latency,
        "realtime_factor": total_latency / total_duration if total_duration else 0,
        "mean_wer": sum(result.wer for result in results) / len(results),
        "mean_cer": sum(result.cer for result in results) / len(results),
    }


def write_json(
    results: list[WerResult],
    output_path: Path,
    *,
    model: str,
    device: str,
) -> None:
    """Write machine-readable WER/CER output."""
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "device": device,
        "summary": summarize_results(results),
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--audio-dir",
        action="append",
        type=Path,
        default=[ROOT / "test_data", ROOT / "test_data_audio"],
        help="Directory containing audio files. Can be passed multiple times.",
    )
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of files.")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    settings = get_settings()
    cases = load_cases(args.manifest, args.audio_dir)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No WER cases found.", file=sys.stderr)
        return 1

    transcriber = FasterWhisperTranscriber(settings)
    results: list[WerResult] = []
    for case in cases:
        print(f"Transcribing {case.audio_path.name}...")
        results.append(await transcribe_case(transcriber, case))

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        format_markdown(results, model=settings.whisper_model, device=settings.whisper_device),
        encoding="utf-8",
    )
    write_json(
        results,
        args.output_json,
        model=settings.whisper_model,
        device=settings.whisper_device,
    )
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
