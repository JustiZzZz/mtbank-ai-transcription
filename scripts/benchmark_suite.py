"""Run end-to-end /analyze latency benchmark for all test media files."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "test_data" / "manifest.json"
DEFAULT_BENCHMARK_META = ROOT / "test_data" / "benchmark_5min_call.json"
DEFAULT_OUTPUT_MD = ROOT / "test_data" / "benchmark_results.md"
DEFAULT_OUTPUT_JSON = ROOT / "test_data" / "benchmark_results.json"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One media file to send to /analyze."""

    file: Path
    title: str
    duration_seconds: float
    expected_limit_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """End-to-end API latency and selected response fields."""

    file: str
    title: str
    duration_seconds: float
    latency_seconds: float
    realtime_factor: float
    status_code: int
    passed_latency_limit: bool | None
    expected_limit_seconds: float | None
    topic: str | None
    priority: str | None
    quality_score: int | None
    compliance_passed: bool | None
    error: str | None = None


def load_cases(
    *,
    manifest_path: Path,
    benchmark_meta_path: Path,
    audio_dir: Path,
    include_five_minute: bool,
) -> list[BenchmarkCase]:
    """Load short manifest cases and optional 5-minute latency case."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [
        BenchmarkCase(
            file=audio_dir / str(item["file"]),
            title=str(item.get("title") or item["file"]),
            duration_seconds=float(item["duration_seconds"]),
        )
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    ]
    if include_five_minute:
        benchmark_meta = json.loads(benchmark_meta_path.read_text(encoding="utf-8"))
        cases.append(
            BenchmarkCase(
                file=audio_dir / str(benchmark_meta["file"]),
                title=str(benchmark_meta.get("purpose") or benchmark_meta["file"]),
                duration_seconds=float(benchmark_meta["duration_seconds"]),
                expected_limit_seconds=float(benchmark_meta.get("expected_limit_seconds", 60)),
            )
        )
    for case in cases:
        if not case.file.exists():
            msg = f"Benchmark file not found: {case.file}"
            raise FileNotFoundError(msg)
    return cases


def content_type_for(path: Path) -> str:
    """Return a reasonable multipart content type for an audio/video file."""
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def response_field(payload: dict[str, Any], *path: str) -> Any:
    """Safely read a nested response field."""
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def run_case(client: httpx.Client, case: BenchmarkCase, url: str) -> BenchmarkResult:
    """Post one file to /analyze and measure wall-clock latency."""
    started_at = time.perf_counter()
    try:
        with case.file.open("rb") as handle:
            response = client.post(
                url,
                files={"file": (case.file.name, handle, content_type_for(case.file))},
            )
        latency = time.perf_counter() - started_at
        payload = response.json() if response.headers.get("content-type", "").startswith(
            "application/json"
        ) else {}
        passed_limit = (
            None
            if case.expected_limit_seconds is None
            else latency <= case.expected_limit_seconds
        )
        return BenchmarkResult(
            file=case.file.name,
            title=case.title,
            duration_seconds=case.duration_seconds,
            latency_seconds=latency,
            realtime_factor=latency / case.duration_seconds if case.duration_seconds else 0,
            status_code=response.status_code,
            passed_latency_limit=passed_limit,
            expected_limit_seconds=case.expected_limit_seconds,
            topic=response_field(payload, "classification", "topic"),
            priority=response_field(payload, "classification", "priority"),
            quality_score=response_field(payload, "quality_score", "total"),
            compliance_passed=response_field(payload, "compliance", "passed"),
            error=None if response.status_code == 200 else response.text[:1000],
        )
    except Exception as exc:
        latency = time.perf_counter() - started_at
        return BenchmarkResult(
            file=case.file.name,
            title=case.title,
            duration_seconds=case.duration_seconds,
            latency_seconds=latency,
            realtime_factor=latency / case.duration_seconds if case.duration_seconds else 0,
            status_code=0,
            passed_latency_limit=False if case.expected_limit_seconds is not None else None,
            expected_limit_seconds=case.expected_limit_seconds,
            topic=None,
            priority=None,
            quality_score=None,
            compliance_passed=None,
            error=str(exc),
        )


def summarize_results(results: list[BenchmarkResult]) -> dict[str, float | int]:
    """Compute aggregate benchmark numbers."""
    total_duration = sum(result.duration_seconds for result in results)
    total_latency = sum(result.latency_seconds for result in results)
    successful = [result for result in results if result.status_code == 200]
    return {
        "files": len(results),
        "successful_files": len(successful),
        "total_duration_seconds": total_duration,
        "total_latency_seconds": total_latency,
        "mean_latency_seconds": total_latency / len(results) if results else 0,
        "overall_realtime_factor": total_latency / total_duration if total_duration else 0,
    }


def format_markdown(results: list[BenchmarkResult], *, url: str) -> str:
    """Build markdown benchmark report."""
    summary = summarize_results(results)
    lines = [
        "# /analyze Benchmark Results",
        "",
        f"Generated at: `{datetime.now(UTC).isoformat()}`",
        f"Endpoint: `{url}`",
        f"Files: {summary['files']}",
        f"Successful files: {summary['successful_files']}",
        f"Total media duration: {summary['total_duration_seconds']:.2f}s",
        f"Total wall-clock latency: {summary['total_latency_seconds']:.2f}s",
        f"Overall realtime factor: {summary['overall_realtime_factor']:.2f}x",
        "",
        "| File | Duration, s | Latency, s | RTF | Status | Limit, s | Topic | "
        "Quality | Compliance |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for result in results:
        limit = (
            ""
            if result.expected_limit_seconds is None
            else f"{result.expected_limit_seconds:.2f}"
        )
        compliance = "" if result.compliance_passed is None else str(result.compliance_passed)
        lines.append(
            f"| {result.file} | {result.duration_seconds:.2f} | "
            f"{result.latency_seconds:.2f} | {result.realtime_factor:.2f} | "
            f"{result.status_code} | {limit} | {result.topic or ''} | "
            f"{result.quality_score if result.quality_score is not None else ''} | "
            f"{compliance} |"
        )
    errors = [result for result in results if result.error]
    if errors:
        lines.extend(["", "## Errors", ""])
        for result in errors:
            lines.append(f"- `{result.file}`: {result.error}")
    return "\n".join(lines) + "\n"


def write_json(results: list[BenchmarkResult], output_path: Path, *, url: str) -> None:
    """Write machine-readable benchmark results."""
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "endpoint": url,
        "summary": summarize_results(results),
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(
    results: list[BenchmarkResult],
    *,
    output_md: Path,
    output_json: Path,
    url: str,
) -> None:
    """Persist current benchmark progress after every case."""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(format_markdown(results, url=url), encoding="utf-8")
    write_json(results, output_json, url=url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/analyze")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-meta", type=Path, default=DEFAULT_BENCHMARK_META)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "test_data")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--no-five-minute", action="store_true")
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove previous benchmark output before starting a new run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clear_output:
        args.output_md.unlink(missing_ok=True)
        args.output_json.unlink(missing_ok=True)
    cases = load_cases(
        manifest_path=args.manifest,
        benchmark_meta_path=args.benchmark_meta,
        audio_dir=args.audio_dir,
        include_five_minute=not args.no_five_minute,
    )
    results: list[BenchmarkResult] = []
    with httpx.Client(timeout=args.timeout_seconds) as client:
        for case in cases:
            print(f"Benchmarking {case.file.name}...", flush=True)
            results.append(run_case(client, case, args.url))
            latest = results[-1]
            print(
                f"  status={latest.status_code} latency={latest.latency_seconds:.2f}s",
                flush=True,
            )
            write_outputs(
                results,
                output_md=args.output_md,
                output_json=args.output_json,
                url=args.url,
            )

    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    return 0 if all(result.status_code == 200 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
