"""Measure /analyze wall-clock latency for a media file."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/analyze")
    parser.add_argument("--file", default="test_data/benchmark_5min_call.mp4")
    parser.add_argument("--limit-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    media_path = Path(args.file)
    if not media_path.exists():
        print(f"File not found: {media_path}", file=sys.stderr)
        return 1

    started_at = time.perf_counter()
    with media_path.open("rb") as handle:
        response = httpx.post(
            args.url,
            files={"file": (media_path.name, handle, "video/mp4")},
            timeout=args.timeout_seconds,
        )
    elapsed = time.perf_counter() - started_at

    print(f"status={response.status_code}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(f"limit_seconds={args.limit_seconds:.2f}")
    if response.status_code != 200:
        print(response.text[:2000], file=sys.stderr)
        return 1
    if elapsed > args.limit_seconds:
        print("FAIL: latency limit exceeded")
        return 2
    print("PASS: latency limit met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
