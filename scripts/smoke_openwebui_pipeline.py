"""Run a direct smoke check for the OpenWebUI Pipeline class."""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import Pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=ROOT / "test_data" / "call06_unknown_nonsense.flac",
        help="Local audio file to pass to pipeline.py.",
    )
    parser.add_argument("--min-chars", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.file.exists():
        print(f"Audio file not found: {args.file}", file=sys.stderr)
        return 1

    pipeline = Pipeline()
    asyncio.run(pipeline.on_startup())
    markdown = pipeline.pipe({"files": [{"path": str(args.file)}]})
    print(markdown)

    required_markers = ("## Анализ звонка", "### Транскрипт", "| Спикер |")
    missing = [marker for marker in required_markers if marker not in markdown]
    if missing:
        print(f"Missing expected markdown markers: {missing}", file=sys.stderr)
        return 2
    if len(markdown) < args.min_chars:
        print(f"Pipeline response is unexpectedly short: {len(markdown)} chars", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
