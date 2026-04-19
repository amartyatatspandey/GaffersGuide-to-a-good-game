from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from scripts.pipeline_core.run_e2e_cloud import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLOW_MAX_WIDTH,
    run_e2e_cloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete CV→Math→Rules→RAG→LLM E2E pipeline."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Video filename or path (example: test.mp4).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="test_mp4",
        help="Output artifact prefix (default: test_mp4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="YOLO inference batch size (default: 16).",
    )
    parser.add_argument(
        "--flow-max-width",
        type=int,
        default=DEFAULT_FLOW_MAX_WIDTH,
        help="Max width for downscaled optical flow frames (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="YOLO device: auto|cuda|mps|cpu (default: auto).",
    )
    return parser.parse_args()


def run_e2e(
    video: str | Path,
    *,
    output_prefix: str = "test_mp4",
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    return run_e2e_cloud(
        video,
        output_prefix=output_prefix,
        progress_callback=progress_callback,
    )


def main() -> None:
    args = parse_args()
    run_e2e_cloud(
        args.video,
        output_prefix=args.output_prefix,
        progress_callback=None,
        batch_size=max(1, int(args.batch_size)),
        flow_max_width=max(64, int(args.flow_max_width)),
        device=args.device,
    )


if __name__ == "__main__":
    main()
