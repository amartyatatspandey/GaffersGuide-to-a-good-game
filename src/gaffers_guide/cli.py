from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal, Sequence


PrecisionMode = Literal["fast", "high_res", "sahi"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaffers-guide",
        description="gaffers-guide command line interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run pipeline (stub)")
    run_parser.add_argument("--video", type=Path, required=True, help="Input video path")
    run_parser.add_argument(
        "--precision",
        choices=("fast", "high_res", "sahi"),
        default="fast",
        help="Precision mode",
    )
    run_parser.add_argument("--output", type=Path, required=True, help="Output directory")

    return parser


def _handle_run(video: Path, precision: PrecisionMode, output: Path) -> int:
    logging.info("CLI stub run command invoked")
    logging.info("video=%s", video)
    logging.info("precision=%s", precision)
    logging.info("output=%s", output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _handle_run(
            video=args.video,
            precision=args.precision,
            output=args.output,
        )

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
