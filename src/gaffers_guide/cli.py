from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal, Sequence

from gaffers_guide.profiles import ProfileConfig, resolve_profile
from gaffers_guide.runtime.tactical_pipeline import TacticalPipeline


PrecisionMode = Literal["fast", "balanced", "high_res", "sahi"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaffers-guide",
        description="Run Gaffer's Guide tactical analysis pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ───────────────── RUN COMMAND ─────────────────
    run_parser = subparsers.add_parser("run", help="Run full pipeline")

    run_parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video path",
    )

    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory",
    )

    run_parser.add_argument(
        "--quality-profile",
        type=str,
        choices=("fast", "balanced", "high_res", "sahi"),
        default=None,
        help="Quality/speed tradeoff profile",
    )

    run_parser.add_argument(
        "--precision",
        choices=("fast", "balanced", "high_res", "sahi"),
        default=None,
        help="(Deprecated) Use --quality-profile instead",
    )

    # ───────────────── PROFILES COMMAND ─────────────────
    profiles_parser = subparsers.add_parser("profiles", help="Profile utilities")
    profiles_sub = profiles_parser.add_subparsers(dest="profiles_cmd", required=True)

    profiles_sub.add_parser("list", help="List available profiles")

    return parser


def _handle_run(
    video: Path,
    quality_profile: str | None,
    precision: PrecisionMode | None,
    output: Path,
) -> int:

    # ── Resolve profile ────────────────────────────
    if quality_profile is not None:
        profile_name = quality_profile
    elif precision is not None:
        logging.warning("--precision is deprecated, use --quality-profile instead")
        profile_name = precision
    else:
        profile_name = "balanced"

    profile: ProfileConfig = resolve_profile(profile_name)

    # ── Logging ────────────────────────────────────
    logging.info("Running pipeline...")
    logging.info("video=%s", video)
    logging.info("output=%s", output)
    logging.info("Active profile — %s", profile)

    pipeline = TacticalPipeline(profile=profile)
    pipeline.run(video=video, output=output)

    logging.info("Pipeline completed successfully.")

    return 0


def _handle_profiles_list() -> int:
    print("Available profiles:")
    print(" - fast       (speed optimized)")
    print(" - balanced   (default)")
    print(" - high_res   (higher quality)")
    print(" - sahi       (maximum quality, slowest)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _handle_run(
            video=args.video,
            quality_profile=args.quality_profile,
            precision=args.precision,
            output=args.output,
        )

    if args.command == "profiles":
        if args.profiles_cmd == "list":
            return _handle_profiles_list()

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())