from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Sequence

from gaffers_guide.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    VALID_PROFILES,
    resolve_profile,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaffers-guide",
        description=(
            "gaffers-guide: football tracking pipeline.\n"
            "Use --quality-profile to control the speed/quality tradeoff."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the tracking pipeline on a video file.",
    )
    run_parser.add_argument(
        "--video", type=Path, required=True,
        help="Path to input video file."
    )
    run_parser.add_argument(
        "--output", type=Path, required=True,
        help="Directory to write output files."
    )
    run_parser.add_argument(
        "--quality-profile",
        choices=VALID_PROFILES,
        default=DEFAULT_PROFILE,
        dest="quality_profile",
        metavar="PROFILE",
        help=(
            f"Quality/speed profile. One of: {', '.join(VALID_PROFILES)}. "
            f"Default: {DEFAULT_PROFILE}."
        ),
    )
    run_parser.add_argument(
        "--precision",
        choices=("fast", "high_res", "sahi"),
        default=None,
        help=argparse.SUPPRESS,
    )

    profiles_parser = subparsers.add_parser(
        "profiles",
        help="Inspect available quality profiles.",
    )
    profiles_sub = profiles_parser.add_subparsers(
        dest="profiles_command", required=True
    )
    profiles_sub.add_parser(
        "list",
        help="List all quality profiles with descriptions and key parameters.",
    )

    return parser


def _handle_run(
    video: Path,
    quality_profile: str,
    output: Path,
    precision: str | None,
) -> int:
    if precision is not None:
        logging.warning(
            "--precision is deprecated. Use --quality-profile %s instead.", precision
        )
        quality_profile = precision

    profile = resolve_profile(quality_profile)

    logging.info("=== gaffers-guide run ===")
    logging.info("video           = %s", video)
    logging.info("output          = %s", output)
    logging.info("quality_profile = %s", profile.name)
    logging.info("description     = %s", profile.description)
    logging.info("--- resolved parameters ---")
    logging.info("sahi_enabled         = %s", profile.sahi_enabled)
    logging.info("imgsz                = %s", profile.imgsz)
    logging.info("confidence_threshold = %s", profile.confidence_threshold)
    logging.info("slice_width          = %s", profile.slice_width)
    logging.info("slice_height         = %s", profile.slice_height)
    logging.info("slice_overlap_ratio  = %s", profile.slice_overlap_ratio)
    logging.info("=========================")

    return 0


def _handle_profiles_list() -> int:
    print(f"\nAvailable quality profiles (default: {DEFAULT_PROFILE})\n")
    print(f"{'Profile':<12} {'SAHI':^6} {'imgsz':^6} {'Conf':^6}  Description")
    print("-" * 72)
    for name, p in PROFILES.items():
        marker = " *" if name == DEFAULT_PROFILE else "  "
        print(
            f"{name:<12}{marker} {str(p.sahi_enabled):^6} {p.imgsz:^6} "
            f"{p.confidence_threshold:^6}  {p.description}"
        )
    print("\n* = default profile\n")
    return 0

def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _handle_run(
            video=args.video,
            quality_profile=args.quality_profile,
            output=args.output,
            precision=args.precision,
        )

    if args.command == "profiles" and args.profiles_command == "list":
        return _handle_profiles_list()

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
