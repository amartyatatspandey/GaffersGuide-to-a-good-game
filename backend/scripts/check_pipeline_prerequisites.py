"""CLI: verify local CV pipeline filesystem prerequisites; exit 1 if anything is missing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.pipeline_paths import (  # noqa: E402
    collect_local_cv_pipeline_gaps,
    format_pipeline_prerequisite_errors,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional path to an upload .mp4 to include homography and video file checks.",
    )
    args = parser.parse_args()

    gaps = collect_local_cv_pipeline_gaps(video_path=args.video)
    if gaps:
        print(format_pipeline_prerequisite_errors(gaps))
        if any("[homography]" in g for g in gaps):
            print(
                "Hint: run `python scripts/verify_sn_calibration.py` from backend/ "
                "to check SoccerNet checkout and resources/.",
                file=sys.stderr,
            )
        return 1
    msg = "OK: YOLO weights and tactical_library.json are present."
    if args.video is not None:
        msg += f" Homography OK for {args.video}."
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
