#!/usr/bin/env python3
"""
Verify SoccerNet sn-calibration layout for local CV homography auto-generation.

Run from repo with PYTHONPATH including backend (e.g. ``cd backend && python scripts/verify_sn_calibration.py``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.pipeline_paths import (  # noqa: E402
    SN_CALIBRATION_REPO_URL,
    sn_calibration_resources_dir,
    sn_calibration_root_dir,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check sn-calibration paths for homography pipeline."
    )
    parser.add_argument(
        "--try-import",
        action="store_true",
        help="Import DynamicPitchCalibrator (fails if repo incomplete).",
    )
    args = parser.parse_args()

    root = sn_calibration_root_dir()
    res = sn_calibration_resources_dir()

    if not root.is_dir():
        logger.error("Missing checkout: %s", root)
        logger.error("Clone %s into that path.", SN_CALIBRATION_REPO_URL)
        return 2

    logger.info("sn-calibration root OK: %s", root)

    if not res.is_dir():
        logger.error("Missing resources directory: %s", res)
        logger.error("Follow upstream sn-calibration README to populate resources/.")
        return 2

    logger.info("resources/ OK: %s", res)

    if args.try_import:
        try:
            from scripts.dynamic_homography import DynamicPitchCalibrator

            _ = DynamicPitchCalibrator(res)
            logger.info("DynamicPitchCalibrator import and construction OK.")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Import or construction failed: %s", exc)
            return 3

    logger.info("Homography auto-generation prerequisites look satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
