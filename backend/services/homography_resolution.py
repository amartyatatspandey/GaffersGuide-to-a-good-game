"""Resolve and auto-generate TacticalRadar homography JSON (services-owned policy)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_EVERY = 30


def ensure_homography_json_for_video(
    video_path: Path,
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    use_advanced_calibration: bool = True,
) -> Path:
    """
    Resolve homography JSON for ``video_path`` (env override or per-stem under ``output/``).

    If missing and calibration weights are available, runs the pitch calibrator synchronously.
    """
    from services.paths.calibration import sn_calibration_resources_dir
    from services.paths.homography import (
        format_homography_missing_error,
        resolve_tracking_homography_json_path,
        validate_homography_json_file,
    )

    vp = video_path.expanduser().resolve()
    expected = resolve_tracking_homography_json_path(vp)
    if expected.is_file():
        bad = validate_homography_json_file(expected)
        if bad:
            raise ValueError(bad)
        return expected

    weights_dir = sn_calibration_resources_dir()
    if weights_dir.is_dir():
        logger.info(
            "Homography JSON not found at %s; running calibrator (resources at %s).",
            expected,
            weights_dir,
        )
        # Local import: ``run_calibrator_on_video`` re-exports this function from here.
        from scripts.pipeline_core.run_calibrator_on_video import run

        run(
            video_path=vp,
            weights_dir=weights_dir,
            sample_every=sample_every,
            output_json=expected,
            use_advanced_calibration=use_advanced_calibration,
        )

    if not expected.is_file():
        raise FileNotFoundError(format_homography_missing_error(vp, expected))
    bad = validate_homography_json_file(expected)
    if bad:
        raise ValueError(bad)
    return expected
