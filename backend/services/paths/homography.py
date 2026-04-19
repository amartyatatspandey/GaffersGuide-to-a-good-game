"""Homography JSON paths and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .calibration import (
    SN_CALIBRATION_REPO_URL,
    sn_calibration_resources_dir,
    sn_calibration_vendor_dir,
)
from .constants import BACKEND_ROOT

GAFFERS_HOMOGRAPHY_JSON_ENV = "GAFFERS_HOMOGRAPHY_JSON"


def resolve_tracking_homography_json_path(video_path: Path) -> Path:
    """
    Path to the homography JSON used by TacticalRadar for this video.

    If ``GAFFERS_HOMOGRAPHY_JSON`` is set, that file wins. Otherwise:
    ``backend/output/{video_path.stem}_homographies.json`` (same convention as
    ``run_calibrator_on_video``).
    """
    raw = os.getenv(GAFFERS_HOMOGRAPHY_JSON_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (BACKEND_ROOT / "output" / f"{video_path.stem}_homographies.json").resolve()


def format_homography_blocker_detail(
    video_path: Path,
    expected_json: Path,
    *,
    env_homography_set: bool,
) -> str:
    """Explain why homography JSON is missing when auto-calibration cannot finish the job."""
    vendor = sn_calibration_vendor_dir()
    res = sn_calibration_resources_dir()
    if env_homography_set:
        return (
            f"{GAFFERS_HOMOGRAPHY_JSON_ENV} is set; resolved path {expected_json} is not a usable file. "
            "Unset it to use per-upload backend/output/{stem}_homographies.json with auto-calibration, "
            "or set it to an existing homographies JSON path."
        )
    if not vendor.is_dir() or not (vendor / "src").is_dir():
        return (
            f"Missing vendored calibration package at {vendor}/src. Restore from repo "
            f"(see ARCHITECTURE_RESTRUCTURING_PLAN.md) or sync from {SN_CALIBRATION_REPO_URL}, "
            f"then place weights in {res} (see sn-calibration README)."
        )
    if not res.is_dir():
        return (
            f"Calibration weights directory missing: {res}. "
            "Copy ``resources`` from upstream sn-calibration into this path, or set "
            "SN_CALIBRATION_RESOURCES_DIR to an existing directory containing "
            "soccer_pitch_segmentation.pth, mean.npy, and std.npy."
        )
    return (
        f"Auto-calibration did not produce a file. From backend: "
        f"python -m scripts.pipeline_core.run_calibrator_on_video --video {video_path} "
        f"--output {expected_json}. Or set {GAFFERS_HOMOGRAPHY_JSON_ENV} to an existing homographies JSON."
    )


def format_homography_missing_error(video_path: Path, expected_json: Path) -> str:
    env_set = bool(os.getenv(GAFFERS_HOMOGRAPHY_JSON_ENV, "").strip())
    detail = format_homography_blocker_detail(
        video_path, expected_json, env_homography_set=env_set
    )
    return f"Homography JSON not found at {expected_json}. {detail}"


def validate_homography_json_file(path: Path) -> str:
    """
    Return a single-line gap message if ``path`` is not a valid non-empty homography artifact, else "".
    """
    if not path.is_file():
        return f"Homography path is not a file: {path}"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return f"Homography JSON invalid at {path}: {exc}"
    homographies = data.get("homographies")
    if not isinstance(homographies, list) or len(homographies) == 0:
        return (
            f"Homography JSON at {path} has no usable 'homographies' entries. "
            "Re-run scripts.pipeline_core.run_calibrator_on_video on this match or fix the file."
        )
    first = homographies[0]
    if not isinstance(first, dict) or "homography" not in first:
        return (
            f"Homography JSON at {path} has invalid homographies[0] "
            "(expected dict with 'homography' matrix)."
        )
    return ""
