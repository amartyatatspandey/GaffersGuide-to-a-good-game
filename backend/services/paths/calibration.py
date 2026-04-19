"""SoccerNet sn-calibration vendor and weight directories."""

from __future__ import annotations

import os
from pathlib import Path

from .constants import BACKEND_ROOT

SN_CALIBRATION_REPO_URL = "https://github.com/SoccerNet/sn-calibration"
SN_CALIBRATION_RESOURCES_DIR_ENV = "SN_CALIBRATION_RESOURCES_DIR"


def sn_calibration_vendor_dir() -> Path:
    """Vendored sn-calibration tree: contains ``src/`` so ``from src.*`` imports resolve (production path)."""
    return BACKEND_ROOT / "calibration" / "sn_calibration_vendor"


def sn_calibration_root_dir() -> Path:
    """Alias for :func:`sn_calibration_vendor_dir` (backward-compatible name for scripts and tests)."""
    return sn_calibration_vendor_dir()


def sn_calibration_resources_dir() -> Path:
    """
    SoccerNet pitch-segmentation weights (``soccer_pitch_segmentation.pth``, ``mean.npy``, ``std.npy``).

    Default: ``backend/models/calibration/sn-calibration/resources``. Override with
    ``SN_CALIBRATION_RESOURCES_DIR`` (absolute path or path relative to ``BACKEND_ROOT``).
    """
    raw = os.getenv(SN_CALIBRATION_RESOURCES_DIR_ENV, "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (BACKEND_ROOT / candidate).resolve()
    return (
        BACKEND_ROOT / "models" / "calibration" / "sn-calibration" / "resources"
    ).resolve()
