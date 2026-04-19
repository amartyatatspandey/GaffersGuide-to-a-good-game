"""Adapter around vendored sn-calibration SoccerPitch and helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REF_DIR = _BACKEND_ROOT / "calibration" / "sn_calibration_vendor"
if not _REF_DIR.is_dir() or not (_REF_DIR / "src").is_dir():
    raise ImportError(f"sn-calibration vendor tree not found at {_REF_DIR}/src")
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from src.soccerpitch import SoccerPitch  # noqa: E402
from src.detect_extremities import join_points  # noqa: E402


def create_pitch_model() -> SoccerPitch:
    return SoccerPitch()
