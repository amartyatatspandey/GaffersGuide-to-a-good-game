from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from services.homography_vendor.dynamic_homography import (
    DynamicPitchCalibrator,
    PitchObservationBundle,
)

CALIB_W = 1280.0
CALIB_H = 720.0


class AdvancedPitchCalibrator:
    """
    Experiment-local advanced calibrator with production-compatible API.

    Output homography is normalized to a fixed 1280x720 tactical calibration space.
    """

    def __init__(self, weights_path: str | Path) -> None:
        self._base = DynamicPitchCalibrator(weights_path)

    def get_homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        base_h = self._base.get_homography(frame)
        if base_h is None:
            return None
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if h <= 0 or w <= 0:
            return None
        scale = np.array(
            [
                [CALIB_W / float(w), 0.0, 0.0],
                [0.0, CALIB_H / float(h), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        out = scale @ base_h
        out /= out[2, 2] if abs(out[2, 2]) > 1e-12 else 1.0
        return out.astype(np.float64)

    def collect_pitch_observations(self, frame: np.ndarray) -> Optional[PitchObservationBundle]:
        return self._base.collect_pitch_observations(frame)
