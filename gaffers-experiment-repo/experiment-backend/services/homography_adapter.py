from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from services.homography_vendor.advanced_pitch_calibration import AdvancedPitchCalibrator


@dataclass(slots=True)
class HomographyEstimate:
    matrix_pitch_to_image: Optional[np.ndarray]
    calibration_latency_ms: float
    used_fallback: bool


class HomographyService:
    def __init__(self, weights_dir: Path) -> None:
        self._enabled = os.getenv("EXP_HOMOGRAPHY_ENABLED", "1") == "1"
        self._sample_every = max(1, int(os.getenv("EXP_HOMOGRAPHY_SAMPLE_EVERY", "5")))
        self._calibrator: Optional[AdvancedPitchCalibrator] = None
        self._last_h: Optional[np.ndarray] = None
        if self._enabled:
            self._calibrator = AdvancedPitchCalibrator(weights_dir)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._calibrator is not None

    def estimate(self, frame: np.ndarray, frame_idx: int) -> HomographyEstimate:
        if not self.enabled:
            return HomographyEstimate(matrix_pitch_to_image=None, calibration_latency_ms=0.0, used_fallback=False)
        should_refresh = (frame_idx % self._sample_every) == 0 or self._last_h is None
        if not should_refresh and self._last_h is not None:
            return HomographyEstimate(
                matrix_pitch_to_image=self._last_h,
                calibration_latency_ms=0.0,
                used_fallback=True,
            )
        start = time.perf_counter()
        matrix = self._calibrator.get_homography(frame) if self._calibrator is not None else None
        latency_ms = (time.perf_counter() - start) * 1000.0
        if matrix is not None:
            self._last_h = matrix.astype(np.float64)
            return HomographyEstimate(matrix_pitch_to_image=self._last_h, calibration_latency_ms=latency_ms, used_fallback=False)
        return HomographyEstimate(matrix_pitch_to_image=self._last_h, calibration_latency_ms=latency_ms, used_fallback=self._last_h is not None)

    @staticmethod
    def image_point_to_pitch(point_xy: tuple[float, float], matrix_pitch_to_image: np.ndarray) -> Optional[list[float]]:
        if matrix_pitch_to_image is None:
            return None
        try:
            inv = np.linalg.inv(matrix_pitch_to_image)
        except np.linalg.LinAlgError:
            return None
        point = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(point, inv)
        if mapped.size != 2:
            return None
        return [float(mapped[0][0][0]), float(mapped[0][0][1])]
