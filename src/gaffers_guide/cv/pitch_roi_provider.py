from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchROIMask:
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int] | None
    area_ratio: float


class PitchROIProvider:
    """Estimate pitch occupancy from HSV green chroma."""

    def __init__(
        self,
        *,
        sat_min: int = 30,
        val_min: int = 20,
        morph_kernel: int = 7,
    ) -> None:
        self._sat_min = max(0, int(sat_min))
        self._val_min = max(0, int(val_min))
        self._kernel_size = max(3, int(morph_kernel) | 1)

    def estimate(self, frame_bgr: np.ndarray) -> PitchROIMask:
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            empty = np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
            return PitchROIMask(mask=empty, bbox_xyxy=None, area_ratio=0.0)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([25, self._sat_min, self._val_min], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._kernel_size, self._kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        non_zero = int(np.count_nonzero(mask))
        area_ratio = float(non_zero / float(h * w))
        if non_zero <= 0:
            return PitchROIMask(mask=mask, bbox_xyxy=None, area_ratio=0.0)

        ys, xs = np.where(mask > 0)
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1

        return PitchROIMask(mask=mask, bbox_xyxy=(x1, y1, x2, y2), area_ratio=area_ratio)

