"""Homography fitting and mapping primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gaffers_guide.core.types import SpatialMapping


@dataclass
class HomographyEngine:
    """Compute spatial mappings from pitch corner correspondences."""

    pitch_corners_meters: NDArray[np.float64] = np.array(
        [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]], dtype=np.float64
    )

    def fit(
        self, pitch_corners_px: NDArray[np.float64], frame_shape: tuple[int, int]
    ) -> SpatialMapping:
        import cv2

        src = self.pitch_corners_meters.astype(np.float32)
        dst = pitch_corners_px.astype(np.float32)
        h, _ = cv2.findHomography(src, dst)
        if h is None:
            raise ValueError("Could not compute homography matrix")
        return SpatialMapping(
            homography_matrix=h.astype(np.float64),
            pitch_corners_px=pitch_corners_px.astype(np.float64),
            pitch_corners_meters=self.pitch_corners_meters.astype(np.float64),
            frame_shape=frame_shape,
        )
