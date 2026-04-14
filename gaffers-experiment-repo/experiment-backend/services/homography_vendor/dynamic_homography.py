from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PitchObservationBundle:
    homography_pitch_to_image: np.ndarray
    frame_shape: tuple[int, int]


class DynamicPitchCalibrator:
    """
    Experiment-local homography calibrator interface compatible with production callers.

    This vendorized module keeps the same callable surface as production calibrators
    while remaining lightweight for experiment backend environments.
    """

    PITCH_WIDTH = 105.0
    PITCH_HEIGHT = 68.0

    def __init__(self, weights_path: str | Path) -> None:
        self._weights_dir = Path(weights_path)
        # Keep constructor parity with production; weights are optional in experiment mode.

    def collect_pitch_observations(self, frame: np.ndarray) -> Optional[PitchObservationBundle]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        h, w = int(frame.shape[0]), int(frame.shape[1])
        # Pitch (meters) -> image pixels coarse mapping.
        homography = np.array(
            [
                [w / self.PITCH_WIDTH, 0.0, 0.0],
                [0.0, h / self.PITCH_HEIGHT, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return PitchObservationBundle(homography_pitch_to_image=homography, frame_shape=(h, w))

    def get_homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        obs = self.collect_pitch_observations(frame)
        if obs is None:
            return None
        return obs.homography_pitch_to_image.astype(np.float64)
