"""
Dynamic pitch calibration using SoccerNet/sn-calibration reference pipeline.

Maps a single broadcast frame to a 2D pitch via a 3x3 homography (pitch plane -> image).
Reference: backend/references/sn-calibration (EVS Camera Calibration Challenge).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Resolve reference repo and extend path for sn-calibration src imports
_REF_DIR = Path(__file__).resolve().parent.parent / "references" / "sn-calibration"
if not _REF_DIR.is_dir():
    raise ImportError(
        f"sn-calibration reference not found at {_REF_DIR}. "
        "Clone https://github.com/SoccerNet/sn-calibration into backend/references/sn-calibration."
    )
import sys
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

from src.soccerpitch import SoccerPitch
from src.detect_extremities import (
    SegmentationNetwork,
    generate_class_synthesis,
    get_line_extremities,
)
from src.baseline_cameras import (
    normalization_transform,
    estimate_homography_from_line_correspondences,
)


def _build_line_matches(
    extremities: dict,
    field: SoccerPitch,
    width: int,
    height: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build list of (pitch_line_2d_homogeneous, image_line_2d_homogeneous) for homography estimation."""
    line_matches: list[tuple[np.ndarray, np.ndarray]] = []
    for k, v in extremities.items():
        if k == "Circle central" or "unknown" in k:
            continue
        if k not in field.line_extremities_keys:
            continue
        p1 = np.array(
            [v[0]["x"] * width, v[0]["y"] * height, 1.0],
            dtype=np.float64,
        )
        p2 = np.array(
            [v[1]["x"] * width, v[1]["y"] * height, 1.0],
            dtype=np.float64,
        )
        line_image = np.cross(p1, p2)
        if np.any(np.isnan(line_image)) or np.any(np.isinf(line_image)):
            continue
        line_pitch = field.get_2d_homogeneous_line(k)
        if line_pitch is not None:
            line_matches.append((line_pitch, line_image))
    return line_matches


def _extremities_to_homography(
    extremities: dict,
    field: SoccerPitch,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """
    Estimate 3x3 homography (pitch plane -> image) from line extremities.
    Returns None if fewer than 4 line correspondences or estimation fails.
    """
    line_matches = _build_line_matches(extremities, field, width, height)
    if len(line_matches) < 4:
        logger.debug("Insufficient line matches for homography: %d < 4", len(line_matches))
        return None

    # Normalization: source = pitch (3D points), target = image (2D points)
    # We need point sets for T1 (pitch) and T2 (image). baseline_cameras uses
    # target_pts = 3D pitch points, src_pts = image points (p1, p2 for each line).
    src_pts: list[np.ndarray] = []
    target_keys: set[str] = set()
    for k, v in extremities.items():
        if k == "Circle central" or "unknown" in k or k not in field.line_extremities_keys:
            continue
        P3D1, P3D2 = field.line_extremities_keys[k]
        target_keys.add(P3D1)
        target_keys.add(P3D2)
        src_pts.append(np.array([v[0]["x"] * width, v[0]["y"] * height]))
        src_pts.append(np.array([v[1]["x"] * width, v[1]["y"] * height]))
    target_pts = [field.point_dict[k][:2] for k in target_keys]

    T1 = normalization_transform(target_pts)
    T2 = normalization_transform(src_pts)
    success, homography = estimate_homography_from_line_correspondences(line_matches, T1, T2)
    if not success:
        logger.debug("Homography estimation failed (SVD/constraints).")
        return None
    return homography


@dataclass(frozen=True)
class PitchObservationBundle:
    """
    Intermediate calibration observations at segmentation resolution (SEG_WIDTH x SEG_HEIGHT).

    Pixel conventions for polylines / OpenCV: extremities use normalized coords; polylines from
    sn-calibration use (row, col) per point — see get_line_extremities in detect_extremities.py.
    """

    semlines: np.ndarray
    skeletons: dict[str, Any]
    extremities: dict[str, Any]
    H_coarse_seg: np.ndarray
    frame_shape: tuple[int, int]


class DynamicPitchCalibrator:
    """
    Standalone calibrator: one video frame (BGR) in -> 3x3 homography (pitch -> image) or None.

    Uses the SoccerNet/sn-calibration segmentation baseline: DeepLabv3-ResNet50 for pitch
    line segmentation, then line extremities and line-based homography estimation.
    """

    # Segmentation network expects this input size (from reference)
    SEG_WIDTH = 640
    SEG_HEIGHT = 360

    def __init__(self, weights_path: str | Path) -> None:
        """
        Initialize the SoccerNet segmentation model and load weights.

        :param weights_path: Path to the directory containing:
            - soccer_pitch_segmentation.pth
            - mean.npy
            - std.npy
        :raises FileNotFoundError: If any of the required files are missing.
        """
        self._weights_dir = Path(weights_path)
        if not self._weights_dir.is_dir():
            raise FileNotFoundError(f"Weights directory not found: {self._weights_dir}")

        pth = self._weights_dir / "soccer_pitch_segmentation.pth"
        mean_npy = self._weights_dir / "mean.npy"
        std_npy = self._weights_dir / "std.npy"
        for f in (pth, mean_npy, std_npy):
            if not f.is_file():
                raise FileNotFoundError(
                    f"Missing required file: {f}. "
                    "Download weights from README (Google Drive) and place in the weights directory."
                )

        self._seg_net = SegmentationNetwork(
            model_file=str(pth),
            mean_file=str(mean_npy),
            std_file=str(std_npy),
            num_classes=29,
            width=self.SEG_WIDTH,
            height=self.SEG_HEIGHT,
        )
        self._field = SoccerPitch()
        logger.info(
            "DynamicPitchCalibrator initialized with weights from %s",
            self._weights_dir,
        )

    @property
    def field(self) -> SoccerPitch:
        """FIFA pitch model used for line correspondences (read-only)."""
        return self._field

    def collect_pitch_observations(self, frame: np.ndarray) -> Optional[PitchObservationBundle]:
        """
        Run segmentation through coarse SVD homography at SEG resolution.

        :param frame: BGR (H, W, 3).
        :return: Bundle with ``H_coarse_seg`` mapping pitch -> segmentation pixels, or None.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("collect_pitch_observations: invalid frame")
            return None
        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning("collect_pitch_observations: frame must be BGR (H, W, 3)")
            return None

        height_img, width_img = int(frame.shape[0]), int(frame.shape[1])
        try:
            semlines = self._seg_net.analyse_image(frame)
        except Exception as e:
            logger.exception("Segmentation inference failed: %s", e)
            return None

        skeletons = generate_class_synthesis(semlines, radius=6)
        extremities = get_line_extremities(
            skeletons,
            maxdist=40,
            width=self.SEG_WIDTH,
            height=self.SEG_HEIGHT,
        )
        if not extremities:
            logger.debug("collect_pitch_observations: no line extremities")
            return None

        H_seg = _extremities_to_homography(
            extremities,
            self._field,
            width=self.SEG_WIDTH,
            height=self.SEG_HEIGHT,
        )
        if H_seg is None:
            return None

        return PitchObservationBundle(
            semlines=semlines,
            skeletons=skeletons,
            extremities=extremities,
            H_coarse_seg=H_seg.astype(np.float64),
            frame_shape=(height_img, width_img),
        )

    def get_homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run SoccerNet inference on a single frame and return the 3x3 homography.

        The returned matrix H maps 2D pitch plane (FIFA coordinates, homogeneous) to image
        coordinates: p_image = H @ p_pitch (with p_* as (x, y, 1)^T). So to map image -> pitch
        use np.linalg.inv(H).

        :param frame: BGR image (OpenCV convention), shape (H, W, 3), any resolution.
        :return: 3x3 numpy array (float64) homography pitch -> image, or None if the pitch
            could not be estimated (e.g. too few line detections).
        """
        obs = self.collect_pitch_observations(frame)
        if obs is None:
            return None

        height_img, width_img = obs.frame_shape[0], obs.frame_shape[1]
        H = obs.H_coarse_seg

        # Scale homography from segmentation resolution to actual frame size
        if (width_img, height_img) != (self.SEG_WIDTH, self.SEG_HEIGHT):
            sx = width_img / self.SEG_WIDTH
            sy = height_img / self.SEG_HEIGHT
            S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
            H = S @ H

        return H.astype(np.float64)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Test DynamicPitchCalibrator on a single frame.")
    parser.add_argument("weights_dir", type=str, help="Path to dir with .pth, mean.npy, std.npy")
    parser.add_argument("image", type=str, help="Path to a BGR image (e.g. frame from broadcast)")
    args = parser.parse_args()

    calibrator = DynamicPitchCalibrator(args.weights_dir)
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    H = calibrator.get_homography(frame)
    if H is not None:
        print("Homography (pitch -> image) 3x3:")
        print(H)
    else:
        print("No homography (pitch not found).")
        raise SystemExit(1)
