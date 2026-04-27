# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""
Dynamic pitch calibration using SoccerNet/sn-calibration reference pipeline.

Maps a single broadcast frame to a 2D pitch via a 3x3 homography (pitch plane -> image).
Vendored SoccerNet calibration ``src`` under ``backend/calibration/sn_calibration_vendor`` (not ``references/``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from calibration.ports import PitchLineModel
else:
    PitchLineModel = Any  # runtime: vendor SoccerPitch

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Vendored sn-calibration ``src`` (production must not import from backend/references/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BACKEND_ROOT = _PROJECT_ROOT / "backend"
_REF_DIR = _BACKEND_ROOT / "calibration" / "sn_calibration_vendor"
_CALIBRATION_DIR = _BACKEND_ROOT / "calibration"
if not _REF_DIR.is_dir() or not (_REF_DIR / "src").is_dir():
    raise ImportError(
        f"sn-calibration vendor tree not found at {_REF_DIR}/src. "
        "See ARCHITECTURE_RESTRUCTURING_PLAN.md — restore backend/calibration/sn_calibration_vendor from upstream."
    )
import sys
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
if str(_CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIBRATION_DIR))
if str(_REF_DIR) not in sys.path:
    # Keep backend package imports (e.g. calibration.soccer_pitch_adapter) ahead of vendor modules.
    sys.path.append(str(_REF_DIR))
_cal_mod = sys.modules.get("calibration")
if _cal_mod is not None:
    cal_file = Path(getattr(_cal_mod, "__file__", "") or "")
    if not str(cal_file).startswith(str(_BACKEND_ROOT / "calibration")):
        # Avoid third-party/vendor calibration package shadowing backend/calibration.
        sys.modules.pop("calibration", None)

from src.detect_extremities import SegmentationNetwork

from soccer_pitch_adapter import create_pitch_model
from gaffers_guide.calibration.geometry import estimate_extremities_homography
from gaffers_guide.calibration.inference import run_segmentation_and_extremities


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
        from gaffers_guide.pipeline.sn_calib_weights import require_sn_calibration_weight_files

        pth, mean_npy, std_npy = require_sn_calibration_weight_files(self._weights_dir)

        self._seg_net = SegmentationNetwork(
            model_file=str(pth),
            mean_file=str(mean_npy),
            std_file=str(std_npy),
            num_classes=29,
            width=self.SEG_WIDTH,
            height=self.SEG_HEIGHT,
        )
        self._field = create_pitch_model()
        logger.info(
            "DynamicPitchCalibrator initialized with weights from %s",
            self._weights_dir,
        )

    @property
    def field(self) -> "PitchLineModel":
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

        semlines, skeletons, extremities = run_segmentation_and_extremities(
            self._seg_net,
            frame,
            seg_width=self.SEG_WIDTH,
            seg_height=self.SEG_HEIGHT,
        )
        if not extremities:
            logger.debug("collect_pitch_observations: no line extremities")
            return None

        H_seg = estimate_extremities_homography(
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
