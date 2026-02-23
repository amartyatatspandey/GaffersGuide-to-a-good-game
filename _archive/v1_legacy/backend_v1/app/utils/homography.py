"""
Homography utilities for mapping image (pixel) coordinates to real-world pitch meters.
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REAL_WORLD_CORNERS: np.ndarray = np.array(
    [[0, 0], [105, 0], [105, 68], [0, 68]],
    dtype=np.float32,
)


def _sort_corners_to_real_world_order(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    sum_xy = pts[:, 0] + pts[:, 1]
    tl_idx = int(np.argmin(sum_xy))
    br_idx = int(np.argmax(sum_xy))
    other = [i for i in range(4) if i not in (tl_idx, br_idx)]
    diff_yx = pts[other, 1] - pts[other, 0]
    tr_idx = other[int(np.argmin(diff_yx))]
    bl_idx = other[int(np.argmax(diff_yx))]
    return np.array([pts[tl_idx], pts[tr_idx], pts[br_idx], pts[bl_idx]], dtype=np.float32)


def calculate_homography(detected_corners: np.ndarray) -> np.ndarray | None:
    if detected_corners is None:
        return None
    pts = np.asarray(detected_corners, dtype=np.float32)
    if pts.size < 8:
        return None
    pts = pts.reshape(4, 2)
    sorted_corners = _sort_corners_to_real_world_order(pts)
    H, _ = cv2.findHomography(sorted_corners, REAL_WORLD_CORNERS)
    return H


def transform_points(
    points: List[Tuple[float, float]] | Sequence[Tuple[float, float]] | np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    src = pts.reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, H)
    return dst.reshape(-1, 2)
