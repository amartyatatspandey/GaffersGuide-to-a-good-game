"""
Pitch and radar visualization utilities.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np

PITCH_LENGTH: int = 105
PITCH_WIDTH: int = 68
CENTER_CIRCLE_RADIUS_M: float = 9.15
PITCH_GREEN: Tuple[int, int, int] = (50, 168, 82)
WHITE: Tuple[int, int, int] = (255, 255, 255)


def draw_pitch(scale: int = 5) -> np.ndarray:
    width = PITCH_LENGTH * scale
    height = PITCH_WIDTH * scale
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = PITCH_GREEN
    thickness = max(1, scale // 5)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), WHITE, thickness)
    cx = width // 2
    cv2.line(img, (cx, 0), (cx, height), WHITE, thickness)
    center = (cx, height // 2)
    radius_px = int(CENTER_CIRCLE_RADIUS_M * scale)
    cv2.circle(img, center, radius_px, WHITE, thickness)
    return img


def draw_radar(
    pitch_image: np.ndarray,
    player_positions: List[Tuple[float, float]] | Sequence[Tuple[float, float]] | np.ndarray,
    color: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (0, 0, 255),
    scale: int = 5,
    radius: int | None = None,
) -> np.ndarray:
    img = pitch_image.copy()
    pts = np.asarray(player_positions, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.size == 0:
        return img
    if radius is None:
        radius = max(2, scale // 2)
    colors_list: List[Tuple[int, int, int]] = (
        color if isinstance(color, list) else [color] * len(pts)
    )
    for i in range(len(pts)):
        x_m, y_m = float(pts[i, 0]), float(pts[i, 1])
        px, py = int(round(x_m * scale)), int(round(y_m * scale))
        c = colors_list[i] if i < len(colors_list) else (0, 0, 255)
        cv2.circle(img, (px, py), radius, c, -1)
    return img
