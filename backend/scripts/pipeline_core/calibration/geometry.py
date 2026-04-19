"""Geometry helpers for line-based homography estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from src.baseline_cameras import estimate_homography_from_line_correspondences, normalization_transform


def build_line_matches(extremities: dict, field: Any, width: int, height: int) -> list[tuple[np.ndarray, np.ndarray]]:
    line_matches: list[tuple[np.ndarray, np.ndarray]] = []
    for k, v in extremities.items():
        if k == "Circle central" or "unknown" in k:
            continue
        if k not in field.line_extremities_keys:
            continue
        p1 = np.array([v[0]["x"] * width, v[0]["y"] * height, 1.0], dtype=np.float64)
        p2 = np.array([v[1]["x"] * width, v[1]["y"] * height, 1.0], dtype=np.float64)
        line_image = np.cross(p1, p2)
        if np.any(np.isnan(line_image)) or np.any(np.isinf(line_image)):
            continue
        line_pitch = field.get_2d_homogeneous_line(k)
        if line_pitch is not None:
            line_matches.append((line_pitch, line_image))
    return line_matches


def estimate_extremities_homography(extremities: dict, field: Any, width: int, height: int) -> np.ndarray | None:
    line_matches = build_line_matches(extremities, field, width, height)
    if len(line_matches) < 4:
        return None

    src_pts: list[np.ndarray] = []
    target_keys: set[str] = set()
    for k, v in extremities.items():
        if k == "Circle central" or "unknown" in k or k not in field.line_extremities_keys:
            continue
        p3d1, p3d2 = field.line_extremities_keys[k]
        target_keys.add(p3d1)
        target_keys.add(p3d2)
        src_pts.append(np.array([v[0]["x"] * width, v[0]["y"] * height]))
        src_pts.append(np.array([v[1]["x"] * width, v[1]["y"] * height]))
    target_pts = [field.point_dict[k][:2] for k in target_keys]

    t1 = normalization_transform(target_pts)
    t2 = normalization_transform(src_pts)
    success, homography = estimate_homography_from_line_correspondences(line_matches, t1, t2)
    if not success:
        return None
    return homography
