"""Segmentation inference helpers for DynamicPitchCalibrator."""

from __future__ import annotations

from typing import Any

import numpy as np
from src.detect_extremities import generate_class_synthesis, get_line_extremities


def run_segmentation_and_extremities(
    seg_net: Any,
    frame: np.ndarray,
    *,
    seg_width: int,
    seg_height: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    semlines = seg_net.analyse_image(frame)
    skeletons = generate_class_synthesis(semlines, radius=6)
    extremities = get_line_extremities(
        skeletons,
        maxdist=40,
        width=seg_width,
        height=seg_height,
    )
    return semlines, skeletons, extremities
