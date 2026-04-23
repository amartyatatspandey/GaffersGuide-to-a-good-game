from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusedCandidate:
    xyxy: np.ndarray
    confidence: float
    score: float
    source: str


def _center_xy(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy.tolist()
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


def rank_candidates(
    candidates: list[FusedCandidate],
    *,
    temporal_anchor_xy: tuple[float, float] | None,
    search_radius_px: int,
) -> FusedCandidate | None:
    if not candidates:
        return None
    if temporal_anchor_xy is None or search_radius_px <= 0:
        return max(candidates, key=lambda c: c.confidence)

    ax, ay = temporal_anchor_xy
    radius = float(max(1, search_radius_px))

    def score(c: FusedCandidate) -> float:
        cx, cy = _center_xy(c.xyxy)
        dist = float(np.hypot(cx - ax, cy - ay))
        proximity = max(0.0, 1.0 - (dist / radius))
        return c.confidence + 0.35 * proximity

    return max(candidates, key=score)

