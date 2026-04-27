# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from libc.math cimport sqrt

try:
    from gaffers_core_math import rank_candidates_rs as _rank_candidates_rs
except ImportError:  # pragma: no cover - optional Rust acceleration
    _rank_candidates_rs = None


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
    cdef double ax
    cdef double ay
    cdef double radius
    cdef double cx
    cdef double cy
    cdef double dx
    cdef double dy
    cdef double dist
    cdef double proximity
    cdef double candidate_score
    cdef double best_score
    cdef object candidate
    cdef object best

    if not candidates:
        return None
    if _rank_candidates_rs is not None:
        payload = [
            (
                float(c.xyxy[0]),
                float(c.xyxy[1]),
                float(c.xyxy[2]),
                float(c.xyxy[3]),
                float(c.confidence),
            )
            for c in candidates
        ]
        idx = _rank_candidates_rs(payload, temporal_anchor_xy, int(search_radius_px))
        if idx is not None:
            return candidates[int(idx)]
    if temporal_anchor_xy is None or search_radius_px <= 0:
        return max(candidates, key=lambda c: c.confidence)

    ax, ay = temporal_anchor_xy
    radius = float(max(1, search_radius_px))
    best = None
    best_score = -1.0
    for candidate in candidates:
        cx, cy = _center_xy(candidate.xyxy)
        dx = cx - ax
        dy = cy - ay
        dist = sqrt(dx * dx + dy * dy)
        proximity = max(0.0, 1.0 - (dist / radius))
        candidate_score = float(candidate.confidence) + 0.35 * proximity
        if best is None or candidate_score > best_score:
            best = candidate
            best_score = candidate_score
    return best

