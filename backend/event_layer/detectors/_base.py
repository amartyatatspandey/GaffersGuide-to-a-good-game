"""
Event Intelligence Layer — Base Detector
=========================================

Shared utilities and base class used by all detectors.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from event_layer.models import EventRecord
from event_layer.ontology import THRESHOLDS, classify_zone, FINAL_THIRD_ZONES


# ──────────────────────────────────────────────────────────────────────────────
# Tracking frame type alias (matches parallel_pipeline.py TrackingFrameArtifact)
# We work with plain dicts to avoid import coupling.
# ──────────────────────────────────────────────────────────────────────────────

# A frame dict looks like:
# {
#   "frame_idx": int,
#   "players": [{"id": int, "team_id": str, "x_pitch": float, "y_pitch": float, ...}],
#   "ball_xy": [float, float] | None,
#   "possession_team_id": str | None,
#   "homography_confidence": float,
# }

FrameList = list[dict[str, Any]]
PlayerRow = dict[str, Any]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _speed_mps(
    pt_a: tuple[float, float],
    pt_b: tuple[float, float],
    frames_elapsed: int,
    fps: float,
) -> float:
    """Return speed in m/s between two radar points across `frames_elapsed` frames."""
    if frames_elapsed <= 0 or fps <= 0:
        return 0.0
    return _dist(pt_a, pt_b) / (frames_elapsed / fps)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Apply a simple rolling mean with 'same' size output."""
    if not values:
        return []
    result = []
    half = window // 2
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _homography_discount(confidence: float) -> float:
    """
    Return a multiplier [0.5, 1.0] applied to spatial confidence calculations
    when homography quality is low.
    """
    if confidence >= THRESHOLDS.HOMOGRAPHY_CONFIDENCE_DISCOUNT_THRESHOLD:
        return 1.0
    return 0.5 + 0.5 * (confidence / THRESHOLDS.HOMOGRAPHY_CONFIDENCE_DISCOUNT_THRESHOLD)


def _build_player_position_history(
    frames: FrameList,
) -> dict[int, list[tuple[int, float, float, float]]]:
    """
    Build a per-player position history from the frame list.

    Returns:
        {player_id: [(frame_idx, x, y, homography_confidence), ...]}
    """
    history: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    for frame in frames:
        hconf = frame.get("homography_confidence", 1.0)
        for p in frame.get("players", []):
            pid = p.get("id")
            x = p.get("x_pitch")
            y = p.get("y_pitch")
            if pid is None or x is None or y is None:
                continue
            history[pid].append((frame["frame_idx"], float(x), float(y), float(hconf)))
    # Ensure sorted by frame
    for pid in history:
        history[pid].sort(key=lambda t: t[0])
    return history


def _make_event(
    *,
    event_type: str,
    event_name: str,
    category: str,
    player_id: int | None,
    team_id: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    confidence: float,
    importance: float,
    description: str,
    job_id: str,
    start_radar_pt: tuple[float, float] | None = None,
    end_radar_pt: tuple[float, float] | None = None,
    peak_radar_pt: tuple[float, float] | None = None,
    tags: list[str] | None = None,
) -> EventRecord:
    """Factory function for EventRecord with computed derived fields."""
    start_time_s = start_frame / fps
    end_time_s = end_frame / fps
    duration_s = end_time_s - start_time_s

    pitch_zone = "unknown"
    ref_pt = peak_radar_pt or end_radar_pt or start_radar_pt
    if ref_pt is not None:
        pitch_zone = classify_zone(ref_pt[0], ref_pt[1])

    # Clip window with padding
    padding_pre = int(THRESHOLDS.CLIP_PRE_PADDING_S * fps)
    padding_post = int(THRESHOLDS.CLIP_POST_PADDING_S * fps)
    clip_start = max(0, start_frame - padding_pre)
    clip_end = end_frame + padding_post  # upper-bound clamped by pipeline

    return EventRecord(
        event_type=event_type,
        event_name=event_name,
        category=category,
        player_id=player_id,
        team_id=team_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time_s=round(start_time_s, 2),
        end_time_s=round(end_time_s, 2),
        duration_s=round(duration_s, 2),
        pitch_zone=pitch_zone,
        start_radar_pt=start_radar_pt,
        end_radar_pt=end_radar_pt,
        peak_radar_pt=peak_radar_pt,
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        importance=round(max(0.0, min(1.0, importance)), 3),
        description=description,
        tags=tags or [],
        clip_start_frame=clip_start,
        clip_end_frame=clip_end,
        job_id=job_id,
        detected_at=_now_utc(),
    )
