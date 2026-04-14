from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from services.homography_adapter import HomographyService
from services.window_selector import Window


@dataclass(slots=True)
class DensePassResult:
    tracked_rows: list[dict[str, object]]
    infer_ms: float
    post_ms: float
    frames_with_homography: int
    frames_without_homography: int
    fallback_frames: int
    calibration_latency_ms: float


def run_dense_pass(
    frames: list[np.ndarray],
    *,
    windows: list[Window],
    runtime_backend: str,
    homography_weights_dir: Path,
) -> DensePassResult:
    infer_start = time.perf_counter()
    tracked: list[dict[str, object]] = []
    homography = HomographyService(homography_weights_dir)
    frames_with_homography = 0
    frames_without_homography = 0
    fallback_frames = 0
    calibration_latency_ms = 0.0
    for idx in _iter_window_frame_indices(windows, len(frames)):
            frame = frames[idx]
            h, w = frame.shape[:2]
            estimate = homography.estimate(frame, idx)
            calibration_latency_ms += estimate.calibration_latency_ms
            ball_pitch_xy = None
            if estimate.matrix_pitch_to_image is not None:
                ball_pitch_xy = homography.image_point_to_pitch(
                    (float(w / 2.0), float(h / 2.0)),
                    estimate.matrix_pitch_to_image,
                )
                frames_with_homography += 1
                if estimate.used_fallback:
                    fallback_frames += 1
            else:
                frames_without_homography += 1
            tracked.append(
                {
                    "frame_idx": idx,
                    "players": [],
                    "ball_xy": ball_pitch_xy,
                    "possession_team_id": 0,
                    "runtime_backend": runtime_backend,
                    "coord_space": "pitch",
                    "homography_applied": ball_pitch_xy is not None,
                }
            )
    infer_ms = (time.perf_counter() - infer_start) * 1000.0
    post_start = time.perf_counter()
    tracked.sort(key=lambda row: int(row["frame_idx"]))
    dedup: dict[int, dict[str, object]] = {}
    for row in tracked:
        dedup[int(row["frame_idx"])] = row
    tracked_rows = list(dedup.values())
    post_ms = (time.perf_counter() - post_start) * 1000.0
    return DensePassResult(
        tracked_rows=tracked_rows,
        infer_ms=infer_ms,
        post_ms=post_ms,
        frames_with_homography=frames_with_homography,
        frames_without_homography=frames_without_homography,
        fallback_frames=fallback_frames,
        calibration_latency_ms=calibration_latency_ms,
    )


def _iter_window_frame_indices(windows: list[Window], frame_count: int):
    for win in windows:
        end = min(win.end_frame + 1, frame_count)
        for idx in range(win.start_frame, end):
            yield idx
