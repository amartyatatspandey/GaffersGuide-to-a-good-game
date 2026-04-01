from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from services.window_selector import Window


@dataclass(slots=True)
class DensePassResult:
    tracked_rows: list[dict[str, object]]
    infer_ms: float
    post_ms: float


def run_dense_pass(
    frames: list[np.ndarray],
    *,
    windows: list[Window],
    runtime_backend: str,
) -> DensePassResult:
    infer_start = time.perf_counter()
    tracked: list[dict[str, object]] = []
    for win in windows:
        for idx in range(win.start_frame, min(win.end_frame + 1, len(frames))):
            frame = frames[idx]
            h, w = frame.shape[:2]
            tracked.append(
                {
                    "frame_idx": idx,
                    "players": [],
                    "ball_xy": [float(w / 2.0), float(h / 2.0)],
                    "possession_team_id": 0,
                    "runtime_backend": runtime_backend,
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
    return DensePassResult(tracked_rows=tracked_rows, infer_ms=infer_ms, post_ms=post_ms)
