from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FastPassResult:
    event_frames: list[int]
    elapsed_ms: float


def run_fast_pass(
    frames: list[np.ndarray],
    *,
    quality_mode: str,
) -> FastPassResult:
    start = time.perf_counter()
    stride = 10 if quality_mode == "fast" else 5
    event_frames = [idx for idx in range(0, len(frames), stride)]
    return FastPassResult(
        event_frames=event_frames,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )
