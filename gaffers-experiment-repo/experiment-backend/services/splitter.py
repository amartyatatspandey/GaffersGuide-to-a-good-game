from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(slots=True)
class ChunkSpec:
    chunk_id: str
    start_frame: int
    end_frame: int


def build_chunks(
    *,
    total_frames: int,
    chunking_policy: str,
    fps: int,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> list[ChunkSpec]:
    if total_frames <= 0:
        return []
    if chunking_policy == "none":
        return [ChunkSpec(chunk_id="chunk_0", start_frame=0, end_frame=total_frames - 1)]

    if chunking_policy == "auto":
        chunk_seconds = 60
    else:
        chunk_seconds = 120
    budget_chunk = compute_max_chunk_frames(
        cgroup_limit_bytes=read_cgroup_memory_limit_bytes(),
        width=frame_width,
        height=frame_height,
        safety_factor=0.60,
    )
    chunk_size = max(1, min(int(fps * chunk_seconds), budget_chunk))
    chunks: list[ChunkSpec] = []
    start = 0
    index = 0
    while start < total_frames:
        end = min(total_frames - 1, start + chunk_size - 1)
        chunks.append(
            ChunkSpec(chunk_id=f"chunk_{index}", start_frame=start, end_frame=end)
        )
        start = end + 1
        index += 1
    return chunks


def read_cgroup_memory_limit_bytes() -> int:
    cgroup_path = Path("/sys/fs/cgroup/memory.max")
    try:
        raw = cgroup_path.read_text(encoding="utf-8").strip()
        if raw and raw != "max":
            return int(raw)
    except Exception:
        pass
    return 8 * 1024 * 1024 * 1024


def compute_max_chunk_frames(
    *,
    cgroup_limit_bytes: int,
    width: int,
    height: int,
    safety_factor: float = 0.60,
) -> int:
    frame_bytes = max(1, int(width) * int(height) * 3 // 2)
    return max(1, math.floor((int(cgroup_limit_bytes) * float(safety_factor)) / frame_bytes))
