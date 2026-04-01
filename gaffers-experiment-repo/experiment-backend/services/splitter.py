from __future__ import annotations

from dataclasses import dataclass


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
) -> list[ChunkSpec]:
    if total_frames <= 0:
        return []
    if chunking_policy == "none":
        return [ChunkSpec(chunk_id="chunk_0", start_frame=0, end_frame=total_frames - 1)]

    if chunking_policy == "auto":
        chunk_seconds = 120
    else:
        chunk_seconds = 300
    chunk_size = max(1, fps * chunk_seconds)
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
