from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MergeResult:
    frames: list[dict[str, object]]


def merge_chunk_rows(rows_by_chunk: list[list[dict[str, object]]]) -> MergeResult:
    rows: list[dict[str, object]] = []
    for chunk_rows in rows_by_chunk:
        rows.extend(chunk_rows)
    rows.sort(key=lambda row: int(row["frame_idx"]))
    dedup: dict[int, dict[str, object]] = {}
    for row in rows:
        dedup[int(row["frame_idx"])] = row
    return MergeResult(frames=list(dedup.values()))
