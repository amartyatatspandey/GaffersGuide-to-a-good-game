from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Iterator


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


def write_chunk_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda r: int(r["frame_idx"])):
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def merge_sorted_chunk_jsonl(chunk_paths: list[Path]) -> Iterator[dict[str, object]]:
    iterables = [iter_jsonl_rows(p) for p in chunk_paths]
    keyed = [((int(row["frame_idx"]), row), idx) for idx, it in enumerate(iterables) for row in []]
    # Build heap lazily to avoid loading all rows.
    heap: list[tuple[int, int, dict[str, object], Iterator[dict[str, object]]]] = []
    for idx, it in enumerate(iterables):
        try:
            row = next(it)
            heapq.heappush(heap, (int(row["frame_idx"]), idx, row, it))
        except StopIteration:
            continue
    while heap:
        _, idx, row, it = heapq.heappop(heap)
        yield row
        try:
            nxt = next(it)
            heapq.heappush(heap, (int(nxt["frame_idx"]), idx, nxt, it))
        except StopIteration:
            continue


def atomic_write_jsonl(final_path: Path, rows: Iterator[dict[str, object]]) -> str:
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    sha = hashlib.sha256()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, separators=(",", ":"))
            encoded = (line + "\n").encode("utf-8")
            f.write(line + "\n")
            sha.update(encoded)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)
    digest = sha.hexdigest()
    final_path.with_suffix(final_path.suffix + ".sha256").write_text(digest, encoding="utf-8")
    return digest
