from __future__ import annotations

import json
from pathlib import Path

from services.merge import atomic_write_jsonl
from services.splitter import compute_max_chunk_frames


def test_compute_max_chunk_frames_respects_memory_budget() -> None:
    frames = compute_max_chunk_frames(
        cgroup_limit_bytes=1024 * 1024 * 1024,
        width=1920,
        height=1080,
        safety_factor=0.5,
    )
    assert frames > 0
    assert frames < 1024


def test_atomic_write_jsonl_writes_sha_sidecar(tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"
    digest = atomic_write_jsonl(out, iter([{"frame_idx": 1}, {"frame_idx": 2}]))
    assert out.exists()
    sidecar = out.with_suffix(".jsonl.sha256")
    assert sidecar.exists()
    assert sidecar.read_text(encoding="utf-8") == digest
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
