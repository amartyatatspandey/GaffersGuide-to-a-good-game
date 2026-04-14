from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.merge import merge_chunk_rows
from services.splitter import build_chunks


def test_chunk_split_and_merge_deterministic() -> None:
    chunks = build_chunks(total_frames=1200, chunking_policy="fixed", fps=25)
    assert len(chunks) >= 1
    rows = [
        [{"frame_idx": 2, "value": "a"}, {"frame_idx": 1, "value": "b"}],
        [{"frame_idx": 2, "value": "c"}, {"frame_idx": 3, "value": "d"}],
    ]
    merged = merge_chunk_rows(rows)
    assert [int(r["frame_idx"]) for r in merged.frames] == [1, 2, 3]
    assert merged.frames[1]["value"] == "c"
