from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from services.parallel_pipeline import (
    VideoChunkManager,
    UnionFind,
    run_e2e_parallel,
)


def test_video_chunk_manager() -> None:
    # Use the test video in the root directory
    video_path = ROOT_DIR / "match_test.mp4"
    assert video_path.is_file(), f"Expected match_test.mp4 at {video_path}"

    # Target chunk duration = 2.0 seconds, overlap = 1.0 second
    manager = VideoChunkManager(video_path, target_chunk_duration=2.0, min_overlap=1.0)
    chunks = manager.calculate_chunks()

    assert len(chunks) > 0
    for i, c in enumerate(chunks):
        assert "chunk_idx" in c
        assert c["chunk_idx"] == i
        assert c["start_frame"] <= c["end_frame"]
        assert c["non_overlap_end"] <= c["end_frame"]

        if i < len(chunks) - 1:
            # Overlap start should be the end of the non-overlap region
            assert c["overlap_start"] == c["non_overlap_end"]
            assert c["overlap_end"] == c["end_frame"]
            assert c["overlap_end"] > c["overlap_start"]


def test_union_find() -> None:
    uf = UnionFind()
    uf.union((0, 1), (1, 2))
    uf.union((1, 2), (2, 3))
    
    # Verify transitive closure
    assert uf.find((0, 1)) == uf.find((2, 3))
    assert uf.find((0, 1)) == uf.find((1, 2))

    # Verify independent element is its own root
    assert uf.find((5, 10)) == (5, 10)


@pytest.mark.asyncio
async def test_parallel_e2e_execution() -> None:
    video_path = ROOT_DIR / "match_test.mp4"
    assert video_path.is_file()

    from unittest.mock import patch
    
    # Override VideoChunkManager.__init__ to force total_frames=500 so the test runs in seconds
    original_init = VideoChunkManager.__init__
    def patched_init(self, video_path, target_chunk_duration=180.0, min_overlap=2.0):
        original_init(self, video_path, target_chunk_duration, min_overlap)
        self.total_frames = 500  # Force limit to 500 frames

    os.environ["USE_PARALLEL"] = "true"
    output_prefix = "test_parallel_e2e"
    
    with patch.object(VideoChunkManager, "__init__", patched_init):
        report_path = await run_e2e_parallel(
            video=video_path,
            output_prefix=output_prefix,
            llm_engine="local",
            enable_zsl=False,
            target_chunk_duration=8.0,
            min_overlap=2.0,
            device="cpu", # Force CPU for testing safety
        )

    assert report_path.is_file()
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert isinstance(report_data, list)
    assert len(report_data) > 0
    
    # Verify that tracking data and metrics files are also written
    out_dir = BACKEND_DIR / "output"
    tracking_data_path = out_dir / f"{output_prefix}_tracking_data.json"
    metrics_path = out_dir / f"{output_prefix}_tactical_metrics.json"

    assert tracking_data_path.is_file()
    assert metrics_path.is_file()

    # Clean up test artifacts
    for p in [report_path, tracking_data_path, metrics_path]:
        if p.exists():
            p.unlink()
