"""Sanity checks for ``backend/data/match_test.mp4`` (workspace / pipeline dev clip)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MATCH = BACKEND_ROOT / "data" / "match_test.mp4"


@pytest.mark.skipif(
    not MATCH.is_file(), reason="match_test.mp4 not present under backend/data/"
)
def test_match_test_mp4_non_trivial_size() -> None:
    assert MATCH.stat().st_size > 1_000_000


@pytest.mark.skipif(
    not MATCH.is_file(), reason="match_test.mp4 not present under backend/data/"
)
def test_match_test_mp4_packet_count_matches_expected() -> None:
    """Expected: ~3 min @ 25 fps → 4501 packets (per ffprobe -count_packets)."""
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not on PATH")

    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(MATCH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    packets = int(proc.stdout.strip())
    assert packets == 4501
