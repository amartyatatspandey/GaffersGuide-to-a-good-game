"""Homography artifact resolution for tracking (isolates calibration policy from CV math)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class HomographyProvider(Protocol):
    """Returns a validated homography JSON path for a given match video."""

    def ensure_homography_json(self, video_path: Path) -> Path: ...


def default_homography_provider() -> HomographyProvider:
    """Lazy import so importing ``services`` stays light when calibration is unused."""

    from services.homography_resolution import ensure_homography_json_for_video

    class _DefaultHomographyProvider:
        def ensure_homography_json(self, video_path: Path) -> Path:
            return ensure_homography_json_for_video(video_path)

    return _DefaultHomographyProvider()
