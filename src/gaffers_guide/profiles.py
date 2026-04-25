# src/gaffers_guide/profiles.py
"""
Quality/speed profile system for Gaffer's Guide.
Single source of truth for all runtime profile parameters.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["fast", "balanced", "high_res", "sahi"]

VALID_PROFILES: tuple[str, ...] = ("fast", "balanced", "high_res", "sahi")


@dataclass(frozen=True)
class ProfileConfig:
    """
    Immutable runtime configuration resolved from a named profile.
    Passed through the pipeline; never stored as a global.
    """
    name: str
    imgsz: int                  # YOLO inference image size
    conf_threshold: float       # Detection confidence threshold
    sahi_enabled: bool          # Whether to use SAHI tiling
    sahi_slice_size: int        # Tile size (pixels) — ignored if sahi_enabled=False
    sahi_overlap_ratio: float   # Tile overlap ratio — ignored if sahi_enabled=False
    frame_skip: int             # Process every Nth frame (1 = no skipping)

    def __str__(self) -> str:
        return (
            f"Profile '{self.name}': imgsz={self.imgsz}, conf={self.conf_threshold}, "
            f"sahi={self.sahi_enabled}, slice={self.sahi_slice_size}, "
            f"overlap={self.sahi_overlap_ratio}, frame_skip={self.frame_skip}"
        )


# ── Single source of truth for all profile parameters ──────────────────────
_PROFILES: dict[str, ProfileConfig] = {
    "fast": ProfileConfig(
        name="fast",
        imgsz=480,
        conf_threshold=0.35,
        sahi_enabled=False,
        sahi_slice_size=320,
        sahi_overlap_ratio=0.1,
        frame_skip=3,
    ),
    "balanced": ProfileConfig(
        name="balanced",
        imgsz=640,
        conf_threshold=0.25,
        sahi_enabled=False,
        sahi_slice_size=320,
        sahi_overlap_ratio=0.2,
        frame_skip=1,
    ),
    "high_res": ProfileConfig(
        name="high_res",
        imgsz=1280,
        conf_threshold=0.20,
        sahi_enabled=False,
        sahi_slice_size=512,
        sahi_overlap_ratio=0.2,
        frame_skip=1,
    ),
    "sahi": ProfileConfig(
        name="sahi",
        imgsz=1280,
        conf_threshold=0.20,
        sahi_enabled=True,
        sahi_slice_size=512,
        sahi_overlap_ratio=0.25,
        frame_skip=1,
    ),
}


def resolve_profile(name: str) -> ProfileConfig:
    """
    Resolve a profile name to its ProfileConfig.
    Fails fast with a clear ValueError on invalid input.

    Args:
        name: One of 'fast', 'balanced', 'high_res', 'sahi'

    Returns:
        Immutable ProfileConfig for the requested profile.

    Raises:
        ValueError: If name is not a recognised profile.
    """
    if name not in _PROFILES:
        raise ValueError(
            f"Unknown quality profile: '{name}'. "
            f"Valid choices are: {', '.join(VALID_PROFILES)}"
        )
    return _PROFILES[name]