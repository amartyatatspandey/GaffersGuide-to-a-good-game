# src/gaffers_guide/profiles.py

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
    sahi_slice_size: int        # Tile size (pixels)
    sahi_overlap_ratio: float   # Tile overlap ratio
    frame_skip: int             # Process every Nth frame (1 = no skipping)
    batch_size: int             # Inference batch size

    def __str__(self) -> str:
        return (
            f"Profile '{self.name}': "
            f"imgsz={self.imgsz}, conf={self.conf_threshold}, "
            f"sahi={self.sahi_enabled}, slice={self.sahi_slice_size}, "
            f"overlap={self.sahi_overlap_ratio}, frame_skip={self.frame_skip}, "
            f"batch_size={self.batch_size}"
        )


# ── Single source of truth ───────────────────────────────────────────────
_PROFILES: dict[str, ProfileConfig] = {
    "fast": ProfileConfig(
        name="fast",
        imgsz=480,
        conf_threshold=0.35,
        sahi_enabled=False,
        sahi_slice_size=320,
        sahi_overlap_ratio=0.1,
        frame_skip=3,
        batch_size=16,
    ),
    "balanced": ProfileConfig(
        name="balanced",
        imgsz=640,
        conf_threshold=0.25,
        sahi_enabled=False,
        sahi_slice_size=320,
        sahi_overlap_ratio=0.2,
        frame_skip=1,
        batch_size=8,
    ),
    "high_res": ProfileConfig(
        name="high_res",
        imgsz=1280,
        conf_threshold=0.20,
        sahi_enabled=False,
        sahi_slice_size=512,
        sahi_overlap_ratio=0.2,
        frame_skip=1,
        batch_size=4,
    ),
    "sahi": ProfileConfig(
        name="sahi",
        imgsz=1280,
        conf_threshold=0.20,
        sahi_enabled=True,
        sahi_slice_size=512,
        sahi_overlap_ratio=0.25,
        frame_skip=1,
        batch_size=2,
    ),
}


def resolve_profile(name: str) -> ProfileConfig:
    if name not in _PROFILES:
        raise ValueError(
            f"Unknown quality profile: '{name}'. "
            f"Valid choices are: {', '.join(VALID_PROFILES)}"
        )
    return _PROFILES[name]