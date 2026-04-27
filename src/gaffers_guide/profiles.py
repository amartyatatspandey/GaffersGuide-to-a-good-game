from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

QualityProfileName = Literal["fast", "balanced", "high_res", "sahi"]

VALID_PROFILES = ("fast", "balanced", "high_res", "sahi")
DEFAULT_PROFILE: QualityProfileName = "balanced"

@dataclass(frozen=True, slots=True)
class QualityProfile:
    name: QualityProfileName
    description: str
    sahi_enabled: bool
    imgsz: int
    confidence_threshold: float
    slice_width: int
    slice_height: int
    slice_overlap_ratio: float

# Keep ProfileConfig as alias for backward compatibility
ProfileConfig = QualityProfile

PROFILES: dict[str, QualityProfile] = {
    "fast": QualityProfile(
        name="fast",
        description="Prioritises speed/latency. Lower resolution, no SAHI.",
        sahi_enabled=False,
        imgsz=640,
        confidence_threshold=0.35,
        slice_width=640,
        slice_height=640,
        slice_overlap_ratio=0.0,
    ),
    "balanced": QualityProfile(
        name="balanced",
        description="Middle ground between speed and quality. Recommended for most use cases.",
        sahi_enabled=False,
        imgsz=960,
        confidence_threshold=0.30,
        slice_width=800,
        slice_height=800,
        slice_overlap_ratio=0.1,
    ),
    "high_res": QualityProfile(
        name="high_res",
        description="Higher quality output at the cost of lower FPS.",
        sahi_enabled=False,
        imgsz=1280,
        confidence_threshold=0.25,
        slice_width=960,
        slice_height=960,
        slice_overlap_ratio=0.2,
    ),
    "sahi": QualityProfile(
        name="sahi",
        description="Maximum ball recall using SAHI slicing. Highest runtime cost.",
        sahi_enabled=True,
        imgsz=1280,
        confidence_threshold=0.20,
        slice_width=640,
        slice_height=640,
        slice_overlap_ratio=0.3,
    ),
}

def resolve_profile(name: str) -> QualityProfile:
    """Resolve a profile name to a QualityProfile. Fails fast on invalid input."""
    if name not in PROFILES:
        valid = ", ".join(VALID_PROFILES)
        raise ValueError(
            f"Unknown quality profile '{name}'. Valid profiles are: {valid}"
        )
    return PROFILES[name]
