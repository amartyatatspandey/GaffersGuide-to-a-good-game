"""Core contracts and exceptions for gaffers-guide."""

from gaffers_guide.core.exceptions import (
    ConfigurationError,
    DetectionError,
    GaffersGuideError,
    SpatialMappingError,
)
from gaffers_guide.core.types import (
    BBoxDetection,
    Detector,
    FrameDetections,
    MatchState,
    MetricsCalculator,
    PitchCoordinate,
    PlayerState,
    SpatialMapping,
)

__all__ = [
    "BBoxDetection",
    "ConfigurationError",
    "DetectionError",
    "Detector",
    "FrameDetections",
    "GaffersGuideError",
    "MatchState",
    "MetricsCalculator",
    "PitchCoordinate",
    "PlayerState",
    "SpatialMapping",
    "SpatialMappingError",
]

