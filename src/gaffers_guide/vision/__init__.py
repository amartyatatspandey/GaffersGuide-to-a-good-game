"""Vision module with lazy detector imports."""

from __future__ import annotations

from typing import Any

__all__ = ["BallDetector", "PlayerDetector", "PitchDetector", "ByteTrackWrapper"]


def __getattr__(name: str) -> Any:
    if name == "BallDetector":
        from gaffers_guide.vision.detectors.ball import BallDetector

        return BallDetector
    if name == "PlayerDetector":
        from gaffers_guide.vision.detectors.player import PlayerDetector

        return PlayerDetector
    if name == "PitchDetector":
        from gaffers_guide.vision.detectors.pitch import PitchDetector

        return PitchDetector
    if name == "ByteTrackWrapper":
        from gaffers_guide.vision.tracking.bytetrack import ByteTrackWrapper

        return ByteTrackWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

