"""Player detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from gaffers_guide.vision.detectors.ball import BallDetector


@dataclass
class PlayerDetector(BallDetector):
    """Currently shares YOLO wrapper mechanics with BallDetector."""
