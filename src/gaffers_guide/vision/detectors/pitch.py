"""Pitch keypoint detector compatibility wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from gaffers_guide.vision.detectors.ball import BallDetector


@dataclass
class PitchDetector(BallDetector):
    """Placeholder pitch detector wrapper sharing current base behavior."""
