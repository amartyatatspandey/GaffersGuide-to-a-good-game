"""Pitch geometry helpers."""

from __future__ import annotations

from gaffers_guide.core.types import PitchCoordinate


def in_pitch_bounds(coord: PitchCoordinate) -> bool:
    """Return True if coordinate lies on the standard 105x68 pitch."""
    return 0.0 <= coord.x <= 105.0 and 0.0 <= coord.y <= 68.0
