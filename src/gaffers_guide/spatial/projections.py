"""Projection helpers from pixels to pitch coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from gaffers_guide.core.types import PitchCoordinate, SpatialMapping


@dataclass
class PitchMapper:
    """Apply a fitted spatial mapping to pixel points."""

    mapping: SpatialMapping

    def to_pitch(self, x: float, y: float) -> PitchCoordinate:
        return self.mapping.pixel_to_pitch((x, y))
