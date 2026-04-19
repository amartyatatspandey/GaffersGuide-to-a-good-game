"""Narrow protocols for pitch / homography (Rule 4.1).

Vendor ``SoccerPitch`` from sn-calibration satisfies :class:`PitchLineModel` structurally.
Use these types in calibrator modules to avoid leaking upstream class names through the codebase.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class PitchLineModel(Protocol):
    """Minimal surface used by line-match and homography helpers."""

    line_extremities_keys: Mapping[str, tuple[str, str]]
    point_dict: Mapping[str, Any]

    def get_2d_homogeneous_line(self, key: str) -> Any | None: ...


@runtime_checkable
class HomographyEstimatorPort(Protocol):
    """Callable port for line-based homography from extremities (pitch plane -> image)."""

    def estimate_from_extremities(
        self,
        extremities: dict[str, Any],
        width: int,
        height: int,
    ) -> Any | None: ...
