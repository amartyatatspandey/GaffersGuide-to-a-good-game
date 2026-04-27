"""
Core data structures for gaffers-guide SDK.

These types define the contracts between modules. All modules depend on core,
but core depends only on the standard library and numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BBoxDetection:
    """Single bounding box detection in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"Invalid bbox: ({self.x1},{self.y1})-({self.x2},{self.y2})"
            )

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


@dataclass
class FrameDetections:
    """All detections for a single video frame."""

    frame_idx: int
    timestamp: float
    detections: list[BBoxDetection]
    frame_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.frame_idx < 0:
            raise ValueError(f"Frame index must be >= 0, got {self.frame_idx}")
        if self.timestamp < 0:
            raise ValueError(f"Timestamp must be >= 0, got {self.timestamp}")

    def filter_by_class(self, class_name: str) -> FrameDetections:
        filtered = [d for d in self.detections if d.class_name == class_name]
        return FrameDetections(
            frame_idx=self.frame_idx,
            timestamp=self.timestamp,
            detections=filtered,
            frame_shape=self.frame_shape,
        )

    def filter_by_confidence(self, min_confidence: float) -> FrameDetections:
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return FrameDetections(
            frame_idx=self.frame_idx,
            timestamp=self.timestamp,
            detections=filtered,
            frame_shape=self.frame_shape,
        )

    @property
    def player_count(self) -> int:
        return len([d for d in self.detections if d.class_name == "player"])

    @property
    def has_ball(self) -> bool:
        return any(d.class_name == "ball" for d in self.detections)


@dataclass(frozen=True)
class PitchCoordinate:
    """2D coordinate on standardized pitch (0-105m x 0-68m)."""

    x: float
    y: float
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0 <= self.x <= 105):
            raise ValueError(f"Pitch x must be 0-105m, got {self.x}")
        if not (0 <= self.y <= 68):
            raise ValueError(f"Pitch y must be 0-68m, got {self.y}")
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    def distance_to(self, other: PitchCoordinate) -> float:
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))

    def to_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }


@dataclass
class SpatialMapping:
    """Homography transformation between pixel space and pitch space."""

    homography_matrix: NDArray[np.float64]
    pitch_corners_px: NDArray[np.float64]
    pitch_corners_meters: NDArray[np.float64]
    frame_shape: tuple[int, int]
    inverse_matrix: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        if self.homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3")
        if self.pitch_corners_px.shape != (4, 2):
            raise ValueError("Pitch corners must be 4x2 array")
        if self.pitch_corners_meters.shape != (4, 2):
            raise ValueError("Pitch meter corners must be 4x2 array")

        if self.inverse_matrix is None:
            object.__setattr__(self, "inverse_matrix", np.linalg.inv(self.homography_matrix))

    def pixel_to_pitch(self, px: tuple[float, float]) -> PitchCoordinate:
        px_homo = np.array([px[0], px[1], 1.0], dtype=np.float64)
        pitch_homo = self.homography_matrix @ px_homo
        pitch_homo /= pitch_homo[2]

        x = float(np.clip(pitch_homo[0], 0, 105))
        y = float(np.clip(pitch_homo[1], 0, 68))
        return PitchCoordinate(x=x, y=y)

    def pitch_to_pixel(self, coord: PitchCoordinate) -> tuple[float, float]:
        if self.inverse_matrix is None:
            raise ValueError("Inverse homography matrix not available")
        pitch_homo = np.array([coord.x, coord.y, 1.0], dtype=np.float64)
        px_homo = self.inverse_matrix @ pitch_homo
        px_homo /= px_homo[2]
        return (float(px_homo[0]), float(px_homo[1]))


@dataclass
class PlayerState:
    """Player state at a single timestamp."""

    track_id: int
    position_px: tuple[float, float]
    position_pitch: Optional[PitchCoordinate]
    team_id: Optional[Literal[0, 1]] = None
    bbox: Optional[BBoxDetection] = None
    jersey_number: Optional[int] = None
    velocity_mps: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "position_px": list(self.position_px),
            "position_pitch": self.position_pitch.to_dict() if self.position_pitch else None,
            "team_id": self.team_id,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "jersey_number": self.jersey_number,
            "velocity_mps": self.velocity_mps,
        }


@dataclass
class MatchState:
    """Complete match state at a single frame."""

    frame_idx: int
    timestamp: float
    players: list[PlayerState]
    ball_position_px: Optional[tuple[float, float]] = None
    ball_position_pitch: Optional[PitchCoordinate] = None
    spatial_mapping: Optional[SpatialMapping] = None
    possession_team: Optional[Literal[0, 1]] = None

    @property
    def home_players(self) -> list[PlayerState]:
        return [p for p in self.players if p.team_id == 0]

    @property
    def away_players(self) -> list[PlayerState]:
        return [p for p in self.players if p.team_id == 1]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "players": [p.to_dict() for p in self.players],
            "ball_position_px": list(self.ball_position_px) if self.ball_position_px else None,
            "ball_position_pitch": self.ball_position_pitch.to_dict()
            if self.ball_position_pitch
            else None,
            "possession_team": self.possession_team,
        }


class Detector(Protocol):
    """Protocol for all detector implementations."""

    def detect(self, frame: NDArray[np.uint8]) -> FrameDetections:
        ...


class MetricsCalculator(Protocol):
    """Protocol for tactical metrics calculators."""

    def calculate(self, states: list[MatchState]) -> dict[str, object]:
        ...
