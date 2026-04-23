from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalSearchRegion:
    xyxy: tuple[int, int, int, int]
    radius_px: int


class TemporalBallPrior:
    """Track last known ball center and build adaptive search windows."""

    def __init__(
        self,
        *,
        base_radius_px: int = 160,
        max_radius_px: int = 520,
        miss_expand_step_px: int = 40,
        confidence_reset: float = 0.55,
    ) -> None:
        self._base_radius = max(16, int(base_radius_px))
        self._max_radius = max(self._base_radius, int(max_radius_px))
        self._expand_step = max(1, int(miss_expand_step_px))
        self._confidence_reset = float(confidence_reset)
        self._last_ball_xy: tuple[float, float] | None = None
        self._miss_streak = 0

    @property
    def last_ball_xy(self) -> tuple[float, float] | None:
        return self._last_ball_xy

    def on_detection(self, center_xy: tuple[float, float], confidence: float) -> None:
        self._last_ball_xy = center_xy
        if confidence >= self._confidence_reset:
            self._miss_streak = 0

    def on_miss(self) -> None:
        self._miss_streak += 1

    def current_radius_px(self) -> int:
        expanded = self._base_radius + self._expand_step * self._miss_streak
        return min(self._max_radius, expanded)

    def search_region(
        self, frame_w: int, frame_h: int
    ) -> TemporalSearchRegion | None:
        if self._last_ball_xy is None:
            return None
        cx, cy = self._last_ball_xy
        radius = self.current_radius_px()
        x1 = max(0, int(cx) - radius)
        y1 = max(0, int(cy) - radius)
        x2 = min(frame_w, int(cx) + radius)
        y2 = min(frame_h, int(cy) + radius)
        if x2 <= x1 or y2 <= y1:
            return None
        return TemporalSearchRegion(xyxy=(x1, y1, x2, y2), radius_px=radius)

