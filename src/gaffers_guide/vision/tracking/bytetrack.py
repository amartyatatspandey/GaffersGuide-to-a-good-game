"""ByteTrack wrapper."""

from __future__ import annotations


class ByteTrackWrapper:
    """Thin wrapper around supervision.ByteTrack."""

    def __init__(self) -> None:
        import supervision as sv

        self._tracker = sv.ByteTrack()

    def update(self, detections):
        return self._tracker.update_with_detections(detections)
