from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Window:
    start_frame: int
    end_frame: int


def select_windows(
    event_frames: list[int],
    *,
    total_frames: int,
    fps: int,
    quality_mode: str,
) -> list[Window]:
    if not event_frames:
        return [Window(start_frame=0, end_frame=max(0, total_frames - 1))]
    radius_seconds = 2 if quality_mode == "fast" else 4
    radius = max(1, fps * radius_seconds)
    windows: list[Window] = []
    for f in event_frames:
        windows.append(
            Window(
                start_frame=max(0, f - radius),
                end_frame=min(total_frames - 1, f + radius),
            )
        )
    windows.sort(key=lambda w: (w.start_frame, w.end_frame))
    merged: list[Window] = []
    for win in windows:
        if not merged or win.start_frame > merged[-1].end_frame + 1:
            merged.append(win)
            continue
        merged[-1].end_frame = max(merged[-1].end_frame, win.end_frame)
    return merged
