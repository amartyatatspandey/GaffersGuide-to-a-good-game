from __future__ import annotations

from typing import Protocol

import numpy as np


class PlayerLike(Protocol):
    team: str
    radar_pt: list[float] | None


class FrameLike(Protocol):
    ball_xy: list[float] | None
    players: list[PlayerLike]
    possession_team_id: int | None


def team_to_id(team: str) -> int | None:
    if team == "team_0":
        return 0
    if team == "team_1":
        return 1
    return None


def compute_possession_team_id(frame: FrameLike) -> int | None:
    if frame.ball_xy is None:
        return None
    nearest_dist = float("inf")
    nearest_team_id: int | None = None
    ball = np.asarray(frame.ball_xy, dtype=np.float32)
    for player in frame.players:
        if player.radar_pt is None:
            continue
        team_id = team_to_id(player.team)
        if team_id is None:
            continue
        dist = float(
            np.linalg.norm(np.asarray(player.radar_pt, dtype=np.float32) - ball)
        )
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_team_id = team_id
    return nearest_team_id


def interpolate_ball_positions(frames: list[FrameLike], max_gap_frames: int) -> int:
    """
    Fill short missing ball tracks via linear interpolation, then backfill possession.
    """
    interpolation_count = 0
    known_idxs = [i for i, frame in enumerate(frames) if frame.ball_xy is not None]
    for left_idx, right_idx in zip(known_idxs, known_idxs[1:], strict=False):
        gap = right_idx - left_idx - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        left_xy = frames[left_idx].ball_xy
        right_xy = frames[right_idx].ball_xy
        if left_xy is None or right_xy is None:
            continue

        left = np.asarray(left_xy, dtype=np.float32)
        right = np.asarray(right_xy, dtype=np.float32)

        for k in range(1, gap + 1):
            t = k / float(gap + 1)
            interp = left + (right - left) * t
            fill_idx = left_idx + k
            if frames[fill_idx].ball_xy is None:
                frames[fill_idx].ball_xy = [float(interp[0]), float(interp[1])]
                interpolation_count += 1

    for frame in frames:
        if frame.ball_xy is not None and frame.possession_team_id is None:
            frame.possession_team_id = compute_possession_team_id(frame)

    return interpolation_count
