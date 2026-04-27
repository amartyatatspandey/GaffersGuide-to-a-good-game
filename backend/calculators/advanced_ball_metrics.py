from __future__ import annotations

import numpy as np
from typing import Protocol


class PlayerLike(Protocol):
    team: str
    radar_pt: list[float] | None


class FrameLike(Protocol):
    possession_team_id: int | None
    ball_xy: list[float] | None
    players: list[PlayerLike]


def team_forward_progress(team_id: int, start_x: float, end_x: float) -> float:
    """Return positive forward progression for a team along radar X."""
    if team_id == 0:
        return end_x - start_x
    return start_x - end_x


def is_defensive_third(team_id: int, ball_x: float) -> bool:
    """Check whether ball is in a team's defensive third on 1050x680 radar."""
    if team_id == 0:
        return ball_x <= 350.0
    return ball_x >= 700.0


def zone14_bounds_for_team(team_id: int) -> tuple[float, float, float, float]:
    """
    Return Zone 14 bounds as (x_min, x_max, y_min, y_max) for attacking direction.
    """
    # Central pocket outside the opposition penalty area on 1050x680 radar.
    y_min, y_max = 255.0, 425.0
    if team_id == 0:
        return (780.0, 885.0, y_min, y_max)
    return (165.0, 270.0, y_min, y_max)


def in_zone14(team_id: int, ball_x: float, ball_y: float) -> bool:
    x_min, x_max, y_min, y_max = zone14_bounds_for_team(team_id)
    return x_min <= ball_x <= x_max and y_min <= ball_y <= y_max


def _nearest_player_distance(
    frame: FrameLike, *, team_id: int, ball_xy: list[float]
) -> float | None:
    """
    Euclidean distance from ball to nearest player on a team for one frame.
    """
    team_name = "team_0" if team_id == 0 else "team_1"
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    dists: list[float] = []
    for player in frame.players:
        if player.team != team_name or player.radar_pt is None:
            continue
        px, py = float(player.radar_pt[0]), float(player.radar_pt[1])
        dists.append(float(np.hypot(px - bx, py - by)))
    if not dists:
        return None
    return min(dists)


def compute_advanced_ball_metrics(
    frames: list[FrameLike],
    *,
    fps: int,
    counter_attack_window_frames: int,
    press_success_window_frames: int,
) -> dict[int, dict[str, float]]:
    """
    Compute advanced ball-dependent tactical metrics from refined frame stream.
    """
    stats: dict[int, dict[str, float]] = {
        0: {
            "rapid_counter_attacks": 0.0,
            "high_press_success": 0.0,
            "lethargic_press_allowed": 0.0,
            "second_ball_won": 0.0,
            "second_ball_lost": 0.0,
            "zone14_penetrations": 0.0,
        },
        1: {
            "rapid_counter_attacks": 0.0,
            "high_press_success": 0.0,
            "lethargic_press_allowed": 0.0,
            "second_ball_won": 0.0,
            "second_ball_lost": 0.0,
            "zone14_penetrations": 0.0,
        },
    }

    n = len(frames)
    if n <= 1:
        return stats

    # --- Rangnick 8-second counter attacks & pressing events from turnovers.
    lethargic_state: dict[int, dict[str, float | int | None]] = {
        0: {"streak": 0, "start_x": None},
        1: {"streak": 0, "start_x": None},
    }

    for i in range(1, n):
        prev_team = frames[i - 1].possession_team_id
        curr_team = frames[i].possession_team_id
        curr_ball = frames[i].ball_xy
        if curr_ball is None:
            continue

        ball_x = float(curr_ball[0])

        if (
            prev_team in (0, 1)
            and curr_team in (0, 1)
            and prev_team != curr_team
        ):
            win_team = int(curr_team)
            start_ball = frames[i].ball_xy
            end_idx = min(n - 1, i + counter_attack_window_frames)
            if start_ball is not None and frames[end_idx].ball_xy is not None:
                start_x = float(start_ball[0])
                end_x = float(frames[end_idx].ball_xy[0])  # type: ignore[index]
                if team_forward_progress(win_team, start_x, end_x) > 35.0:
                    stats[win_team]["rapid_counter_attacks"] += 1.0

            # Press success: prior pressing team is the one not in possession.
            pressing_team = int(prev_team)
            prior_ball_xy = (
                frames[i - 1].ball_xy if frames[i - 1].ball_xy is not None else curr_ball
            )
            nearest = _nearest_player_distance(
                frames[i - 1], team_id=pressing_team, ball_xy=prior_ball_xy
            )
            if nearest is not None and nearest < 2.5:
                lookahead_end = min(n - 1, i + press_success_window_frames)
                success = False
                for j in range(i, lookahead_end + 1):
                    p_team = (
                        frames[j - 1].possession_team_id if j > 0 else None
                    )
                    c_team = frames[j].possession_team_id
                    if (
                        p_team in (0, 1)
                        and c_team in (0, 1)
                        and p_team != c_team
                        and c_team == pressing_team
                    ):
                        success = True
                        break
                if success:
                    stats[pressing_team]["high_press_success"] += 1.0

        # Lethargic press allowed while opponent builds in defensive third.
        if curr_team in (0, 1):
            poss_team = int(curr_team)
            pressing_team = 1 - poss_team
            if is_defensive_third(poss_team, ball_x):
                nearest_press = _nearest_player_distance(
                    frames[i], team_id=pressing_team, ball_xy=curr_ball
                )
                state = lethargic_state[pressing_team]
                if nearest_press is not None and nearest_press > 3.0:
                    if state["streak"] == 0:
                        state["start_x"] = ball_x
                    state["streak"] = int(state["streak"]) + 1
                else:
                    state["streak"] = 0
                    state["start_x"] = None

                if int(state["streak"]) > 100 and state["start_x"] is not None:
                    start_x = float(state["start_x"])
                    forward = team_forward_progress(poss_team, start_x, ball_x)
                    if forward > 10.0:
                        stats[pressing_team][
                            "lethargic_press_allowed"
                        ] += 1.0
                        state["streak"] = 0
                        state["start_x"] = None
            else:
                state = lethargic_state[pressing_team]
                state["streak"] = 0
                state["start_x"] = None

    # --- Direct play second-ball win rate.
    i = 1
    while i < n:
        curr_team = frames[i].possession_team_id
        prev_team = frames[i - 1].possession_team_id
        if curr_team not in (0, 1) or prev_team != curr_team:
            i += 1
            continue

        start_ball = frames[i - 1].ball_xy
        if start_ball is None:
            i += 1
            continue

        max_j = min(n - 1, i + fps)
        long_pass_idx: int | None = None
        for j in range(i, max_j + 1):
            if (
                frames[j].possession_team_id != curr_team
                or frames[j].ball_xy is None
            ):
                break
            dist = float(
                np.hypot(
                    float(frames[j].ball_xy[0]) - float(start_ball[0]),
                    float(frames[j].ball_xy[1]) - float(start_ball[1]),
                )
            )
            if dist > 30.0:
                long_pass_idx = j
                break
        if long_pass_idx is None:
            i += 1
            continue

        stabilize_end = min(n - 1, long_pass_idx + 20)
        stable_idx: int | None = None
        for j in range(long_pass_idx + 1, stabilize_end + 1):
            if frames[j - 1].ball_xy is None or frames[j].ball_xy is None:
                continue
            step = float(
                np.hypot(
                    float(frames[j].ball_xy[0]) - float(frames[j - 1].ball_xy[0]),
                    float(frames[j].ball_xy[1]) - float(frames[j - 1].ball_xy[1]),
                )
            )
            if step < 5.0:
                stable_idx = j
                break

        if stable_idx is not None:
            final_team = frames[stable_idx].possession_team_id
            init_team = int(curr_team)
            if final_team == init_team:
                stats[init_team]["second_ball_won"] += 1.0
            elif final_team in (0, 1) and final_team != init_team:
                stats[init_team]["second_ball_lost"] += 1.0

        i = long_pass_idx + 1

    # --- Zone 14 penetrations by distinct possession sequences.
    for team_id in (0, 1):
        in_zone_seq = False
        for frame in frames:
            if frame.possession_team_id != team_id or frame.ball_xy is None:
                in_zone_seq = False
                continue
            bx, by = float(frame.ball_xy[0]), float(frame.ball_xy[1])
            if in_zone14(team_id, bx, by):
                if not in_zone_seq:
                    stats[team_id]["zone14_penetrations"] += 1.0
                    in_zone_seq = True
            else:
                in_zone_seq = False

    return stats

