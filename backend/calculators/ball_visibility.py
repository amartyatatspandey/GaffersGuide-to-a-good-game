from __future__ import annotations

from typing import Any, Literal, Protocol

from .advanced_ball_metrics import compute_advanced_ball_metrics


class BallFrameLike(Protocol):
    ball_xy: list[float] | None
    possession_team_id: int | None


def compute_ball_visibility_ratio(frames: list[BallFrameLike]) -> float:
    """
    Share of frames with valid ball coordinates (including interpolated points).
    """
    total = len(frames)
    if total <= 0:
        return 0.0
    valid = sum(1 for frame in frames if frame.ball_xy is not None)
    return valid / float(total)


def apply_ball_metrics_gate(
    timeline: list[dict[str, Any]],
    frames: list[BallFrameLike],
    *,
    visibility_ratio: float,
    min_ball_confidence: float,
    fps: int,
    counter_attack_window_frames: int,
    press_success_window_frames: int,
) -> tuple[list[dict[str, Any]], Literal["sufficient", "insufficient"]]:
    """
    Gate ball-dependent metrics based on visibility confidence.
    """
    quality: Literal["sufficient", "insufficient"] = (
        "sufficient" if visibility_ratio >= min_ball_confidence else "insufficient"
    )
    enabled = quality == "sufficient"

    team_ids: tuple[str, str] = ("team_0", "team_1")
    possession_counts: dict[int, int] = {0: 0, 1: 0}
    valid_possession_frames = 0
    turnovers = 0
    previous_possession: int | None = None

    for row in timeline:
        possession_team_id = row.get("possession_team_id")
        if possession_team_id in (0, 1):
            valid_possession_frames += 1
            possession_counts[int(possession_team_id)] += 1
            if (
                previous_possession is not None
                and int(possession_team_id) != previous_possession
            ):
                turnovers += 1
            previous_possession = int(possession_team_id)

    team_possession_pct: dict[int, float] = {
        0: (possession_counts[0] / valid_possession_frames * 100.0)
        if valid_possession_frames > 0
        else 0.0,
        1: (possession_counts[1] / valid_possession_frames * 100.0)
        if valid_possession_frames > 0
        else 0.0,
    }

    advanced = (
        compute_advanced_ball_metrics(
            frames,
            fps=fps,
            counter_attack_window_frames=counter_attack_window_frames,
            press_success_window_frames=press_success_window_frames,
        )
        if enabled
        else None
    )

    for row in timeline:
        row["ball_visibility_ratio"] = visibility_ratio
        row["ball_data_quality"] = quality

        for team_idx, team_key in enumerate(team_ids):
            team_metrics = row.get(team_key)
            if not isinstance(team_metrics, dict):
                continue

            if enabled and advanced is not None:
                team_metrics["possession_pct"] = float(team_possession_pct[team_idx])
                team_metrics["turnovers"] = float(turnovers)
                team_metrics["high_press_success"] = float(
                    advanced[team_idx]["high_press_success"]
                )
                # Kept for backward compatibility with existing timeline schema.
                team_metrics["counter_attack_velocity"] = float(
                    advanced[team_idx]["rapid_counter_attacks"]
                )
                team_metrics["rapid_counter_attacks"] = float(
                    advanced[team_idx]["rapid_counter_attacks"]
                )
                team_metrics["lethargic_press_allowed"] = float(
                    advanced[team_idx]["lethargic_press_allowed"]
                )
                team_metrics["second_ball_won"] = float(
                    advanced[team_idx]["second_ball_won"]
                )
                team_metrics["second_ball_lost"] = float(
                    advanced[team_idx]["second_ball_lost"]
                )
                team_metrics["zone14_penetrations"] = float(
                    advanced[team_idx]["zone14_penetrations"]
                )
            else:
                team_metrics["possession_pct"] = None
                team_metrics["turnovers"] = 0.0
                team_metrics["high_press_success"] = None
                team_metrics["counter_attack_velocity"] = None
                team_metrics["rapid_counter_attacks"] = 0.0
                team_metrics["lethargic_press_allowed"] = 0.0
                team_metrics["second_ball_won"] = 0.0
                team_metrics["second_ball_lost"] = 0.0
                team_metrics["zone14_penetrations"] = 0.0

    return timeline, quality

