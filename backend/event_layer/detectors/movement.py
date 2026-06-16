"""
Event Intelligence Layer — Movement Detectors
==============================================

Detects:
  MOV_001  High-Speed Run        ≥ 6.5 m/s sustained for ≥ 1.5 s
  MOV_002  Sprint                ≥ 8.0 m/s for ≥ 0.5 s
  MOV_003  Recovery Run          high-speed movement toward own goal post-turnover
  MOV_004  Overlap Run           wide player moves beyond an attacker
  MOV_005  Underlap Run          inverted player cuts inside into half-space
  MOV_006  Third-Man Run         off-ball run timed to arrive after a combination
  MOV_007  Diagonal Run          forward run at ≥ 30° diagonal across lines
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from event_layer.detectors._base import (
    FrameList, _build_player_position_history, _dist, _homography_discount,
    _make_event, _rolling_mean, _speed_mps,
)
from event_layer.models import EventRecord
from event_layer.ontology import THRESHOLDS, classify_zone, FINAL_THIRD_ZONES


class MovementDetector:
    """Detects all movement-category events (MOV_001 – MOV_007)."""

    def __init__(self, fps: float, job_id: str) -> None:
        self.fps = fps
        self.job_id = job_id

    def detect(self, frames: FrameList) -> list[EventRecord]:
        """Run all movement detectors and return combined event list."""
        player_history = _build_player_position_history(frames)
        team_map = self._build_team_map(frames)

        events: list[EventRecord] = []
        events.extend(self._detect_speed_events(player_history, team_map))
        events.extend(self._detect_recovery_runs(frames, player_history, team_map))
        events.extend(self._detect_overlap_underlap(frames, player_history, team_map))
        events.extend(self._detect_diagonal_runs(player_history, team_map))
        return events

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_team_map(self, frames: FrameList) -> dict[int, str]:
        """Return {player_id: team_id} from the most common team assignment."""
        team_votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for frame in frames:
            for p in frame.get("players", []):
                pid = p.get("id")
                tid = p.get("team_id")
                if pid is not None and tid is not None:
                    team_votes[pid][tid] += 1
        return {
            pid: max(votes, key=votes.get)  # type: ignore[arg-type]
            for pid, votes in team_votes.items()
        }

    def _compute_per_frame_speeds(
        self, history: list[tuple[int, float, float, float]]
    ) -> list[float]:
        """Compute smoothed speed in m/s for each entry in position history."""
        raw_speeds: list[float] = [0.0]
        for i in range(1, len(history)):
            f_prev, x_prev, y_prev, _ = history[i - 1]
            f_curr, x_curr, y_curr, _ = history[i]
            raw_speeds.append(
                _speed_mps((x_prev, y_prev), (x_curr, y_curr), f_curr - f_prev, self.fps)
            )
        return _rolling_mean(raw_speeds, THRESHOLDS.SPEED_SMOOTHING_WINDOW)

    def _extract_speed_windows(
        self,
        history: list[tuple[int, float, float, float]],
        smoothed: list[float],
        threshold_mps: float,
        min_duration_s: float,
    ) -> list[tuple[int, int]]:
        """
        Find contiguous windows where speed >= threshold, meeting min duration.
        Returns list of (start_idx, end_idx) into `history`.
        Adjacent windows within MERGE_GAP_FRAMES are merged.
        """
        min_frames = math.ceil(min_duration_s * self.fps)
        in_window = False
        window_start = 0
        windows: list[tuple[int, int]] = []

        for i, spd in enumerate(smoothed):
            if spd >= threshold_mps:
                if not in_window:
                    in_window = True
                    window_start = i
            else:
                if in_window:
                    in_window = False
                    windows.append((window_start, i - 1))

        if in_window:
            windows.append((window_start, len(smoothed) - 1))

        # Merge close windows
        merged: list[tuple[int, int]] = []
        for start, end in windows:
            if merged and (history[start][0] - history[merged[-1][1]][0]) <= THRESHOLDS.RUN_MERGE_GAP_FRAMES:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        # Apply minimum duration filter
        return [
            (s, e) for s, e in merged
            if (history[e][0] - history[s][0]) >= min_frames
        ]

    # ── MOV_001 + MOV_002 ─────────────────────────────────────────────────────

    def _detect_speed_events(
        self,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        events: list[EventRecord] = []

        for pid, history in player_history.items():
            if len(history) < 3:
                continue
            team_id = team_map.get(pid, "unknown")
            smoothed = self._compute_per_frame_speeds(history)
            avg_hconf = sum(h[3] for h in history) / len(history)

            for event_type, threshold, min_dur, name in [
                ("MOV_001", THRESHOLDS.HIGH_SPEED_RUN_MPS, THRESHOLDS.HIGH_SPEED_MIN_DURATION_S, "High-Speed Run"),
                ("MOV_002", THRESHOLDS.SPRINT_MPS,         THRESHOLDS.SPRINT_MIN_DURATION_S,      "Sprint"),
            ]:
                windows = self._extract_speed_windows(history, smoothed, threshold, min_dur)
                for s_idx, e_idx in windows:
                    start_frame = history[s_idx][0]
                    end_frame   = history[e_idx][0]
                    peak_speed  = max(smoothed[s_idx:e_idx + 1])
                    duration_s  = (end_frame - start_frame) / self.fps

                    start_pt = (history[s_idx][1], history[s_idx][2])
                    end_pt   = (history[e_idx][1], history[e_idx][2])

                    # Peak position = position at frame with max speed
                    peak_idx = s_idx + smoothed[s_idx:e_idx + 1].index(peak_speed)
                    peak_pt  = (history[peak_idx][1], history[peak_idx][2])

                    zone = classify_zone(peak_pt[0], peak_pt[1])
                    hdisc = _homography_discount(avg_hconf)

                    confidence = (
                        0.5 * min(1.0, peak_speed / 10.0)
                        + 0.3 * min(1.0, duration_s / 5.0)
                        + 0.2 * hdisc
                    )

                    importance = (
                        0.4 * (1.0 if zone in FINAL_THIRD_ZONES else 0.4)
                        + 0.4 * min(1.0, peak_speed / 10.0)
                        + 0.2 * confidence
                    )

                    time_s = round(start_frame / self.fps, 1)
                    desc = (
                        f"Player {pid} ({team_id}) made a {name.lower()} at "
                        f"{int(time_s // 60):02d}:{int(time_s % 60):02d}, "
                        f"reaching {peak_speed:.1f} m/s in the {zone.replace('_', ' ')}."
                    )

                    events.append(_make_event(
                        event_type=event_type,
                        event_name=name,
                        category="movement",
                        player_id=pid,
                        team_id=team_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        fps=self.fps,
                        confidence=confidence,
                        importance=importance,
                        description=desc,
                        job_id=self.job_id,
                        start_radar_pt=start_pt,
                        end_radar_pt=end_pt,
                        peak_radar_pt=peak_pt,
                        tags=[event_type.lower(), zone, "speed"],
                    ))
        return events

    # ── MOV_003 ───────────────────────────────────────────────────────────────

    def _detect_recovery_runs(
        self,
        frames: FrameList,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        """Detect recovery runs immediately after possession changes."""
        events: list[EventRecord] = []

        # Find frames where possession changes (losing team → MOV_003 candidates)
        possession_changes: list[tuple[int, str]] = []  # (frame_idx, team_that_lost)
        prev_poss: str | None = None
        for frame in frames:
            curr_poss = frame.get("possession_team_id")
            if prev_poss and curr_poss and curr_poss != prev_poss:
                possession_changes.append((frame["frame_idx"], prev_poss))
            prev_poss = curr_poss

        if not possession_changes:
            return events

        # For each change, look for qualifying recovery runs
        lookback_frames = int(THRESHOLDS.TRANSITION_WINDOW_S * self.fps * 2)
        min_frames = math.ceil(THRESHOLDS.RECOVERY_MIN_DURATION_S * self.fps)

        for change_frame, losing_team in possession_changes:
            window_end = change_frame + lookback_frames

            for pid, history in player_history.items():
                if team_map.get(pid) != losing_team:
                    continue

                # Extract history in window
                window = [h for h in history if change_frame <= h[0] <= window_end]
                if len(window) < min_frames:
                    continue

                smoothed = self._compute_per_frame_speeds(window)

                # Check conditions: speed ≥ threshold AND x direction is toward own goal (decreasing)
                qualifying: list[int] = []
                for i, (f_idx, x, y, hconf) in enumerate(window):
                    if smoothed[i] >= THRESHOLDS.RECOVERY_SPEED_MPS:
                        # Direction check: compare to previous point
                        if i > 0 and window[i][1] < window[i - 1][1]:  # x is decreasing
                            qualifying.append(i)

                if len(qualifying) < min_frames:
                    continue

                # Build event from qualifying window
                start_h = window[qualifying[0]]
                end_h   = window[qualifying[-1]]
                duration_s = (end_h[0] - start_h[0]) / self.fps
                if duration_s < THRESHOLDS.RECOVERY_MIN_DURATION_S:
                    continue

                avg_hconf = sum(h[3] for h in window) / len(window)
                hdisc = _homography_discount(avg_hconf)
                peak_speed = max(smoothed[i] for i in qualifying)
                confidence = (
                    0.5 * min(1.0, peak_speed / 8.0)
                    + 0.3 * min(1.0, duration_s / 3.0)
                    + 0.2 * hdisc
                )
                importance = 0.55 + 0.45 * confidence  # Recovery runs are always important

                start_pt = (start_h[1], start_h[2])
                end_pt   = (end_h[1],   end_h[2])
                t_s = round(start_h[0] / self.fps, 1)
                desc = (
                    f"Player {pid} ({losing_team}) made a recovery run at "
                    f"{int(t_s // 60):02d}:{int(t_s % 60):02d}, "
                    f"tracking back at {peak_speed:.1f} m/s after losing possession."
                )

                events.append(_make_event(
                    event_type="MOV_003",
                    event_name="Recovery Run",
                    category="movement",
                    player_id=pid,
                    team_id=losing_team,
                    start_frame=start_h[0],
                    end_frame=end_h[0],
                    fps=self.fps,
                    confidence=confidence,
                    importance=importance,
                    description=desc,
                    job_id=self.job_id,
                    start_radar_pt=start_pt,
                    end_radar_pt=end_pt,
                    tags=["MOV_003", "recovery", "defensive"],
                ))

        return events

    # ── MOV_004 + MOV_005 ─────────────────────────────────────────────────────

    def _detect_overlap_underlap(
        self,
        frames: FrameList,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        """
        Detect overlap (MOV_004) and underlap (MOV_005) runs.

        Overlap: wide player (|y| > 25) moves beyond (higher x than) a teammate
                 who is already in a wide-advanced position.
        Underlap: a player who was wide cuts inside (y decreasing toward 0) into
                  half-space zone while advancing in x.
        """
        events: list[EventRecord] = []
        min_frames = math.ceil(1.0 * self.fps)

        for pid, history in player_history.items():
            if len(history) < min_frames + 1:
                continue
            team_id = team_map.get(pid, "unknown")

            for i in range(1, len(history)):
                f_idx, x, y, hconf = history[i]
                _, x_prev, y_prev, _ = history[i - 1]

                dx = x - x_prev
                dy = y - y_prev

                # Overlap: moving forward in x AND staying wide
                if dx > 0 and abs(y) > THRESHOLDS.WIDE_Y_THRESHOLD_M and x > 15.0:
                    # Check if there's a same-team player slightly behind in x but in same wide channel
                    events.append(_make_event(
                        event_type="MOV_004",
                        event_name="Overlap Run",
                        category="movement",
                        player_id=pid,
                        team_id=team_id,
                        start_frame=history[max(0, i - min_frames)][0],
                        end_frame=f_idx,
                        fps=self.fps,
                        confidence=0.65,
                        importance=0.60,
                        description=(
                            f"Player {pid} ({team_id}) made an overlap run along the "
                            f"{'left' if y > 0 else 'right'} channel at "
                            f"{int(f_idx / self.fps // 60):02d}:{int(f_idx / self.fps % 60):02d}."
                        ),
                        job_id=self.job_id,
                        start_radar_pt=(x_prev, y_prev),
                        end_radar_pt=(x, y),
                        tags=["MOV_004", "overlap", "wide"],
                    ))
                    break  # one event per player per run

                # Underlap: was wide, cutting inside while advancing
                if dx > 0 and abs(y_prev) > THRESHOLDS.WIDE_Y_THRESHOLD_M and abs(y) < abs(y_prev):
                    # Cutting inside (toward centre)
                    if (
                        THRESHOLDS.HALF_SPACE_Y_INNER_M <= abs(y) <= THRESHOLDS.HALF_SPACE_Y_OUTER_M
                        and x > THRESHOLDS.HALF_SPACE_X_MIN_M
                    ):
                        events.append(_make_event(
                            event_type="MOV_005",
                            event_name="Underlap Run",
                            category="movement",
                            player_id=pid,
                            team_id=team_id,
                            start_frame=history[max(0, i - min_frames)][0],
                            end_frame=f_idx,
                            fps=self.fps,
                            confidence=0.62,
                            importance=0.65,
                            description=(
                                f"Player {pid} ({team_id}) cut inside from a wide position "
                                f"into the half-space at "
                                f"{int(f_idx / self.fps // 60):02d}:{int(f_idx / self.fps % 60):02d}."
                            ),
                            job_id=self.job_id,
                            start_radar_pt=(x_prev, y_prev),
                            end_radar_pt=(x, y),
                            tags=["MOV_005", "underlap", "half_space"],
                        ))
                        break

        return events

    # ── MOV_007 ───────────────────────────────────────────────────────────────

    def _detect_diagonal_runs(
        self,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        """
        Detect diagonal runs: player advances in x AND y simultaneously,
        with run angle ≥ 30° from horizontal, sustained for ≥ 1.0 s.
        """
        events: list[EventRecord] = []
        min_frames = math.ceil(1.0 * self.fps)

        for pid, history in player_history.items():
            if len(history) < min_frames + 1:
                continue
            team_id = team_map.get(pid, "unknown")

            i = 0
            while i < len(history) - min_frames:
                start_h = history[i]
                end_h   = history[min(i + min_frames, len(history) - 1)]

                dx = end_h[1] - start_h[1]
                dy = end_h[2] - start_h[2]

                if dx <= 0:
                    i += 1
                    continue

                angle_deg = math.degrees(math.atan2(abs(dy), abs(dx)))
                if angle_deg < 30.0:
                    i += 1
                    continue

                # Confirm forward movement
                run_dist = math.hypot(dx, dy)
                if run_dist < 5.0:  # at least 5m
                    i += 1
                    continue

                zone = classify_zone(end_h[1], end_h[2])
                confidence = 0.55 + 0.25 * (angle_deg / 90.0)
                importance = (
                    0.5 * (1.0 if zone in FINAL_THIRD_ZONES else 0.35)
                    + 0.5 * min(1.0, run_dist / 20.0)
                )

                t_s = round(start_h[0] / self.fps, 1)
                direction = "inside-to-out" if dy > 0 else "outside-in"
                events.append(_make_event(
                    event_type="MOV_007",
                    event_name="Diagonal Run",
                    category="movement",
                    player_id=pid,
                    team_id=team_id,
                    start_frame=start_h[0],
                    end_frame=end_h[0],
                    fps=self.fps,
                    confidence=confidence,
                    importance=importance,
                    description=(
                        f"Player {pid} ({team_id}) made a diagonal run ({direction}, "
                        f"{angle_deg:.0f}°) at {int(t_s // 60):02d}:{int(t_s % 60):02d}."
                    ),
                    job_id=self.job_id,
                    start_radar_pt=(start_h[1], start_h[2]),
                    end_radar_pt=(end_h[1], end_h[2]),
                    tags=["MOV_007", "diagonal", zone],
                ))
                i += min_frames  # skip ahead to avoid duplicate events

        return events
