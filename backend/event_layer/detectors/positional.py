"""
Event Intelligence Layer — Positional Detectors
================================================

Detects sustained positional states (player must hold for ≥ 3 seconds):
  POS_001  Wide Positioning         |y| >= 25 m sustained
  POS_002  Half-Space Occupation    x >= 25, 12 <= |y| <= 25, sustained
  POS_003  Between-Lines            between opponent's defensive and mid lines
  POS_004  Deep Positioning         attacker retreats to x <= -10 m
  POS_005  Advanced Positioning     defender pushes to x >= +15 m
  POS_006  Pressing Trap Position   player holds forced corridor during coordinated press
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from event_layer.detectors._base import (
    FrameList, _build_player_position_history, _homography_discount, _make_event,
)
from event_layer.models import EventRecord
from event_layer.ontology import THRESHOLDS, classify_zone


class PositionalDetector:
    """Detects all positional-category events (POS_001 – POS_006)."""

    def __init__(self, fps: float, job_id: str) -> None:
        self.fps = fps
        self.job_id = job_id
        self._min_frames = math.ceil(THRESHOLDS.MIN_POSITIONAL_DURATION_S * self.fps)

    def detect(self, frames: FrameList) -> list[EventRecord]:
        player_history = _build_player_position_history(frames)
        team_map = self._build_team_map(frames)
        opp_line_by_frame = self._compute_opponent_lines(frames)

        events: list[EventRecord] = []
        events.extend(self._detect_wide(player_history, team_map))
        events.extend(self._detect_half_space(player_history, team_map))
        events.extend(self._detect_between_lines(player_history, team_map, opp_line_by_frame))
        events.extend(self._detect_deep_advanced(player_history, team_map, frames))
        return events

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_team_map(self, frames: FrameList) -> dict[int, str]:
        team_votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for frame in frames:
            for p in frame.get("players", []):
                pid = p.get("id")
                tid = p.get("team_id")
                if pid is not None and tid is not None:
                    team_votes[pid][tid] += 1
        return {pid: max(v, key=v.get) for pid, v in team_votes.items()}  # type: ignore[arg-type]

    def _compute_opponent_lines(
        self, frames: FrameList
    ) -> dict[int, dict[str, dict[str, float]]]:
        """
        For each frame, compute defensive and midfield line x positions for each team.
        Returns: {frame_idx: {"team_0": {"def_x": f, "mid_x": f}, "team_1": {...}}}
        """
        result: dict[int, dict[str, dict[str, float]]] = {}
        for frame in frames:
            f_idx = frame["frame_idx"]
            team_xs: dict[str, list[float]] = {"team_0": [], "team_1": []}
            for p in frame.get("players", []):
                tid = p.get("team_id")
                x = p.get("x_pitch")
                if tid in team_xs and x is not None:
                    team_xs[tid].append(float(x))

            frame_lines: dict[str, dict[str, float]] = {}
            for tid, xs in team_xs.items():
                if len(xs) >= 4:
                    xs_sorted = sorted(xs)
                    third = max(1, len(xs_sorted) // 3)
                    frame_lines[tid] = {
                        "def_x": float(xs_sorted[third - 1]),
                        "mid_x": float(xs_sorted[2 * third - 1]),
                    }
            result[f_idx] = frame_lines
        return result

    def _player_zone_windows(
        self,
        history: list[tuple[int, float, float, float]],
        in_zone_fn,
    ) -> list[tuple[int, int]]:
        """
        Return (start_frame, end_frame) windows where in_zone_fn(x, y) is True
        for at least self._min_frames consecutive frames.
        Gaps of <= POSITIONAL_EXIT_GAP_FRAMES frames are bridged.
        """
        windows: list[tuple[int, int]] = []
        in_zone = False
        zone_start_idx = 0
        out_count = 0

        for i, (f_idx, x, y, _) in enumerate(history):
            if in_zone_fn(x, y):
                if not in_zone:
                    in_zone = True
                    zone_start_idx = i
                out_count = 0
            else:
                if in_zone:
                    out_count += 1
                    if out_count > THRESHOLDS.POSITIONAL_EXIT_GAP_FRAMES:
                        # Close the window
                        span = i - 1 - zone_start_idx
                        if span >= self._min_frames:
                            windows.append((history[zone_start_idx][0], history[i - out_count][0]))
                        in_zone = False
                        out_count = 0

        # Close trailing window
        if in_zone:
            span = len(history) - 1 - zone_start_idx
            if span >= self._min_frames:
                windows.append((history[zone_start_idx][0], history[-1][0]))

        return windows

    # ── POS_001 ───────────────────────────────────────────────────────────────

    def _detect_wide(
        self,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        events: list[EventRecord] = []
        threshold = THRESHOLDS.WIDE_Y_THRESHOLD_M

        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            windows = self._player_zone_windows(history, lambda x, y: abs(y) >= threshold)

            for start_f, end_f in windows:
                window_h = [h for h in history if start_f <= h[0] <= end_f]
                if not window_h:
                    continue
                avg_hconf = sum(h[3] for h in window_h) / len(window_h)
                duration_s = (end_f - start_f) / self.fps
                avg_y = sum(h[2] for h in window_h) / len(window_h)
                side = "left" if avg_y > 0 else "right"

                confidence = (
                    0.5 * min(1.0, (sum(1 for h in window_h if abs(h[2]) >= threshold) / len(window_h)))
                    + 0.3 * min(1.0, duration_s / 10.0)
                    + 0.2 * _homography_discount(avg_hconf)
                )
                importance = (
                    0.4 * min(1.0, duration_s / 20.0)
                    + 0.4 * (1.0 if any(h[1] >= 25 for h in window_h) else 0.5)
                    + 0.2 * confidence
                )

                mid_h = window_h[len(window_h) // 2]
                t_s = round(start_f / self.fps, 1)
                events.append(_make_event(
                    event_type="POS_001",
                    event_name="Wide Positioning",
                    category="positional",
                    player_id=pid,
                    team_id=team_id,
                    start_frame=start_f,
                    end_frame=end_f,
                    fps=self.fps,
                    confidence=confidence,
                    importance=importance,
                    description=(
                        f"Player {pid} ({team_id}) held a wide {side} position for "
                        f"{duration_s:.1f}s from {int(t_s // 60):02d}:{int(t_s % 60):02d}."
                    ),
                    job_id=self.job_id,
                    start_radar_pt=(window_h[0][1], window_h[0][2]),
                    end_radar_pt=(window_h[-1][1], window_h[-1][2]),
                    peak_radar_pt=(mid_h[1], mid_h[2]),
                    tags=["POS_001", "wide", side],
                ))
        return events

    # ── POS_002 ───────────────────────────────────────────────────────────────

    def _detect_half_space(
        self,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        events: list[EventRecord] = []

        def in_half_space(x: float, y: float) -> bool:
            return (
                x >= THRESHOLDS.HALF_SPACE_X_MIN_M
                and THRESHOLDS.HALF_SPACE_Y_INNER_M <= abs(y) <= THRESHOLDS.HALF_SPACE_Y_OUTER_M
            )

        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            windows = self._player_zone_windows(history, in_half_space)

            for start_f, end_f in windows:
                window_h = [h for h in history if start_f <= h[0] <= end_f]
                if not window_h:
                    continue
                avg_hconf = sum(h[3] for h in window_h) / len(window_h)
                duration_s = (end_f - start_f) / self.fps
                avg_y = sum(h[2] for h in window_h) / len(window_h)
                side = "left" if avg_y > 0 else "right"

                # Higher importance because half-space is the most dangerous zone
                confidence = 0.5 + 0.3 * min(1.0, duration_s / 10.0) + 0.2 * _homography_discount(avg_hconf)
                importance = 0.60 + 0.20 * min(1.0, duration_s / (10 * self.fps)) + 0.20 * confidence

                t_s = round(start_f / self.fps, 1)
                events.append(_make_event(
                    event_type="POS_002",
                    event_name="Half-Space Occupation",
                    category="positional",
                    player_id=pid,
                    team_id=team_id,
                    start_frame=start_f,
                    end_frame=end_f,
                    fps=self.fps,
                    confidence=confidence,
                    importance=importance,
                    description=(
                        f"Player {pid} ({team_id}) occupied the {side} half-space for "
                        f"{duration_s:.1f}s from {int(t_s // 60):02d}:{int(t_s % 60):02d}. "
                        f"This is a high-threat receiving zone."
                    ),
                    job_id=self.job_id,
                    start_radar_pt=(window_h[0][1], window_h[0][2]),
                    end_radar_pt=(window_h[-1][1], window_h[-1][2]),
                    tags=["POS_002", "half_space", side, "danger_zone"],
                ))
        return events

    # ── POS_003: Between-Lines ────────────────────────────────────────────────

    def _detect_between_lines(
        self,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
        opp_line_by_frame: dict[int, dict[str, dict[str, float]]],
    ) -> list[EventRecord]:
        events: list[EventRecord] = []
        min_consecutive = self._min_frames

        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            opp_team = "team_1" if team_id == "team_0" else "team_0"

            consecutive: list[tuple[int, float, float]] = []

            for f_idx, x, y, hconf in history:
                lines = opp_line_by_frame.get(f_idx, {}).get(opp_team, {})
                def_x = lines.get("def_x")
                mid_x = lines.get("mid_x")

                if def_x is None or mid_x is None:
                    if consecutive:
                        self._close_between_lines(pid, team_id, consecutive, events)
                    consecutive = []
                    continue

                lo = min(def_x, mid_x)
                hi = max(def_x, mid_x)

                if lo <= x <= hi:
                    consecutive.append((f_idx, x, y))
                else:
                    if len(consecutive) >= min_consecutive:
                        self._close_between_lines(pid, team_id, consecutive, events)
                    consecutive = []

            if len(consecutive) >= min_consecutive:
                self._close_between_lines(pid, team_id, consecutive, events)

        return events

    def _close_between_lines(
        self,
        pid: int,
        team_id: str,
        window: list[tuple[int, float, float]],
        events: list[EventRecord],
    ) -> None:
        start_f = window[0][0]
        end_f   = window[-1][0]
        duration_s = (end_f - start_f) / self.fps
        t_s = round(start_f / self.fps, 1)
        confidence = 0.55 + 0.25 * min(1.0, duration_s / 10.0)
        importance = 0.65 + 0.35 * min(1.0, duration_s / 15.0)

        events.append(_make_event(
            event_type="POS_003",
            event_name="Between-Lines",
            category="positional",
            player_id=pid,
            team_id=team_id,
            start_frame=start_f,
            end_frame=end_f,
            fps=self.fps,
            confidence=confidence,
            importance=importance,
            description=(
                f"Player {pid} ({team_id}) held a between-lines position for "
                f"{duration_s:.1f}s from {int(t_s // 60):02d}:{int(t_s % 60):02d}."
            ),
            job_id=self.job_id,
            start_radar_pt=(window[0][1], window[0][2]),
            end_radar_pt=(window[-1][1], window[-1][2]),
            tags=["POS_003", "between_lines"],
        ))

    # ── POS_004 + POS_005 ─────────────────────────────────────────────────────

    def _detect_deep_advanced(
        self,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
        frames: FrameList,
    ) -> list[EventRecord]:
        """
        POS_004: Attacker drops deep to own half (x <= -10 m)
        POS_005: Defender pushes beyond centre (x >= +15 m)
        """
        events: list[EventRecord] = []

        # Rough role classification: players who spend most time in defensive half → defenders
        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            mean_x = sum(h[1] for h in history) / len(history)
            is_typical_attacker = mean_x > 5.0
            is_typical_defender = mean_x < -5.0

            # POS_004: Attacker moving deep
            if is_typical_attacker:
                windows = self._player_zone_windows(history, lambda x, y: x <= -10.0)
                for start_f, end_f in windows:
                    dur = (end_f - start_f) / self.fps
                    t_s = round(start_f / self.fps, 1)
                    events.append(_make_event(
                        event_type="POS_004",
                        event_name="Deep Positioning",
                        category="positional",
                        player_id=pid,
                        team_id=team_id,
                        start_frame=start_f,
                        end_frame=end_f,
                        fps=self.fps,
                        confidence=0.65,
                        importance=0.55,
                        description=(
                            f"Player {pid} ({team_id}) dropped deep into own half for "
                            f"{dur:.1f}s from {int(t_s // 60):02d}:{int(t_s % 60):02d}."
                        ),
                        job_id=self.job_id,
                        tags=["POS_004", "deep", "press_evasion"],
                    ))

            # POS_005: Defender pushing high
            if is_typical_defender:
                windows = self._player_zone_windows(history, lambda x, y: x >= 15.0)
                for start_f, end_f in windows:
                    dur = (end_f - start_f) / self.fps
                    t_s = round(start_f / self.fps, 1)
                    events.append(_make_event(
                        event_type="POS_005",
                        event_name="Advanced Positioning",
                        category="positional",
                        player_id=pid,
                        team_id=team_id,
                        start_frame=start_f,
                        end_frame=end_f,
                        fps=self.fps,
                        confidence=0.70,
                        importance=0.65,
                        description=(
                            f"Defender {pid} ({team_id}) pushed to an advanced position "
                            f"(high defensive line indicator) for {dur:.1f}s from "
                            f"{int(t_s // 60):02d}:{int(t_s % 60):02d}."
                        ),
                        job_id=self.job_id,
                        tags=["POS_005", "high_line", "advanced_defender"],
                    ))

        return events
