"""
Event Intelligence Layer — Shape Detectors
==========================================

Detects collective team shape events (team-level, player_id = None):
  SHP_001  High Press Moment       defensive line x >= +20 m; pressure index <= 4.5 m
  SHP_002  Mid Block               defensive line x -5 to +15 m; area <= 900 sq m
  SHP_003  Low Block               deepest defender x <= -25 m; area <= 600 sq m
  SHP_004  Compact Shape           width <= 35 m AND length <= 40 m
  SHP_005  Stretched Shape         length >= 65 m OR width >= 55 m
  SHP_006  Overload Zone           >= 3 same-team players in 15m x 15m zone
  SHP_007  Pressing Trap Triggered coordinated press with ball forced to corner
  SHP_008  Counter-Attack Launch   >= 3 players advance >= 5 m within 3 seconds
"""
from __future__ import annotations

import math
from typing import Any

from event_layer.detectors._base import FrameList, _make_event
from event_layer.models import EventRecord
from event_layer.ontology import THRESHOLDS


class ShapeDetector:
    """Detects all team-shape-category events (SHP_001 – SHP_008)."""

    def __init__(self, fps: float, job_id: str, metrics_timeline: list[dict[str, Any]] | None = None) -> None:
        self.fps = fps
        self.job_id = job_id
        # Optional: pre-computed metrics timeline from existing tactical rule engine
        self.metrics_timeline = metrics_timeline or []

    def detect(self, frames: FrameList) -> list[EventRecord]:
        events: list[EventRecord] = []

        if self.metrics_timeline:
            # Use pre-computed metrics for efficiency (avoids re-computing spatial math)
            events.extend(self._detect_press_from_metrics())
            events.extend(self._detect_blocks_from_metrics())
            events.extend(self._detect_shape_from_metrics())

        # Frame-by-frame detection for events requiring per-frame player positions
        events.extend(self._detect_overload_zones(frames))
        events.extend(self._detect_counter_attack_launch(frames))
        return events

    # ── From Metrics Timeline ─────────────────────────────────────────────────

    def _detect_press_from_metrics(self) -> list[EventRecord]:
        """SHP_001: High Press Moment — uses pre-computed pressure_index_m and team positions."""
        events: list[EventRecord] = []
        t = THRESHOLDS
        min_frames = math.ceil(t.HIGH_PRESS_MIN_DURATION_S * self.fps)

        for team_id in ("team_0", "team_1"):
            windows: list[tuple[int, int]] = []
            in_press = False
            press_start_idx = 0

            for i, frame in enumerate(self.metrics_timeline):
                metrics = frame.get(team_id, {}) or {}
                deepest_x = float(metrics.get("deepest_x", 0.0) or 0.0)
                press_idx = float(metrics.get("pressure_index_m", 99.0) or 99.0)

                is_pressing = deepest_x >= t.HIGH_PRESS_DEF_LINE_X_M and press_idx <= t.HIGH_PRESS_PRESSURE_INDEX_M

                if is_pressing:
                    if not in_press:
                        in_press = True
                        press_start_idx = i
                else:
                    if in_press and (i - press_start_idx) >= min_frames:
                        windows.append((press_start_idx, i - 1))
                    in_press = False

            if in_press and (len(self.metrics_timeline) - 1 - press_start_idx) >= min_frames:
                windows.append((press_start_idx, len(self.metrics_timeline) - 1))

            for s_idx, e_idx in windows:
                start_frame = self.metrics_timeline[s_idx].get("frame_idx", s_idx)
                end_frame   = self.metrics_timeline[e_idx].get("frame_idx", e_idx)
                duration_s  = (end_frame - start_frame) / self.fps

                # Average metrics over window
                window_metrics = [
                    self.metrics_timeline[i].get(team_id, {}) or {}
                    for i in range(s_idx, e_idx + 1)
                ]
                avg_deepest = sum(float(m.get("deepest_x", 0) or 0) for m in window_metrics) / len(window_metrics)
                avg_press   = sum(float(m.get("pressure_index_m", 99) or 99) for m in window_metrics) / len(window_metrics)
                coverage    = sum(1 for m in window_metrics if m) / len(window_metrics)

                confidence = (
                    0.4 * min(1.0, (avg_deepest - 15.0) / 15.0)
                    + 0.4 * min(1.0, (8.0 - avg_press) / 5.0)
                    + 0.2 * coverage
                )
                importance = 0.75 + 0.25 * min(1.0, duration_s / 30.0)

                t_s = round(start_frame / self.fps, 1)
                events.append(_make_event(
                    event_type="SHP_001",
                    event_name="High Press Moment",
                    category="shape",
                    player_id=None,
                    team_id=team_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=self.fps,
                    confidence=confidence,
                    importance=importance,
                    description=(
                        f"{team_id} sustained a high press for {duration_s:.1f}s at "
                        f"{int(t_s // 60):02d}:{int(t_s % 60):02d}. "
                        f"Defensive line at avg {avg_deepest:.1f}m, pressure index {avg_press:.1f}m."
                    ),
                    job_id=self.job_id,
                    tags=["SHP_001", "high_press", team_id],
                ))

        return events

    def _detect_blocks_from_metrics(self) -> list[EventRecord]:
        """SHP_002 + SHP_003: Mid/Low Block."""
        events: list[EventRecord] = []
        t = THRESHOLDS
        min_frames = math.ceil(5.0 * self.fps)  # Blocks must be sustained 5 seconds

        for team_id in ("team_0", "team_1"):
            for block_type, x_min, x_max, area_thresh, code, name in [
                ("mid",  t.MID_BLOCK_X_MIN_M, t.MID_BLOCK_X_MAX_M, t.MID_BLOCK_AREA_M2, "SHP_002", "Mid Block"),
                ("low",  None,                t.LOW_BLOCK_X_M,      t.LOW_BLOCK_AREA_M2, "SHP_003", "Low Block"),
            ]:
                in_block = False
                block_start = 0

                for i, frame in enumerate(self.metrics_timeline):
                    metrics = frame.get(team_id, {}) or {}
                    deepest_x = float(metrics.get("deepest_x", 0.0) or 0.0)
                    area = float(metrics.get("area_sq_meters", 9999.0) or 9999.0)

                    if block_type == "mid":
                        cond = x_min <= deepest_x <= x_max and area <= area_thresh  # type: ignore[operator]
                    else:
                        cond = deepest_x <= x_max and area <= area_thresh  # type: ignore[operator]

                    if cond:
                        if not in_block:
                            in_block = True
                            block_start = i
                    else:
                        if in_block and (i - block_start) >= min_frames:
                            self._emit_block_event(team_id, block_start, i - 1, code, name, events)
                        in_block = False

                if in_block and (len(self.metrics_timeline) - 1 - block_start) >= min_frames:
                    self._emit_block_event(team_id, block_start, len(self.metrics_timeline) - 1, code, name, events)

        return events

    def _emit_block_event(
        self,
        team_id: str,
        s: int,
        e: int,
        code: str,
        name: str,
        events: list[EventRecord],
    ) -> None:
        start_frame = self.metrics_timeline[s].get("frame_idx", s)
        end_frame   = self.metrics_timeline[e].get("frame_idx", e)
        duration_s  = (end_frame - start_frame) / self.fps
        t_s = round(start_frame / self.fps, 1)
        confidence = 0.72 + 0.18 * min(1.0, duration_s / 60.0)
        importance = 0.60 + 0.40 * min(1.0, duration_s / 90.0)

        events.append(_make_event(
            event_type=code,
            event_name=name,
            category="shape",
            player_id=None,
            team_id=team_id,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=self.fps,
            confidence=confidence,
            importance=importance,
            description=(
                f"{team_id} maintained a {name.lower()} for {duration_s:.0f}s "
                f"from {int(t_s // 60):02d}:{int(t_s % 60):02d}."
            ),
            job_id=self.job_id,
            tags=[code, name.lower().replace(" ", "_"), team_id],
        ))

    def _detect_shape_from_metrics(self) -> list[EventRecord]:
        """SHP_004 + SHP_005: Compact / Stretched Shape."""
        events: list[EventRecord] = []
        t = THRESHOLDS
        min_frames = math.ceil(3.0 * self.fps)

        for team_id in ("team_0", "team_1"):
            for code, name, condition_fn in [
                ("SHP_004", "Compact Shape",
                 lambda m: float(m.get("team_width_m", 99) or 99) <= t.COMPACT_WIDTH_M
                           and float(m.get("team_length_m", 99) or 99) <= t.COMPACT_LENGTH_M),
                ("SHP_005", "Stretched Shape",
                 lambda m: float(m.get("team_length_m", 0) or 0) >= t.STRETCHED_LENGTH_M
                           or float(m.get("team_width_m", 0) or 0) >= t.STRETCHED_WIDTH_M),
            ]:
                in_shape = False
                shape_start = 0

                for i, frame in enumerate(self.metrics_timeline):
                    metrics = frame.get(team_id, {}) or {}
                    if condition_fn(metrics):
                        if not in_shape:
                            in_shape = True
                            shape_start = i
                    else:
                        if in_shape and (i - shape_start) >= min_frames:
                            self._emit_block_event(team_id, shape_start, i - 1, code, name, events)
                        in_shape = False

                if in_shape and (len(self.metrics_timeline) - 1 - shape_start) >= min_frames:
                    self._emit_block_event(team_id, shape_start, len(self.metrics_timeline) - 1, code, name, events)

        return events

    # ── Per-Frame Detection ───────────────────────────────────────────────────

    def _detect_overload_zones(self, frames: FrameList) -> list[EventRecord]:
        """SHP_006: >= 3 players from same team in a 15m x 15m zone."""
        events: list[EventRecord] = []
        zone_size = THRESHOLDS.OVERLOAD_ZONE_SIZE_M
        min_players = THRESHOLDS.OVERLOAD_MIN_PLAYERS
        cooldown = int(5.0 * self.fps)

        # Track last overload per (team_id, zone_key)
        last_overload: dict[tuple[str, str], int] = {}

        for frame in frames:
            f_idx = frame["frame_idx"]
            team_pts: dict[str, list[tuple[float, float, int]]] = {"team_0": [], "team_1": []}

            for p in frame.get("players", []):
                tid = p.get("team_id")
                x = p.get("x_pitch")
                y = p.get("y_pitch")
                pid = p.get("id", 0)
                if tid in team_pts and x is not None and y is not None:
                    team_pts[tid].append((float(x), float(y), pid))

            for team_id, pts in team_pts.items():
                if len(pts) < min_players:
                    continue

                # Grid-based zone check: for each player, count teammates within zone_size
                for i, (cx, cy, _) in enumerate(pts):
                    in_zone = [
                        pid for j, (x2, y2, pid) in enumerate(pts)
                        if i != j and abs(x2 - cx) <= zone_size and abs(y2 - cy) <= zone_size
                    ]
                    if len(in_zone) + 1 >= min_players:
                        zone_key = f"{int(cx // zone_size)}_{int(cy // zone_size)}"
                        cache_key = (team_id, zone_key)
                        if f_idx - last_overload.get(cache_key, -cooldown) >= cooldown:
                            last_overload[cache_key] = f_idx
                            t_s = round(f_idx / self.fps, 1)
                            events.append(_make_event(
                                event_type="SHP_006",
                                event_name="Overload Zone",
                                category="shape",
                                player_id=None,
                                team_id=team_id,
                                start_frame=f_idx,
                                end_frame=f_idx,
                                fps=self.fps,
                                confidence=0.75,
                                importance=0.70,
                                description=(
                                    f"{team_id} created a numerical overload "
                                    f"({len(in_zone) + 1} players in {zone_size}m zone) "
                                    f"at {int(t_s // 60):02d}:{int(t_s % 60):02d}."
                                ),
                                job_id=self.job_id,
                                start_radar_pt=(cx, cy),
                                tags=["SHP_006", "overload", team_id],
                            ))
                            break  # One overload event per team per frame
        return events

    def _detect_counter_attack_launch(self, frames: FrameList) -> list[EventRecord]:
        """SHP_008: >= 3 players advance >= 5 m within 3 seconds of gaining possession."""
        events: list[EventRecord] = []
        advance_m = THRESHOLDS.COUNTER_ADVANCE_M
        window_frames = int(THRESHOLDS.COUNTER_ADVANCE_TIME_S * self.fps)
        min_players = 3

        # Build possession change frames
        possession_changes: list[tuple[int, str]] = []
        prev_poss = None
        for frame in frames:
            curr_poss = frame.get("possession_team_id")
            if prev_poss and curr_poss and curr_poss != prev_poss:
                possession_changes.append((frame["frame_idx"], curr_poss))
            prev_poss = curr_poss

        frame_map = {f["frame_idx"]: f for f in frames}

        for change_frame, gaining_team in possession_changes:
            # Snapshot positions at change frame
            start_frame_data = frame_map.get(change_frame, {})
            start_positions: dict[int, tuple[float, float]] = {
                p["id"]: (float(p["x_pitch"]), float(p["y_pitch"]))
                for p in start_frame_data.get("players", [])
                if p.get("team_id") == gaining_team
                and p.get("id") is not None
                and p.get("x_pitch") is not None and p.get("y_pitch") is not None
            }

            # Check positions after window
            end_frame_idx = change_frame + window_frames
            end_frame_data = frame_map.get(end_frame_idx, {})
            if not end_frame_data:
                # Try nearby frames
                for delta in range(-5, 6):
                    end_frame_data = frame_map.get(end_frame_idx + delta, {})
                    if end_frame_data:
                        break

            advanced_players = [
                pid for pid, (sx, sy) in start_positions.items()
                for p in end_frame_data.get("players", [])
                if p.get("id") == pid
                and p.get("team_id") == gaining_team
                and p.get("x_pitch") is not None
                and (float(p["x_pitch"]) - sx) >= advance_m
            ]

            if len(advanced_players) >= min_players:
                t_s = round(change_frame / self.fps, 1)
                events.append(_make_event(
                    event_type="SHP_008",
                    event_name="Counter-Attack Launch",
                    category="shape",
                    player_id=None,
                    team_id=gaining_team,
                    start_frame=change_frame,
                    end_frame=end_frame_idx,
                    fps=self.fps,
                    confidence=0.78,
                    importance=0.85,
                    description=(
                        f"{gaining_team} launched a counter-attack at "
                        f"{int(t_s // 60):02d}:{int(t_s % 60):02d} — "
                        f"{len(advanced_players)} players advanced ≥ {advance_m}m in "
                        f"{THRESHOLDS.COUNTER_ADVANCE_TIME_S:.0f}s."
                    ),
                    job_id=self.job_id,
                    tags=["SHP_008", "counter_attack", gaining_team, "transition"],
                ))

        return events
