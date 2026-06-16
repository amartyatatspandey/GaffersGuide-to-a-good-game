"""
Event Intelligence Layer — Threat Detectors
============================================

Detects the highest-value individual threat events:
  THR_001  Dangerous Run          forward run into final third, uncontested
  THR_002  Final-Third Entry      any player crosses x = +35 m forward
  THR_003  Box Entry              player enters penalty box
  THR_004  Transition Involvement player active in chain within 5s of transition
  THR_005  Dangerous Reception    receives near ball in final third with ≥ 2 m space
  THR_006  Channel Exploitation   runs through CB-FB corridor uncontested
  THR_007  Isolated Defender      1v1 wide with ≥ 3 m defender support separation
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from event_layer.detectors._base import (
    FrameList, _build_player_position_history, _dist, _homography_discount,
    _make_event, _speed_mps,
)
from event_layer.models import EventRecord
from event_layer.ontology import THRESHOLDS, classify_zone, FINAL_THIRD_ZONES


class ThreatDetector:
    """Detects all threat-category events (THR_001 – THR_007)."""

    def __init__(self, fps: float, job_id: str) -> None:
        self.fps = fps
        self.job_id = job_id

    def detect(self, frames: FrameList) -> list[EventRecord]:
        player_history = _build_player_position_history(frames)
        team_map = self._build_team_map(frames)

        events: list[EventRecord] = []
        events.extend(self._detect_final_third_entries(player_history, team_map))
        events.extend(self._detect_box_entries(player_history, team_map))
        events.extend(self._detect_dangerous_runs(frames, player_history, team_map))
        events.extend(self._detect_channel_exploitations(frames, player_history, team_map))
        events.extend(self._detect_isolated_defender(frames, player_history, team_map))
        events.extend(self._detect_transition_involvement(frames, player_history, team_map))
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

    def _defenders_in_frame(self, frame: dict, attacking_team: str) -> list[tuple[float, float]]:
        """Return list of (x, y) for all opponents in frame."""
        defending_team = "team_1" if attacking_team == "team_0" else "team_0"
        pts = []
        for p in frame.get("players", []):
            if p.get("team_id") == defending_team:
                x = p.get("x_pitch")
                y = p.get("y_pitch")
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
        return pts

    def _min_defender_dist(
        self, pt: tuple[float, float], defenders: list[tuple[float, float]]
    ) -> float:
        if not defenders:
            return 99.0
        return min(_dist(pt, d) for d in defenders)

    # ── THR_002: Final-Third Entry ────────────────────────────────────────────

    def _detect_final_third_entries(
        self,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        events: list[EventRecord] = []
        threshold_x = THRESHOLDS.FINAL_THIRD_X_M
        cooldown_frames = int(15.0 * self.fps)  # 15s cooldown per player

        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            last_event_frame = -cooldown_frames

            for i in range(1, len(history)):
                f_idx, x, y, hconf = history[i]
                _, x_prev, _, _ = history[i - 1]

                if x_prev < threshold_x <= x:  # Crossing threshold forward
                    if f_idx - last_event_frame > cooldown_frames:
                        last_event_frame = f_idx
                        zone = classify_zone(x, y)
                        hdisc = _homography_discount(hconf)
                        confidence = 0.80 * hdisc
                        importance = (
                            0.5 * (1.0 if zone in {"box", "left_channel", "right_channel", "half_space_left", "half_space_right"} else 0.6)
                            + 0.3 * confidence
                            + 0.2 * min(1.0, abs(x - threshold_x) / 10.0)
                        )
                        t_s = round(f_idx / self.fps, 1)
                        events.append(_make_event(
                            event_type="THR_002",
                            event_name="Final-Third Entry",
                            category="threat",
                            player_id=pid,
                            team_id=team_id,
                            start_frame=f_idx,
                            end_frame=f_idx,
                            fps=self.fps,
                            confidence=confidence,
                            importance=importance,
                            description=(
                                f"Player {pid} ({team_id}) entered the final third at "
                                f"{int(t_s // 60):02d}:{int(t_s % 60):02d} "
                                f"(zone: {zone.replace('_', ' ')})."
                            ),
                            job_id=self.job_id,
                            start_radar_pt=(x_prev, history[i - 1][2]),
                            end_radar_pt=(x, y),
                            tags=["THR_002", "final_third_entry", zone],
                        ))
        return events

    # ── THR_003: Box Entry ────────────────────────────────────────────────────

    def _detect_box_entries(
        self,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        events: list[EventRecord] = []
        cooldown_frames = int(10.0 * self.fps)

        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            last_event_frame = -cooldown_frames

            for i in range(1, len(history)):
                f_idx, x, y, hconf = history[i]
                _, x_prev, y_prev, _ = history[i - 1]

                in_box_now  = x >= THRESHOLDS.BOX_X_M and abs(y) <= THRESHOLDS.BOX_Y_M
                in_box_prev = x_prev >= THRESHOLDS.BOX_X_M and abs(y_prev) <= THRESHOLDS.BOX_Y_M

                if in_box_now and not in_box_prev:  # Just entered
                    if f_idx - last_event_frame > cooldown_frames:
                        last_event_frame = f_idx
                        hdisc = _homography_discount(hconf)
                        confidence = 0.85 * hdisc
                        importance = 0.90  # Box entry is always highly important
                        t_s = round(f_idx / self.fps, 1)
                        events.append(_make_event(
                            event_type="THR_003",
                            event_name="Box Entry",
                            category="threat",
                            player_id=pid,
                            team_id=team_id,
                            start_frame=f_idx,
                            end_frame=f_idx,
                            fps=self.fps,
                            confidence=confidence,
                            importance=importance,
                            description=(
                                f"Player {pid} ({team_id}) entered the penalty box at "
                                f"{int(t_s // 60):02d}:{int(t_s % 60):02d}."
                            ),
                            job_id=self.job_id,
                            start_radar_pt=(x_prev, y_prev),
                            end_radar_pt=(x, y),
                            tags=["THR_003", "box_entry", "danger_zone"],
                        ))
        return events

    # ── THR_001: Dangerous Run ────────────────────────────────────────────────

    def _detect_dangerous_runs(
        self,
        frames: FrameList,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        """
        Detect forward runs into the final third that are uncontested (no defender
        within UNCONTESTED_RADIUS_M for ≥ UNCONTESTED_FRACTION of run frames).
        """
        events: list[EventRecord] = []
        min_frames = math.ceil(THRESHOLDS.DANGEROUS_RUN_MIN_DURATION_S * self.fps)

        # Build frame-level lookup for fast defender position access
        frame_defenders: dict[int, dict[str, list[tuple[float, float]]]] = {}
        for frame in frames:
            f_idx = frame["frame_idx"]
            frame_defenders[f_idx] = {"team_0": [], "team_1": []}
            for p in frame.get("players", []):
                tid = p.get("team_id")
                x = p.get("x_pitch")
                y = p.get("y_pitch")
                if tid and x is not None and y is not None:
                    frame_defenders[f_idx].setdefault(tid, []).append((float(x), float(y)))

        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            opp_team = "team_1" if team_id == "team_0" else "team_0"

            # Find forward-advancing segments in final third
            run_start = None
            uncontested_count = 0
            total_count = 0

            for i in range(1, len(history)):
                f_idx, x, y, hconf = history[i]
                _, x_prev, _, _ = history[i - 1]
                dx = x - x_prev

                # Must be in final third and moving forward
                if x >= THRESHOLDS.FINAL_THIRD_X_M and dx >= 0:
                    if run_start is None:
                        run_start = i - 1
                        uncontested_count = 0
                        total_count = 0

                    # Check uncontested status
                    defenders = frame_defenders.get(f_idx, {}).get(opp_team, [])
                    min_dist = self._min_defender_dist((x, y), defenders)
                    total_count += 1
                    if min_dist >= THRESHOLDS.UNCONTESTED_RADIUS_M:
                        uncontested_count += 1
                else:
                    # End of run segment
                    if run_start is not None and total_count >= min_frames:
                        fraction = uncontested_count / total_count if total_count > 0 else 0
                        if fraction >= THRESHOLDS.UNCONTESTED_FRACTION:
                            self._emit_dangerous_run(
                                pid, team_id, history, run_start, i - 1,
                                frame_defenders, opp_team, events
                            )
                    run_start = None
                    uncontested_count = 0
                    total_count = 0

            # Close trailing run
            if run_start is not None and total_count >= min_frames:
                fraction = uncontested_count / total_count if total_count > 0 else 0
                if fraction >= THRESHOLDS.UNCONTESTED_FRACTION:
                    self._emit_dangerous_run(
                        pid, team_id, history, run_start, len(history) - 1,
                        frame_defenders, opp_team, events
                    )

        return events

    def _emit_dangerous_run(
        self,
        pid: int,
        team_id: str,
        history: list[tuple[int, float, float, float]],
        s: int,
        e: int,
        frame_defenders: dict[int, dict[str, list[tuple[float, float]]]],
        opp_team: str,
        events: list[EventRecord],
    ) -> None:
        start_h = history[s]
        end_h   = history[e]
        duration_s = (end_h[0] - start_h[0]) / self.fps
        avg_hconf = sum(history[j][3] for j in range(s, e + 1)) / max(1, e - s + 1)

        # Compute mean defender distance over run
        distances = []
        for j in range(s, e + 1):
            f_idx, x, y, _ = history[j]
            defs = frame_defenders.get(f_idx, {}).get(opp_team, [])
            distances.append(self._min_defender_dist((x, y), defs))
        mean_dist = sum(distances) / len(distances) if distances else 0.0

        hdisc = _homography_discount(avg_hconf)
        confidence = (
            0.5 * min(1.0, mean_dist / 10.0)
            + 0.3 * min(1.0, duration_s / 3.0)
            + 0.2 * hdisc
        )
        zone = classify_zone(end_h[1], end_h[2])
        importance = (
            0.5 * (1.0 if zone in {"box", "left_channel", "right_channel"} else 0.6)
            + 0.3 * min(1.0, mean_dist / 10.0)
            + 0.2 * confidence
        )

        t_s = round(start_h[0] / self.fps, 1)
        events.append(_make_event(
            event_type="THR_001",
            event_name="Dangerous Run",
            category="threat",
            player_id=pid,
            team_id=team_id,
            start_frame=start_h[0],
            end_frame=end_h[0],
            fps=self.fps,
            confidence=confidence,
            importance=importance,
            description=(
                f"Player {pid} ({team_id}) made a dangerous run into the {zone.replace('_', ' ')} "
                f"at {int(t_s // 60):02d}:{int(t_s % 60):02d}, "
                f"{mean_dist:.1f}m from the nearest defender."
            ),
            job_id=self.job_id,
            start_radar_pt=(start_h[1], start_h[2]),
            end_radar_pt=(end_h[1], end_h[2]),
            peak_radar_pt=(end_h[1], end_h[2]),
            tags=["THR_001", "dangerous_run", zone, "uncontested"],
        ))

    # ── THR_006: Channel Exploitation ─────────────────────────────────────────

    def _detect_channel_exploitations(
        self,
        frames: FrameList,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        events: list[EventRecord] = []
        min_frames = math.ceil(THRESHOLDS.DANGEROUS_RUN_MIN_DURATION_S * self.fps)

        frame_map = {f["frame_idx"]: f for f in frames}

        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            opp_team = "team_1" if team_id == "team_0" else "team_0"
            in_channel = False
            channel_start = 0
            channel_frames: list[tuple[int, float, float]] = []
            uncontested_count = 0

            for i, (f_idx, x, y, hconf) in enumerate(history):
                in_left  = x >= THRESHOLDS.CHANNEL_X_MIN_M and THRESHOLDS.CHANNEL_Y_INNER_M <= y <= THRESHOLDS.CHANNEL_Y_OUTER_M
                in_right = x >= THRESHOLDS.CHANNEL_X_MIN_M and -THRESHOLDS.CHANNEL_Y_OUTER_M <= y <= -THRESHOLDS.CHANNEL_Y_INNER_M

                if in_left or in_right:
                    if not in_channel:
                        in_channel = True
                        channel_start = i
                        channel_frames = []
                        uncontested_count = 0

                    frame = frame_map.get(f_idx, {})
                    defenders = [(p.get("x_pitch", 0), p.get("y_pitch", 0))
                                 for p in frame.get("players", [])
                                 if p.get("team_id") == opp_team and p.get("x_pitch") is not None]

                    min_dist = self._min_defender_dist((x, y), defenders)  # type: ignore[arg-type]
                    channel_frames.append((f_idx, x, y))
                    if min_dist >= THRESHOLDS.CHANNEL_UNCONTESTED_M:
                        uncontested_count += 1
                else:
                    if in_channel and len(channel_frames) >= min_frames:
                        fraction = uncontested_count / len(channel_frames)
                        if fraction >= 0.50:
                            self._emit_channel_event(pid, team_id, channel_frames, events)
                    in_channel = False
                    channel_frames = []
                    uncontested_count = 0

            if in_channel and len(channel_frames) >= min_frames:
                fraction = uncontested_count / len(channel_frames)
                if fraction >= 0.50:
                    self._emit_channel_event(pid, team_id, channel_frames, events)

        return events

    def _emit_channel_event(
        self,
        pid: int,
        team_id: str,
        channel_frames: list[tuple[int, float, float]],
        events: list[EventRecord],
    ) -> None:
        start_f, start_x, start_y = channel_frames[0]
        end_f, end_x, end_y       = channel_frames[-1]
        duration_s = (end_f - start_f) / self.fps
        side = "left" if start_y > 0 else "right"
        t_s = round(start_f / self.fps, 1)

        confidence = 0.70 + 0.20 * min(1.0, duration_s / 3.0)
        importance = 0.75 + 0.15 * min(1.0, duration_s / 5.0) + 0.10 * confidence

        events.append(_make_event(
            event_type="THR_006",
            event_name="Channel Exploitation",
            category="threat",
            player_id=pid,
            team_id=team_id,
            start_frame=start_f,
            end_frame=end_f,
            fps=self.fps,
            confidence=confidence,
            importance=importance,
            description=(
                f"Player {pid} ({team_id}) exploited the {side} channel between "
                f"centre-back and full-back for {duration_s:.1f}s at "
                f"{int(t_s // 60):02d}:{int(t_s % 60):02d}."
            ),
            job_id=self.job_id,
            start_radar_pt=(start_x, start_y),
            end_radar_pt=(end_x, end_y),
            tags=["THR_006", "channel", side, "in_behind"],
        ))

    # ── THR_007: Isolated Defender Exploit ────────────────────────────────────

    def _detect_isolated_defender(
        self,
        frames: FrameList,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        """
        1v1 wide situation: attacker in wide area, has one nearby defender (< 4 m)
        but nearest *supporting* defender is >= ISOLATED_DEFENDER_SUPPORT_M away.
        """
        events: list[EventRecord] = []
        cooldown = int(8.0 * self.fps)
        frame_map = {f["frame_idx"]: f for f in frames}

        for pid, history in player_history.items():
            team_id = team_map.get(pid, "unknown")
            opp_team = "team_1" if team_id == "team_0" else "team_0"
            last_event = -cooldown

            for f_idx, x, y, hconf in history:
                if f_idx - last_event <= cooldown:
                    continue
                if not (abs(y) >= THRESHOLDS.WIDE_Y_THRESHOLD_M and x >= 15.0):
                    continue

                frame = frame_map.get(f_idx, {})
                defenders = [(float(p["x_pitch"]), float(p["y_pitch"]))
                             for p in frame.get("players", [])
                             if p.get("team_id") == opp_team
                             and p.get("x_pitch") is not None and p.get("y_pitch") is not None]

                if len(defenders) < 1:
                    continue

                dists = sorted(_dist((x, y), d) for d in defenders)
                nearest = dists[0]
                second_nearest = dists[1] if len(dists) > 1 else 99.0

                is_1v1 = nearest < 4.0 and second_nearest >= THRESHOLDS.ISOLATED_DEFENDER_SUPPORT_M

                if is_1v1:
                    last_event = f_idx
                    side = "left" if y > 0 else "right"
                    t_s = round(f_idx / self.fps, 1)
                    hdisc = _homography_discount(hconf)
                    confidence = 0.72 * hdisc
                    importance = 0.75 + 0.25 * min(1.0, second_nearest / 15.0)

                    events.append(_make_event(
                        event_type="THR_007",
                        event_name="Isolated Defender Exploit",
                        category="threat",
                        player_id=pid,
                        team_id=team_id,
                        start_frame=f_idx,
                        end_frame=f_idx,
                        fps=self.fps,
                        confidence=confidence,
                        importance=importance,
                        description=(
                            f"Player {pid} ({team_id}) isolated a defender 1v1 in the "
                            f"{side} wide area at {int(t_s // 60):02d}:{int(t_s % 60):02d}. "
                            f"Nearest defensive support was {second_nearest:.1f}m away."
                        ),
                        job_id=self.job_id,
                        start_radar_pt=(x, y),
                        tags=["THR_007", "isolated_1v1", side, "wide"],
                    ))

        return events

    # ── THR_004: Transition Involvement ───────────────────────────────────────

    def _detect_transition_involvement(
        self,
        frames: FrameList,
        player_history: dict[int, list[tuple[int, float, float, float]]],
        team_map: dict[int, str],
    ) -> list[EventRecord]:
        """Flag players who were actively moving near the ball within 5s of a possession change."""
        events: list[EventRecord] = []
        window_frames = int(THRESHOLDS.PRESS_SUCCESS_WINDOW_S * self.fps)

        possession_changes: list[tuple[int, str]] = []
        prev_poss = None
        frame_ball: dict[int, tuple[float, float] | None] = {}

        for frame in frames:
            f_idx = frame["frame_idx"]
            curr_poss = frame.get("possession_team_id")
            ball = frame.get("ball_xy")
            frame_ball[f_idx] = (ball[0], ball[1]) if ball else None

            if prev_poss and curr_poss and curr_poss != prev_poss:
                possession_changes.append((f_idx, curr_poss))  # team that gained possession
            prev_poss = curr_poss

        seen: set[tuple[int, int]] = set()  # (pid, change_frame) already emitted

        for change_frame, gaining_team in possession_changes:
            end_window = change_frame + window_frames

            for pid, history in player_history.items():
                if team_map.get(pid) != gaining_team:
                    continue
                key = (pid, change_frame)
                if key in seen:
                    continue

                # Check if player was active (speed > 2 m/s) near ball during window
                window_pts = [(f, x, y) for f, x, y, _ in history
                              if change_frame <= f <= end_window]
                if not window_pts:
                    continue

                for j, (f, x, y) in enumerate(window_pts):
                    if j == 0:
                        continue
                    spd = _speed_mps(
                        (window_pts[j - 1][1], window_pts[j - 1][2]),
                        (x, y),
                        f - window_pts[j - 1][0],
                        self.fps,
                    )
                    ball_pt = frame_ball.get(f)
                    if spd >= 2.0 and ball_pt and _dist((x, y), ball_pt) <= 20.0:
                        seen.add(key)
                        t_s = round(change_frame / self.fps, 1)
                        events.append(_make_event(
                            event_type="THR_004",
                            event_name="Transition Involvement",
                            category="threat",
                            player_id=pid,
                            team_id=gaining_team,
                            start_frame=change_frame,
                            end_frame=min(end_window, (window_pts[-1][0])),
                            fps=self.fps,
                            confidence=0.60,
                            importance=0.45,
                            description=(
                                f"Player {pid} ({gaining_team}) was involved in the "
                                f"transition at {int(t_s // 60):02d}:{int(t_s % 60):02d}."
                            ),
                            job_id=self.job_id,
                            start_radar_pt=(window_pts[0][1], window_pts[0][2]),
                            tags=["THR_004", "transition", "involvement"],
                        ))
                        break  # One per player per transition

        return events
