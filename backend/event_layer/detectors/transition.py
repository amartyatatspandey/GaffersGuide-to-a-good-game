"""
Event Intelligence Layer — Transition Detectors
================================================

Detects possession-change moments and their consequences:
  TRN_001  Defensive Transition    team loses possession; players moving back
  TRN_002  Offensive Transition    team gains possession; forward players advance
  TRN_003  Press Success           ball won within 5s of high press
  TRN_004  Press Failure           opposition plays through press with >= 2 changes
  TRN_005  Counter-Attack Sequence reaching final third within 8s of gaining possession
"""
from __future__ import annotations

import math
from typing import Any

from event_layer.detectors._base import (
    FrameList, _build_player_position_history, _dist, _make_event,
)
from event_layer.models import EventRecord
from event_layer.ontology import THRESHOLDS


class TransitionDetector:
    """Detects all transition-category events (TRN_001 – TRN_005)."""

    def __init__(self, fps: float, job_id: str) -> None:
        self.fps = fps
        self.job_id = job_id

    def detect(self, frames: FrameList) -> list[EventRecord]:
        events: list[EventRecord] = []
        events.extend(self._detect_transitions(frames))
        events.extend(self._detect_press_outcomes(frames))
        events.extend(self._detect_counter_sequences(frames))
        return events

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_possession_changes(
        self, frames: FrameList
    ) -> list[tuple[int, str, str]]:
        """Return list of (frame_idx, losing_team, gaining_team) for every possession change."""
        changes = []
        prev_poss = None
        for frame in frames:
            curr = frame.get("possession_team_id")
            if prev_poss and curr and curr != prev_poss:
                changes.append((frame["frame_idx"], prev_poss, curr))
            prev_poss = curr
        return changes

    # ── TRN_001 + TRN_002 ─────────────────────────────────────────────────────

    def _detect_transitions(self, frames: FrameList) -> list[EventRecord]:
        events: list[EventRecord] = []
        changes = self._find_possession_changes(frames)
        window_frames = int(THRESHOLDS.TRANSITION_WINDOW_S * self.fps)
        frame_map = {f["frame_idx"]: f for f in frames}

        for change_frame, losing_team, gaining_team in changes:
            end_frame = min(change_frame + window_frames, frames[-1]["frame_idx"])
            t_s = round(change_frame / self.fps, 1)

            # TRN_001: Losing team defensive transition
            events.append(_make_event(
                event_type="TRN_001",
                event_name="Defensive Transition",
                category="transition",
                player_id=None,
                team_id=losing_team,
                start_frame=change_frame,
                end_frame=end_frame,
                fps=self.fps,
                confidence=0.85,
                importance=0.70,
                description=(
                    f"{losing_team} entered a defensive transition at "
                    f"{int(t_s // 60):02d}:{int(t_s % 60):02d} after losing possession."
                ),
                job_id=self.job_id,
                tags=["TRN_001", "defensive_transition", losing_team],
            ))

            # TRN_002: Gaining team offensive transition
            events.append(_make_event(
                event_type="TRN_002",
                event_name="Offensive Transition",
                category="transition",
                player_id=None,
                team_id=gaining_team,
                start_frame=change_frame,
                end_frame=end_frame,
                fps=self.fps,
                confidence=0.85,
                importance=0.72,
                description=(
                    f"{gaining_team} launched an offensive transition at "
                    f"{int(t_s // 60):02d}:{int(t_s % 60):02d} after gaining possession."
                ),
                job_id=self.job_id,
                tags=["TRN_002", "offensive_transition", gaining_team],
            ))

        return events

    # ── TRN_003 + TRN_004 ─────────────────────────────────────────────────────

    def _detect_press_outcomes(self, frames: FrameList) -> list[EventRecord]:
        """
        TRN_003 (Press Success): Possession changes back to pressing team within 5s.
        TRN_004 (Press Failure): No recovery within 5s — opposition plays through.
        
        Proxy: We identify high-press windows (from frame possession data) and check
        if possession flips back within the press-success window.
        """
        events: list[EventRecord] = []
        changes = self._find_possession_changes(frames)
        if not changes:
            return events

        success_window = int(THRESHOLDS.PRESS_SUCCESS_WINDOW_S * self.fps)
        frame_poss = {f["frame_idx"]: f.get("possession_team_id") for f in frames}
        all_frame_idxs = sorted(frame_poss.keys())

        for i, (change_frame, losing_team, gaining_team) in enumerate(changes):
            # Look for re-possession by the originally-losing team within success_window
            end_check = change_frame + success_window
            possession_flipped_back = False

            for f_idx in all_frame_idxs:
                if f_idx <= change_frame:
                    continue
                if f_idx > end_check:
                    break
                if frame_poss.get(f_idx) == losing_team:
                    possession_flipped_back = True
                    break

            t_s = round(change_frame / self.fps, 1)

            if possession_flipped_back:
                events.append(_make_event(
                    event_type="TRN_003",
                    event_name="Press Success",
                    category="transition",
                    player_id=None,
                    team_id=losing_team,  # The pressing team that recovered
                    start_frame=change_frame,
                    end_frame=min(end_check, all_frame_idxs[-1]),
                    fps=self.fps,
                    confidence=0.78,
                    importance=0.80,
                    description=(
                        f"{losing_team} successfully won back possession within "
                        f"{THRESHOLDS.PRESS_SUCCESS_WINDOW_S:.0f}s at "
                        f"{int(t_s // 60):02d}:{int(t_s % 60):02d} — press success."
                    ),
                    job_id=self.job_id,
                    tags=["TRN_003", "press_success", losing_team],
                ))
            else:
                events.append(_make_event(
                    event_type="TRN_004",
                    event_name="Press Failure",
                    category="transition",
                    player_id=None,
                    team_id=losing_team,  # The pressing team that failed
                    start_frame=change_frame,
                    end_frame=min(end_check, all_frame_idxs[-1]),
                    fps=self.fps,
                    confidence=0.72,
                    importance=0.65,
                    description=(
                        f"{losing_team} failed to recover possession within "
                        f"{THRESHOLDS.PRESS_SUCCESS_WINDOW_S:.0f}s at "
                        f"{int(t_s // 60):02d}:{int(t_s % 60):02d} — press bypassed."
                    ),
                    job_id=self.job_id,
                    tags=["TRN_004", "press_failure", losing_team],
                ))

        return events

    # ── TRN_005: Counter-Attack Sequence ──────────────────────────────────────

    def _detect_counter_sequences(self, frames: FrameList) -> list[EventRecord]:
        """
        TRN_005: A player from the gaining team reaches the final third within
        COUNTER_REACH_FINAL_THIRD_S seconds of a possession change.
        """
        events: list[EventRecord] = []
        changes = self._find_possession_changes(frames)
        final_third_x = THRESHOLDS.FINAL_THIRD_X_M
        window_s = THRESHOLDS.COUNTER_REACH_FINAL_THIRD_S
        window_frames = int(window_s * self.fps)
        frame_map = {f["frame_idx"]: f for f in frames}

        for change_frame, _, gaining_team in changes:
            end_window = change_frame + window_frames

            for f_idx in range(change_frame + 1, end_window + 1):
                frame = frame_map.get(f_idx)
                if not frame:
                    continue

                for p in frame.get("players", []):
                    if p.get("team_id") != gaining_team:
                        continue
                    x = p.get("x_pitch")
                    if x is None or x < final_third_x:
                        continue

                    # Check previous frames to confirm this player wasn't already there
                    prev_frame = frame_map.get(change_frame, {})
                    was_in_final_third = any(
                        prev_p.get("id") == p.get("id")
                        and (prev_p.get("x_pitch") or 0) >= final_third_x
                        for prev_p in prev_frame.get("players", [])
                    )

                    if not was_in_final_third:
                        speed_of_counter = (x - (prev_frame.get("players", [{}])[0].get("x_pitch", 0) or 0)) / max(1, (f_idx - change_frame) / self.fps)
                        elapsed_s = (f_idx - change_frame) / self.fps
                        t_s = round(change_frame / self.fps, 1)

                        events.append(_make_event(
                            event_type="TRN_005",
                            event_name="Counter-Attack Sequence",
                            category="transition",
                            player_id=p.get("id"),
                            team_id=gaining_team,
                            start_frame=change_frame,
                            end_frame=f_idx,
                            fps=self.fps,
                            confidence=0.80,
                            importance=0.88,
                            description=(
                                f"{gaining_team} completed a counter-attack sequence — "
                                f"player {p.get('id')} reached the final third in "
                                f"{elapsed_s:.1f}s at {int(t_s // 60):02d}:{int(t_s % 60):02d}."
                            ),
                            job_id=self.job_id,
                            end_radar_pt=(float(x), float(p.get("y_pitch", 0) or 0)),
                            tags=["TRN_005", "counter_attack", gaining_team],
                        ))
                        break  # One counter event per possession change
                else:
                    continue
                break

        return events
