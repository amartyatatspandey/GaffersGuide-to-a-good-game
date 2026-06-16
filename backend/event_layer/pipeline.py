"""
Event Intelligence Layer — Detection Pipeline
==============================================

Orchestrates all five detectors in dependency order and produces
the complete EventIndex for a job.

Dependency order (required):
  1. Movement (radar_pt sequences only)
  2. Positional (radar_pt + zone logic)
  3. Transition (possession sequence)
  4. Shape (team aggregations + metrics timeline)
  5. Threat (movement + positions + opponent locations)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from event_layer.models import EventIndex, EventRecord
from event_layer.detectors.movement import MovementDetector
from event_layer.detectors.positional import PositionalDetector
from event_layer.detectors.threat import ThreatDetector
from event_layer.detectors.shape import ShapeDetector
from event_layer.detectors.transition import TransitionDetector

LOGGER = logging.getLogger(__name__)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventDetectionPipeline:
    """
    Runs all event detectors against a frame sequence and produces an EventIndex.

    Usage:
        pipeline = EventDetectionPipeline(fps=25.0, job_id="abc123")
        index = pipeline.run(frames, metrics_timeline)
        index.save(output_path)
    """

    def __init__(self, fps: float, job_id: str) -> None:
        self.fps = fps
        self.job_id = job_id

    def run(
        self,
        frames: list[dict[str, Any]],
        metrics_timeline: list[dict[str, Any]] | None = None,
        *,
        progress_callback=None,
    ) -> EventIndex:
        """
        Run all detectors and return a populated EventIndex.

        Args:
            frames: List of TrackingFrameArtifact dicts from parallel_pipeline.py.
                    Each dict must have: frame_idx, players, ball_xy,
                    possession_team_id, homography_confidence.
                    Players must have: id, team_id, x_pitch, y_pitch.
            metrics_timeline: Optional pre-computed metrics from build_metrics_timeline().
                              Used by ShapeDetector for efficiency.
            progress_callback: Optional callable(str) for step reporting.
        """
        if not frames:
            LOGGER.warning("EventDetectionPipeline: empty frame list, no events to detect.")
            return EventIndex(
                job_id=self.job_id,
                fps=self.fps,
                total_frames=0,
                generated_at=_now_utc(),
            )

        # Normalize player field names: parallel_pipeline uses x_pitch/y_pitch
        # but older artifacts may use radar_pt. We support both.
        frames = _normalize_frames(frames)

        total_frames = frames[-1]["frame_idx"] - frames[0]["frame_idx"] + 1
        all_events: list[EventRecord] = []

        def _step(msg: str) -> None:
            LOGGER.info("EventDetectionPipeline: %s", msg)
            if progress_callback:
                progress_callback(f"Event Layer: {msg}")

        # ── Step 1: Movement ──────────────────────────────────────────────────
        _step("Detecting movement events (MOV)")
        t0 = time.perf_counter()
        mov = MovementDetector(fps=self.fps, job_id=self.job_id)
        mov_events = mov.detect(frames)
        all_events.extend(mov_events)
        _step(f"Movement: {len(mov_events)} events ({time.perf_counter() - t0:.1f}s)")

        # ── Step 2: Positional ────────────────────────────────────────────────
        _step("Detecting positional events (POS)")
        t0 = time.perf_counter()
        pos = PositionalDetector(fps=self.fps, job_id=self.job_id)
        pos_events = pos.detect(frames)
        all_events.extend(pos_events)
        _step(f"Positional: {len(pos_events)} events ({time.perf_counter() - t0:.1f}s)")

        # ── Step 3: Transition ────────────────────────────────────────────────
        _step("Detecting transition events (TRN)")
        t0 = time.perf_counter()
        trn = TransitionDetector(fps=self.fps, job_id=self.job_id)
        trn_events = trn.detect(frames)
        all_events.extend(trn_events)
        _step(f"Transition: {len(trn_events)} events ({time.perf_counter() - t0:.1f}s)")

        # ── Step 4: Shape ─────────────────────────────────────────────────────
        _step("Detecting shape events (SHP)")
        t0 = time.perf_counter()
        shp = ShapeDetector(fps=self.fps, job_id=self.job_id, metrics_timeline=metrics_timeline)
        shp_events = shp.detect(frames)
        all_events.extend(shp_events)
        _step(f"Shape: {len(shp_events)} events ({time.perf_counter() - t0:.1f}s)")

        # ── Step 5: Threat ────────────────────────────────────────────────────
        _step("Detecting threat events (THR)")
        t0 = time.perf_counter()
        thr = ThreatDetector(fps=self.fps, job_id=self.job_id)
        thr_events = thr.detect(frames)
        all_events.extend(thr_events)
        _step(f"Threat: {len(thr_events)} events ({time.perf_counter() - t0:.1f}s)")

        # ── Clamp clip_end_frame to actual video bounds ────────────────────────
        last_frame = frames[-1]["frame_idx"]
        for event in all_events:
            event.clip_end_frame = min(event.clip_end_frame, last_frame)

        # ── Sort chronologically ───────────────────────────────────────────────
        all_events.sort(key=lambda e: e.start_frame)

        index = EventIndex(
            job_id=self.job_id,
            fps=self.fps,
            total_frames=total_frames,
            events=all_events,
            generated_at=_now_utc(),
        )

        stats = index.stats()
        _step(
            f"Complete — {stats['total_events']} events detected across "
            f"{stats['players_with_events']} players. "
            f"By category: {stats['by_category']}"
        )

        return index


def _normalize_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ensure player dicts have x_pitch / y_pitch fields.

    TrackingFrameArtifact from parallel_pipeline stores these directly.
    Legacy formats may use radar_pt = [x, y].
    """
    normalized = []
    for frame in frames:
        players_out = []
        for p in frame.get("players", []):
            if p.get("x_pitch") is None and "radar_pt" in p:
                rp = p["radar_pt"]
                if isinstance(rp, (list, tuple)) and len(rp) >= 2:
                    p = dict(p)
                    p["x_pitch"] = float(rp[0])
                    p["y_pitch"] = float(rp[1])
            players_out.append(p)
        frame = dict(frame)
        frame["players"] = players_out
        normalized.append(frame)
    return normalized


def run_event_detection(
    frames: list[dict[str, Any]],
    *,
    fps: float,
    job_id: str,
    output_dir: Path,
    metrics_timeline: list[dict[str, Any]] | None = None,
    progress_callback=None,
) -> Path:
    """
    Convenience function: run the pipeline and write the EventIndex to disk.

    Returns the path to the written JSON file.
    """
    pipeline = EventDetectionPipeline(fps=fps, job_id=job_id)
    index = pipeline.run(
        frames,
        metrics_timeline=metrics_timeline,
        progress_callback=progress_callback,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{job_id}_events.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index.model_dump(), f, indent=2, ensure_ascii=False)

    LOGGER.info("EventIndex written to %s (%d events)", out_path, len(index.events))
    return out_path


def load_event_index(path: Path) -> EventIndex:
    """Load a previously-saved EventIndex from disk."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return EventIndex.model_validate(data)
