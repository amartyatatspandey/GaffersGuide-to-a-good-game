"""Optional ReID + radar ID healing (extracted from track_teams for Rule 4)."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import supervision as sv

from gaffers_guide.pipeline.track_teams_constants import (
    CLASS_PLAYER,
    RADAR_DISTANCE_HEAL_PX,
    REID_COSINE_THRESHOLD,
    cosine_similarity,
)

logger = logging.getLogger(__name__)

try:
    from reid_healer import VisualFingerprint

    REID_AVAILABLE = True
except Exception as e:  # noqa: BLE001 — intentional graceful degradation
    REID_AVAILABLE = False
    VisualFingerprint = None  # type: ignore[misc, assignment]
    logger.warning(
        "Failed to load reid_healer.py: %s. ID Healing OFFLINE. Defaulting to ByteTrack.",
        e,
    )


class HybridIDHealer:
    """
    Firewall between ByteTrack IDs and reid_healer: optional ReID + radar distance checks.
    Degrades cleanly when ReID is unavailable or VisualFingerprint fails to construct.
    """

    def __init__(self) -> None:
        self.active = REID_AVAILABLE
        self.fingerprint_engine: Any = None
        if self.active and VisualFingerprint is not None:
            try:
                self.fingerprint_engine = VisualFingerprint()
            except Exception as e:  # noqa: BLE001 — disable healing, keep tracking
                logger.warning(
                    "VisualFingerprint() failed (%s). ID Healing OFFLINE. Defaulting to ByteTrack.",
                    e,
                )
                self.active = False
                self.fingerprint_engine = None

        self.id_fingerprints: dict[int, np.ndarray] = {}
        self.id_last_radar: dict[int, tuple[float, float]] = {}
        self.id_last_seen_frame: dict[int, int] = {}
        self.heal_map: dict[int, int] = {}

    def _resolve_heal_chain(self, raw_tid: int) -> int:
        tid = raw_tid
        seen: set[int] = set()
        while tid in self.heal_map and tid not in seen:
            seen.add(tid)
            tid = self.heal_map[tid]
        return tid

    def cleanup_ghost_ids(self, current_frame: int) -> None:
        """Drop stale ReID / radar state for IDs not seen for >300 frames (~12 s @ 25 fps)."""
        if not self.active:
            return
        dead_ids = [
            tid for tid, last_seen in self.id_last_seen_frame.items() if current_frame - last_seen > 300
        ]
        for tid in dead_ids:
            self.id_fingerprints.pop(tid, None)
            self.id_last_radar.pop(tid, None)
            self.id_last_seen_frame.pop(tid, None)

    def process_and_heal(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
        radar_pts: list[tuple[int, int] | None],
        frame_idx: int,
    ) -> np.ndarray | None:
        """
        Optionally rewrites tracker IDs on ``detections`` using ReID + radar proximity.
        ``radar_pts[i]`` must match ``detections`` row ``i`` (precomputed image→radar projection).
        Returns the tracker_id array (or None if tracking disabled).
        """
        tracker_id = getattr(detections, "tracker_id", None)
        if not self.active or self.fingerprint_engine is None or tracker_id is None:
            return tracker_id

        n = len(detections)
        if n == 0:
            return tracker_id
        if len(radar_pts) != n:
            logger.warning(
                "radar_pts length %d != detections %d; skipping heal for this frame",
                len(radar_pts),
                n,
            )
            return tracker_id

        raw_ids = [int(detections.tracker_id[i]) for i in range(n)]
        logical_ids = [self._resolve_heal_chain(r) for r in raw_ids]
        on_screen = set(logical_ids)

        new_tracker_ids: list[int] = []
        for i in range(n):
            cid = int(detections.class_id[i])
            bbox = detections.xyxy[i]
            raw_tid = raw_ids[i]
            tid = logical_ids[i]

            if cid == CLASS_PLAYER:
                new_fp: np.ndarray | None = None
                if tid not in self.id_fingerprints:
                    new_fp = self.fingerprint_engine.extract_features(frame, bbox)
                    pr = radar_pts[i]
                    new_radar_pt = (float(pr[0]), float(pr[1])) if pr is not None else None
                    best_match_id: int | None = None
                    best_sim = 0.0
                    if new_fp is not None and new_radar_pt is not None:
                        for old_id, old_fp in self.id_fingerprints.items():
                            if old_id in on_screen:
                                continue
                            sim = cosine_similarity(new_fp, old_fp)
                            if sim > REID_COSINE_THRESHOLD:
                                old_radar_pt = self.id_last_radar.get(old_id)
                                if old_radar_pt is not None:
                                    dist = math.dist(new_radar_pt, old_radar_pt)
                                    if dist < RADAR_DISTANCE_HEAL_PX and sim > best_sim:
                                        best_sim = sim
                                        best_match_id = old_id
                    if best_match_id is not None:
                        logger.info(
                            "[HEALER] ReID Match! Swapping ID %s -> %s (Sim: %.2f)",
                            raw_tid,
                            best_match_id,
                            best_sim,
                        )
                        self.heal_map[raw_tid] = best_match_id
                        on_screen.discard(tid)
                        tid = best_match_id
                        on_screen.add(tid)
                        logical_ids[i] = tid
                        if new_fp is not None:
                            self.id_fingerprints[tid] = new_fp
                    elif new_fp is not None:
                        self.id_fingerprints[tid] = new_fp

                if frame_idx % 15 == 0 or tid not in self.id_fingerprints:
                    fp = self.fingerprint_engine.extract_features(frame, bbox)
                    if fp is not None:
                        self.id_fingerprints[tid] = fp

                pr2 = radar_pts[i]
                if pr2 is not None:
                    self.id_last_radar[tid] = (float(pr2[0]), float(pr2[1]))
                self.id_last_seen_frame[tid] = frame_idx

            new_tracker_ids.append(tid)

        out = np.asarray(new_tracker_ids, dtype=np.int32)
        detections.tracker_id = out
        return out
