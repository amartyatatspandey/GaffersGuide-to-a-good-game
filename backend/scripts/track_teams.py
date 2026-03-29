"""
Team-aware tracking: YOLO + ByteTrack + team classification with soft lock and goalkeeper detection.

Uses HSV dominant color, 90-frame sliding window for auto-correction (no permanent mistake trap),
and distance-based outlier detection to isolate goalkeepers. Global K-Means fits when 10+ players
have 15+ color observations; players stay gray until 15 frames observed.

Setup: pip install scikit-learn supervision ultralytics opencv-python

Execution: python backend/scripts/track_teams.py
"""

import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths: script lives in backend/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent

# Ensure sibling modules (e.g. reid_healer) resolve when run as a script or from another cwd.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# --- DECOUPLED REID IMPORT ---
# Explicit fallback: pipeline keeps running if ReID deps or reid_healer.py are missing or broken.
try:
    from reid_healer import VisualFingerprint

    REID_AVAILABLE = True
    logger.info("reid_healer.py found. Hybrid ID Healing ONLINE.")
except Exception as e:  # noqa: BLE001 — intentional graceful degradation
    REID_AVAILABLE = False
    VisualFingerprint = None  # type: ignore[misc, assignment]
    logger.warning(
        "Failed to load reid_healer.py: %s. ID Healing OFFLINE. Defaulting to ByteTrack.",
        e,
    )


def cosine_similarity(v1: np.ndarray | Sequence[float], v2: np.ndarray | Sequence[float]) -> float:
    """Cosine similarity between two vectors (e.g. 512-D ReID embeddings)."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    norm1 = float(np.linalg.norm(a))
    norm2 = float(np.linalg.norm(b))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm1 * norm2))
MODEL_PATH = BACKEND_ROOT / "models" / "pretrained" / "best.pt"
VIDEO_PATH = BACKEND_ROOT / "data" / "match_test.mp4"
OUTPUT_PATH = BACKEND_ROOT / "output" / "tracking_teams.mp4"

# Class IDs: 0=Player, 1=Ball, 2=Referee
CLASS_PLAYER = 0
CLASS_BALL = 1
CLASS_REF = 2

# Team box colors (BGR for OpenCV)
COLOR_TEAM_0 = (0, 0, 255)       # Red (Team A)
COLOR_TEAM_1 = (255, 0, 0)       # Blue (Team B)
COLOR_UNKNOWN = (128, 128, 128)  # Gray (waiting for buffer)
COLOR_GOALKEEPER = (0, 255, 255) # Yellow for GK (team-associated)
COLOR_BALL = (255, 255, 0)       # Cyan
COLOR_REF = (0, 165, 255)        # Orange
TEAM_COLORS = {0: COLOR_TEAM_0, 1: COLOR_TEAM_1}

# Soft lock: min frames before assigning; max history for auto-correction
MIN_COLORS_BEFORE_PREDICT = 15
MIN_PLAYERS_TO_FIT = 10
# HSV green range (OpenCV H 0–180): pitch background
HSV_GREEN_H_LOW, HSV_GREEN_H_HIGH = 35, 85
HSV_GREEN_S_MIN = 40

# Hybrid healer: cosine ReID + tactical radar gate (scaled pitch pixels ~= meters * 10)
REID_COSINE_THRESHOLD = 0.85
RADAR_DISTANCE_HEAL_PX = 150.0  # ~15 m on radar at scale=10


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


class TeamClassifier:
    """
    Dynamic Anchor architecture: base anchors (teams + ref) at frame 200, lazy-loaded GK anchors
    hunted on the edges when true color outliers appear.
    """

    def __init__(self) -> None:
        self.player_colors: defaultdict[int, list[np.ndarray]] = defaultdict(list)
        self.player_positions: defaultdict[int, list[float]] = defaultdict(list)
        self.player_pitch_x: defaultdict[int, list[float]] = defaultdict(list)   # 2D Radar X
        self.max_history = 90
        self.global_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        # --- DYNAMIC ANCHOR STATE ---
        self.yolo_class_history: defaultdict[int, list[int]] = defaultdict(list)
        self.anchors_extracted = False
        self.color_anchors: dict[str, np.ndarray] = {}
        # INVERTED LOCK: role -> single tracker_id (max one ID per singular role)
        self.locked_role_ids: dict[str, int | None] = {"gk_left": None, "gk_right": None, "referee": None}

    def get_dominant_color(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        """
        Extract dominant jersey color from center-top of bbox (chest) in HSV.
        Crops center-half width and top half height, masks green, K-Means on remaining pixels.
        Returns (H, S) only as numpy array. Bounds clamped to prevent memory corruption.
        """
        h, w = image.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        center_x = (x1 + x2) / 2
        width = x2 - x1
        half_h = (y2 - y1) / 2
        cx1 = int(center_x - width / 4)
        cx2 = int(center_x + width / 4)
        cy1 = int(y1)
        cy2 = int(y1 + half_h)
        cx1, cx2 = max(0, cx1), min(w, cx2)
        cy1, cy2 = max(0, cy1), min(h, cy2)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch, s_ch = hsv_crop[:, :, 0], hsv_crop[:, :, 1]
        green_mask = (
            (h_ch >= HSV_GREEN_H_LOW) & (h_ch <= HSV_GREEN_H_HIGH) & (s_ch > HSV_GREEN_S_MIN)
        )
        non_green = ~green_mask
        if not np.any(non_green):
            return None
        remaining = hsv_crop[non_green]
        if len(remaining) < 5:
            return None
        pixels_hs = remaining[:, :2].astype(np.float64)
        n_clusters = min(2, len(np.unique(pixels_hs, axis=0)))
        if n_clusters < 1:
            return None
        if n_clusters == 1:
            return np.median(pixels_hs, axis=0).astype(np.float64)
        kmeans_local = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans_local.fit(pixels_hs)
        counts = np.bincount(kmeans_local.labels_, minlength=2)
        dominant_idx = int(np.argmax(counts))
        center_hs = kmeans_local.cluster_centers_[dominant_idx]
        return np.array([center_hs[0], center_hs[1]], dtype=np.float64)

    def _extract_base_anchors(self) -> None:
        """Extract exact color profiles for the two teams and referee at kickoff. Lock ref ID."""
        avg_positions: dict[int, float] = {}
        for pid, x_coords in self.player_positions.items():
            if len(x_coords) > 15 and pid in self.player_colors and len(self.player_colors[pid]) > 0:
                avg_positions[pid] = float(np.mean(x_coords))

        if len(avg_positions) < 10:
            return

        ref_id: int | None = None
        max_ref_detections = 0
        for pid, classes in self.yolo_class_history.items():
            ref_count = classes.count(2)  # class_id=2 is referee
            if ref_count > max_ref_detections:
                max_ref_detections = ref_count
                ref_id = pid

        if ref_id is not None and ref_id in self.player_colors and len(self.player_colors[ref_id]) > 0:
            self.color_anchors["referee"] = np.median(self.player_colors[ref_id], axis=0).astype(np.float64)
            self.locked_role_ids["referee"] = ref_id

        outfield_colors: list[np.ndarray] = []
        for pid in avg_positions:
            if pid == ref_id:
                continue
            if pid not in self.player_colors or len(self.player_colors[pid]) == 0:
                continue
            outfield_colors.append(np.median(self.player_colors[pid], axis=0).astype(np.float64))

        if len(outfield_colors) < 2:
            return

        X = np.array(outfield_colors, dtype=np.float64)
        self.global_kmeans.fit(X)
        self.color_anchors["team_0"] = self.global_kmeans.cluster_centers_[0].copy()
        self.color_anchors["team_1"] = self.global_kmeans.cluster_centers_[1].copy()
        self.anchors_extracted = True
        logger.info("Base anchors extracted. Ref ID=%s", ref_id)

    def _hunt_for_gk_anchors(self) -> None:
        """Geographical Hunt: Scans Defensive Thirds with Spatial Threshold Override."""
        if "gk_left" in self.color_anchors and "gk_right" in self.color_anchors:
            return

        avg_pitch_x: dict[int, float] = {}
        for pid, x_coords in self.player_pitch_x.items():
            if len(x_coords) > 5 and pid in self.player_colors:
                avg_pitch_x[pid] = float(np.mean(x_coords))

        def is_true_outlier(pid: int, x_pos: float) -> bool:
            if pid not in self.player_colors or len(self.player_colors[pid]) == 0:
                return False
            player_color = np.median(self.player_colors[pid], axis=0).ravel()
            min_dist_to_known = float("inf")
            for role, anchor_color in self.color_anchors.items():
                if role in ("team_0", "team_1", "referee"):
                    dist = float(np.linalg.norm(player_color - np.asarray(anchor_color).ravel()))
                    min_dist_to_known = min(min_dist_to_known, dist)

            # THE SPATIAL OVERRIDE: 16.5m penalty box = 165 scaled pixels
            threshold = 15.0 if (x_pos < 165.0 or x_pos > 885.0) else 45.0
            return min_dist_to_known > threshold

        # Left Defensive Third (< 35m)
        if "gk_left" not in self.color_anchors:
            left_candidates = [pid for pid, x in avg_pitch_x.items() if x < 350.0]
            left_candidates.sort(key=lambda p: avg_pitch_x[p])
            for pid in left_candidates[:3]:
                if is_true_outlier(pid, avg_pitch_x[pid]):
                    self.color_anchors["gk_left"] = np.median(self.player_colors[pid], axis=0).astype(np.float64)
                    self.locked_role_ids["gk_left"] = pid
                    logger.info("Lazy-loaded Left GK in defensive third: ID=%s", pid)
                    break

        # Right Defensive Third (> 70m)
        if "gk_right" not in self.color_anchors:
            right_candidates = [pid for pid, x in avg_pitch_x.items() if x > 700.0]
            right_candidates.sort(key=lambda p: avg_pitch_x[p], reverse=True)
            for pid in right_candidates[:3]:
                if is_true_outlier(pid, avg_pitch_x[pid]):
                    self.color_anchors["gk_right"] = np.median(self.player_colors[pid], axis=0).astype(np.float64)
                    self.locked_role_ids["gk_right"] = pid
                    logger.info("Lazy-loaded Right GK in defensive third: ID=%s", pid)
                    break

    def predict_frame(
        self,
        image: np.ndarray,
        frame_data: list[dict[str, Any]],
        frame_idx: int,
    ) -> dict[Any, str]:
        """Process the entire frame at once; global draft guarantees at most one tracker_id per singular role."""
        available_player_ids_set: set[int] = set()
        for data in frame_data:
            tid = data["id"]
            bbox = data["bbox"]
            cid = int(data["cid"])
            if tid is None:
                continue
            self.yolo_class_history[tid].append(cid)
            if len(self.yolo_class_history[tid]) > self.max_history:
                self.yolo_class_history[tid].pop(0)

            if cid == CLASS_REF:
                self.locked_role_ids["referee"] = tid
                available_player_ids_set.add(tid)
            elif cid == CLASS_PLAYER:
                available_player_ids_set.add(tid)
                x_center = float(bbox[0] + bbox[2]) / 2
                self.player_positions[tid].append(x_center)
                if len(self.player_positions[tid]) > self.max_history:
                    self.player_positions[tid].pop(0)

                # Track physical pitch X location from radar projection when available
                if data.get("radar_pt") is not None:
                    self.player_pitch_x[tid].append(float(data["radar_pt"][0]))
                    if len(self.player_pitch_x[tid]) > self.max_history:
                        self.player_pitch_x[tid].pop(0)

                # K-Means throttle: need samples until baseline (15), then refresh every 5th frame
                if tid not in self.player_colors or len(self.player_colors[tid]) < 15 or frame_idx % 5 == 0:
                    color = self.get_dominant_color(image, bbox)
                    if color is not None:
                        self.player_colors[tid].append(color)
                        if len(self.player_colors[tid]) > self.max_history:
                            self.player_colors[tid].pop(0)
        available_player_ids = list(available_player_ids_set)

        if frame_idx >= 200 and not self.anchors_extracted:
            self._extract_base_anchors()

        if not self.anchors_extracted:
            return {data["id"]: "unknown" for data in frame_data}

        self._hunt_for_gk_anchors()

        roles: dict[Any, str] = {}

        def draft_singular_role(role_name: str) -> None:
            if role_name not in self.color_anchors:
                return

            # 1. ROAMING: If locked ID is alive, they keep the role regardless of pitch position!
            locked_id = self.locked_role_ids.get(role_name)
            if locked_id is not None and locked_id in available_player_ids:
                roles[locked_id] = role_name
                available_player_ids.remove(locked_id)
                return

            # 2. RE-DRAFTING: Find best replacement with spatial safeguards.
            best_id: int | None = None
            min_dist = float("inf")

            for tid in available_player_ids:
                if tid not in self.player_colors or len(self.player_colors[tid]) == 0:
                    continue

                current_pitch_x = 525.0
                if tid in self.player_pitch_x and len(self.player_pitch_x[tid]) > 0:
                    current_pitch_x = self.player_pitch_x[tid][-1]

                # The Half-Pitch Safeguard
                if role_name == "gk_left" and current_pitch_x > 525.0:
                    continue
                if role_name == "gk_right" and current_pitch_x < 525.0:
                    continue

                player_color = np.median(self.player_colors[tid], axis=0).ravel()
                anchor = np.asarray(self.color_anchors[role_name]).ravel()
                dist = float(np.linalg.norm(player_color - anchor))

                # Spatial Override for Re-Drafting
                dynamic_threshold = 35.0
                if role_name == "gk_left" and current_pitch_x < 165.0:
                    dynamic_threshold = 60.0
                if role_name == "gk_right" and current_pitch_x > 885.0:
                    dynamic_threshold = 60.0

                if dist < min_dist and dist < dynamic_threshold:
                    min_dist = dist
                    best_id = tid

            if best_id is not None:
                roles[best_id] = role_name
                self.locked_role_ids[role_name] = best_id
                available_player_ids.remove(best_id)

        draft_singular_role("referee")
        draft_singular_role("gk_left")
        draft_singular_role("gk_right")

        team_0_anchor = self.color_anchors.get("team_0", np.zeros(2, dtype=np.float64))
        team_1_anchor = self.color_anchors.get("team_1", np.zeros(2, dtype=np.float64))
        for tid in available_player_ids:
            if tid not in self.player_colors or len(self.player_colors[tid]) < 5:
                roles[tid] = "unknown"
                continue
            player_color = np.median(self.player_colors[tid], axis=0).ravel()
            dist_t0 = float(np.linalg.norm(player_color - np.asarray(team_0_anchor).ravel()))
            dist_t1 = float(np.linalg.norm(player_color - np.asarray(team_1_anchor).ravel()))
            # RELAXED THRESHOLD: 65.0 accommodates shadows / compression while limiting jersey mix-ups
            if min(dist_t0, dist_t1) > 65.0:
                roles[tid] = "outlier"
            else:
                roles[tid] = "team_0" if dist_t0 < dist_t1 else "team_1"

        for data in frame_data:
            if data["id"] not in roles:
                roles[data["id"]] = "unknown"
        return roles


class TacticalRadar:
    """
    Panning-aware 2D tactical map using dynamic homographies.
    Unified Architecture: 720p Normalization, Anti-Vortex, and Fail-Safe Clamping.
    """

    def __init__(self, json_path: str | Path | None = None, video_res: tuple[int, int] = (640, 360)) -> None:
        if json_path is None:
            env_h = os.getenv("GAFFERS_HOMOGRAPHY_JSON", "").strip()
            json_path = (
                Path(env_h).expanduser()
                if env_h
                else BACKEND_ROOT / "output" / "match_test_homographies.json"
            )
        self.json_path = Path(json_path)
        self.scale = 10
        self.radar_w = 105 * self.scale
        self.radar_h = 68 * self.scale

        self.video_w = max(video_res[0], 1)
        self.video_h = max(video_res[1], 1)

        # 1. THE CRITICAL CALIBRATION LOCK
        # SoccerNet matrices are native to 1280x720.
        # (Using 1920x1080 here causes a 3x multiplier that shoots players to X=-15000).
        self.calib_w = 1280
        self.calib_h = 720

        self.inv_homographies: dict[int, np.ndarray] = {}
        self.current_frame_idx = 0

        with open(self.json_path, "r") as f:
            data = json.load(f)

        for item in data["homographies"]:
            matrix = np.array(item["homography"], dtype=np.float64)
            try:
                self.inv_homographies[item["frame"]] = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                pass

        self.available_frames = sorted(self.inv_homographies.keys())
        logger.info("Tactical Radar loaded and inverted %d dynamic camera matrices", len(self.available_frames))

    def update_camera_angle(self, current_frame_idx: int) -> None:
        self.current_frame_idx = current_frame_idx

    def draw_blank_pitch(self) -> np.ndarray:
        pitch = np.ones((self.radar_h, self.radar_w, 3), dtype=np.uint8) * np.array([40, 130, 40], dtype=np.uint8)
        white = (255, 255, 255)
        thickness = 2
        mid_x = self.radar_w // 2
        center_y = self.radar_h // 2

        cv2.rectangle(pitch, (0, 0), (self.radar_w, self.radar_h), white, thickness)
        cv2.line(pitch, (mid_x, 0), (mid_x, self.radar_h), white, thickness)
        cv2.circle(pitch, (mid_x, center_y), int(9.15 * self.scale), white, thickness)
        cv2.circle(pitch, (mid_x, center_y), max(2, int(0.4 * self.scale)), white, -1)

        pen_w, pen_h = int(16.5 * self.scale), int(40.32 * self.scale)
        pen_y = (self.radar_h - pen_h) // 2
        cv2.rectangle(pitch, (0, pen_y), (pen_w, pen_y + pen_h), white, thickness)
        cv2.rectangle(pitch, (self.radar_w - pen_w, pen_y), (self.radar_w, pen_y + pen_h), white, thickness)

        goal_w, goal_h = int(5.5 * self.scale), int(18.32 * self.scale)
        goal_y = (self.radar_h - goal_h) // 2
        cv2.rectangle(pitch, (0, goal_y), (goal_w, goal_y + goal_h), white, thickness)
        cv2.rectangle(pitch, (self.radar_w - goal_w, goal_y), (self.radar_w, goal_y + goal_h), white, thickness)

        spot_r = max(2, int(0.4 * self.scale))
        cv2.circle(pitch, (int(11 * self.scale), center_y), spot_r, white, -1)
        cv2.circle(pitch, (self.radar_w - int(11 * self.scale), center_y), spot_r, white, -1)
        return pitch

    def map_to_2d(self, bbox: np.ndarray) -> tuple[int, int] | None:
        if not self.available_frames:
            return None

        # 1. RESOLUTION NORMALIZER
        scale_x = self.calib_w / self.video_w
        scale_y = self.calib_h / self.video_h

        x_center = ((bbox[0] + bbox[2]) / 2.0) * scale_x
        y_bottom = (bbox[3]) * scale_y

        point = np.array([[[x_center, y_bottom]]], dtype=np.float32)

        def project(inv_matrix: np.ndarray) -> tuple[float, float]:
            try:
                mapped = cv2.perspectiveTransform(point, inv_matrix)
                return float(mapped[0][0][0]), float(mapped[0][0][1])
            except cv2.error:
                return -9999.0, -9999.0

        f_idx = self.current_frame_idx

        # 2. ANTI-VORTEX INTERPOLATION
        if f_idx in self.inv_homographies:
            x_raw, y_raw = project(self.inv_homographies[f_idx])
        else:
            before = [f for f in self.available_frames if f < f_idx]
            after = [f for f in self.available_frames if f > f_idx]

            if not before:
                x_raw, y_raw = project(self.inv_homographies[after[0]])
            elif not after:
                x_raw, y_raw = project(self.inv_homographies[before[-1]])
            else:
                f0, f1 = before[-1], after[0]
                # VORTEX GUARD: Do not interpolate across camera cuts
                if f1 - f0 > 10:
                    x_raw, y_raw = project(self.inv_homographies[f0])
                else:
                    x0, y0 = project(self.inv_homographies[f0])
                    x1, y1 = project(self.inv_homographies[f1])
                    weight = (f_idx - f0) / float(f1 - f0)
                    x_raw = x0 + weight * (x1 - x0)
                    y_raw = y0 + weight * (y1 - y0)

        if x_raw == -9999.0:
            return None

        # 3. RESTORE METER-TO-PIXEL CONVERSION
        # CRITICAL FIX: The matrix outputs physical meters, not pixels!
        # We must add 52.5m (half pitch) and multiply by scale (10 pixels/meter).
        radar_x = int(round((x_raw + 52.5) * self.scale))
        radar_y = int(round((y_raw + 34.0) * self.scale))

        # 4. FAIL-SAFE CLAMP
        radar_x = max(0, min(self.radar_w, radar_x))
        radar_y = max(0, min(self.radar_h, radar_y))

        return (radar_x, radar_y)


def role_to_payload(prediction: str) -> int | tuple[int, bool] | None:
    """Map Global Draft role string to annotation payload: -1, (team_id, is_gk), or None (ball)."""
    if prediction == "team_0":
        return (0, False)
    if prediction == "team_1":
        return (1, False)
    if prediction == "gk_left":
        return (0, True)
    if prediction == "gk_right":
        return (1, True)
    return -1  # unknown, referee, outlier


def get_detection_color(
    class_id: int,
    payload: int | tuple[int, bool] | None,
) -> tuple[int, int, int]:
    """Return BGR color: Team 0/1 for players, gray Unknown, yellow GK, default Ball/Ref."""
    if class_id == CLASS_PLAYER:
        if payload is None or payload == -1:
            return COLOR_UNKNOWN
        if isinstance(payload, tuple):
            team_id, is_gk = payload
            if is_gk:
                return COLOR_GOALKEEPER
            return TEAM_COLORS.get(team_id, (255, 255, 255))
    if class_id == CLASS_BALL:
        return COLOR_BALL
    if class_id == CLASS_REF:
        return COLOR_REF
    return COLOR_UNKNOWN


def get_detection_label(
    class_id: int,
    payload: int | tuple[int, bool] | None,
    tracker_id: int | None = None,
) -> str:
    """Return label string; for players optionally include ID and team (e.g. ID:3 T0-GK)."""
    if class_id == CLASS_PLAYER:
        if payload is None or payload == -1:
            return "Unknown" if tracker_id is None else f"ID:{tracker_id} Unknown"
        if isinstance(payload, tuple):
            team_id, is_gk = payload
            if tracker_id is not None:
                return f"ID:{tracker_id} T{team_id}-GK" if is_gk else f"ID:{tracker_id} T{team_id}"
            return f"T{team_id}-GK" if is_gk else f"Team {team_id}"
    if class_id == CLASS_BALL:
        return "Ball"
    if class_id == CLASS_REF:
        return "Ref"
    return "Other"


def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    team_ids: list[int | tuple[int, bool] | None],
    tracker_ids: list[int] | None = None,
) -> np.ndarray:
    """Draw boxes and labels with per-detection colors and labels."""
    n = len(detections)
    colors = [get_detection_color(int(detections.class_id[i]), team_ids[i]) for i in range(n)]
    tid_at = (lambda i: int(tracker_ids[i]) if tracker_ids is not None and i < len(tracker_ids) and tracker_ids[i] is not None else None)
    labels = [
        get_detection_label(int(detections.class_id[i]), team_ids[i], tid_at(i))
        for i in range(n)
    ]
    # Supervision expects Color for BoxAnnotator; we pass a list of (B,G,R) as ColorPalette or draw manually.
    # Draw boxes and labels manually for per-detection color/label.
    annotated = frame.copy()
    for i in range(n):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        color = colors[i]
        label = labels[i]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return annotated


def main() -> None:
    """Load model, run YOLO + ByteTrack + team classification, write output video."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found at {VIDEO_PATH}")

    logger.info("Loading model from %s", MODEL_PATH)
    model: YOLO = YOLO(str(MODEL_PATH))
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    classifier = TeamClassifier()
    healer = HybridIDHealer()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %dx%d @ %d fps, %d frames", width, height, fps, total_frames)

    radar = TacticalRadar(video_res=(width, height))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Side-by-side layout: video left, radar right; no overlap
    output_height = max(height, radar.radar_h)
    output_width = width + radar.radar_w
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output writer: {OUTPUT_PATH}")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Standard inference
            results: list[Any] = model(frame, conf=0.3, verbose=False)
            if not results:
                radar.update_camera_angle(frame_idx)
                current_radar = radar.draw_blank_pitch()
                composed = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                composed[0:height, 0:width] = frame
                composed[0 : radar.radar_h, width : width + radar.radar_w] = current_radar
                out.write(composed)
                frame_idx += 1
                continue
            r0 = results[0]
            detections = sv.Detections.from_ultralytics(r0)
            detections = tracker.update_with_detections(detections)

            radar.update_camera_angle(frame_idx)
            radar_pts: list[tuple[int, int] | None] = [
                radar.map_to_2d(detections.xyxy[i]) for i in range(len(detections))
            ]

            tracker_ids = healer.process_and_heal(detections, frame, radar_pts, frame_idx)

            if tracker_ids is None:
                tracker_ids = getattr(detections, "tracker_id", None)
            frame_data: list[dict[str, Any]] = []
            for i in range(len(detections)):
                tid = None
                if tracker_ids is not None and i < len(tracker_ids):
                    t = tracker_ids[i]
                    tid = int(t) if t is not None else None

                frame_data.append(
                    {
                        "id": tid,
                        "bbox": detections.xyxy[i],
                        "cid": int(detections.class_id[i]),
                        "radar_pt": radar_pts[i],
                    },
                )

            # Global Draft: one prediction per tracker per frame
            role_mapping = classifier.predict_frame(frame, frame_data, frame_idx)

            # Build team_ids for annotation (same order as detections)
            team_ids: list[int | tuple[int, bool] | None] = [None] * len(detections)
            for i, data in enumerate(frame_data):
                tid, cid = data["id"], data["cid"]
                if cid == CLASS_BALL:
                    prediction = "ball"
                else:
                    prediction = role_mapping.get(tid, "unknown")
                if cid == CLASS_BALL:
                    team_ids[i] = None
                else:
                    team_ids[i] = role_to_payload(prediction)

            annotated = annotate_frame(frame, detections, team_ids, tracker_ids)

            # Tactical radar: blank pitch every frame, then plot every player (all role predictions)
            current_radar = radar.draw_blank_pitch()
            for data in frame_data:
                tid = data["id"]
                cid = data["cid"]
                if tid is None or cid != CLASS_PLAYER:
                    continue
                prediction = role_mapping.get(tid, "unknown")
                pt = data["radar_pt"]
                if pt is None:
                    continue
                if prediction == "team_0":
                    color = TEAM_COLORS[0]
                elif prediction == "team_1":
                    color = TEAM_COLORS[1]
                elif prediction in ("gk_left", "gk_right"):
                    color = COLOR_GOALKEEPER
                elif prediction == "referee":
                    color = COLOR_REF
                else:
                    color = COLOR_UNKNOWN
                cv2.circle(current_radar, pt, 5, color, -1)
                cv2.circle(current_radar, pt, 5, (0, 0, 0), 1)
                cv2.putText(
                    current_radar,
                    str(tid),
                    (pt[0] - 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

            # Side-by-side: video left, radar right
            composed = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            composed[0:height, 0:width] = annotated
            composed[0 : radar.radar_h, width : width + radar.radar_w] = current_radar
            out.write(composed)
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info("Processed %d/%d frames", frame_idx, total_frames)
            if frame_idx > 0 and frame_idx % 300 == 0:
                healer.cleanup_ghost_ids(frame_idx)
                logger.info("Executed Healer garbage collection at frame %d", frame_idx)

    finally:
        cap.release()
        out.release()
        logger.info("Saved output to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
