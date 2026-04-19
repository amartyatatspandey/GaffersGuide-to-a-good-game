"""Team classification module extracted from track_teams."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .track_teams_constants import (
    CLASS_PLAYER,
    CLASS_REF,
    HSV_GREEN_H_HIGH,
    HSV_GREEN_H_LOW,
    HSV_GREEN_S_MIN,
)

logger = logging.getLogger(__name__)

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


