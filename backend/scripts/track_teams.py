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
from collections import defaultdict
from pathlib import Path
from typing import Any

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


class TeamClassifier:
    """
    Dynamic Anchor architecture: base anchors (teams + ref) at frame 200, lazy-loaded GK anchors
    hunted on the edges when true color outliers appear.
    """

    def __init__(self) -> None:
        self.player_colors: defaultdict[int, list[np.ndarray]] = defaultdict(list)
        self.player_positions: defaultdict[int, list[float]] = defaultdict(list)
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
        """Continuously scan the edges for true outliers; lazy-load GK anchors and strict-lock their IDs."""
        if "gk_left" in self.color_anchors and "gk_right" in self.color_anchors:
            return

        avg_positions: dict[int, float] = {}
        for pid, x_coords in self.player_positions.items():
            if len(x_coords) > 5 and pid in self.player_colors:
                avg_positions[pid] = float(np.mean(x_coords))

        if len(avg_positions) < 4:
            return

        sorted_pids = sorted(avg_positions.keys(), key=lambda p: avg_positions[p])

        def is_true_outlier(pid: int) -> bool:
            if pid not in self.player_colors or len(self.player_colors[pid]) == 0:
                return False
            player_color = np.median(self.player_colors[pid], axis=0).ravel()
            min_dist_to_known = float("inf")
            for role, anchor_color in self.color_anchors.items():
                if role in ("team_0", "team_1", "referee"):
                    dist = float(np.linalg.norm(player_color - np.asarray(anchor_color).ravel()))
                    min_dist_to_known = min(min_dist_to_known, dist)
            return min_dist_to_known > 45.0

        if "gk_left" not in self.color_anchors:
            for pid in sorted_pids[:3]:
                if is_true_outlier(pid):
                    self.color_anchors["gk_left"] = np.median(self.player_colors[pid], axis=0).astype(np.float64)
                    self.locked_role_ids["gk_left"] = pid
                    logger.info("Lazy-loaded left GK anchor: ID=%s", pid)
                    break

        if "gk_right" not in self.color_anchors:
            for pid in sorted_pids[-3:]:
                if is_true_outlier(pid):
                    self.color_anchors["gk_right"] = np.median(self.player_colors[pid], axis=0).astype(np.float64)
                    self.locked_role_ids["gk_right"] = pid
                    logger.info("Lazy-loaded right GK anchor: ID=%s", pid)
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

        def draft_singular_role(role_name: str, threshold: float = 35.0) -> None:
            if role_name not in self.color_anchors:
                return
            locked_id = self.locked_role_ids.get(role_name)
            if locked_id is not None and locked_id in available_player_ids:
                roles[locked_id] = role_name
                available_player_ids.remove(locked_id)
                return
            best_id: int | None = None
            min_dist = float("inf")
            for tid in available_player_ids:
                if tid not in self.player_colors or len(self.player_colors[tid]) == 0:
                    continue
                player_color = np.median(self.player_colors[tid], axis=0).ravel()
                anchor = np.asarray(self.color_anchors[role_name]).ravel()
                dist = float(np.linalg.norm(player_color - anchor))
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
            if best_id is not None and min_dist < threshold:
                roles[best_id] = role_name
                self.locked_role_ids[role_name] = best_id
                available_player_ids.remove(best_id)

        draft_singular_role("referee", threshold=45.0)
        draft_singular_role("gk_left", threshold=35.0)
        draft_singular_role("gk_right", threshold=35.0)

        team_0_anchor = self.color_anchors.get("team_0", np.zeros(2, dtype=np.float64))
        team_1_anchor = self.color_anchors.get("team_1", np.zeros(2, dtype=np.float64))
        for tid in available_player_ids:
            if tid not in self.player_colors or len(self.player_colors[tid]) < 5:
                roles[tid] = "unknown"
                continue
            player_color = np.median(self.player_colors[tid], axis=0).ravel()
            dist_t0 = float(np.linalg.norm(player_color - np.asarray(team_0_anchor).ravel()))
            dist_t1 = float(np.linalg.norm(player_color - np.asarray(team_1_anchor).ravel()))
            if min(dist_t0, dist_t1) > 45.0:
                roles[tid] = "outlier"
            else:
                roles[tid] = "team_0" if dist_t0 < dist_t1 else "team_1"

        for data in frame_data:
            if data["id"] not in roles:
                roles[data["id"]] = "unknown"
        return roles


class TacticalRadar:
    """
    Panning-aware 2D tactical map using dynamic homographies from SoccerNet calibration.
    Loads homography JSON (pitch -> image), inverts per frame for image -> pitch mapping.
    """

    def __init__(
        self,
        json_path: str | Path | None = None,
    ) -> None:
        if json_path is None:
            json_path = BACKEND_ROOT / "output" / "match_test_homographies.json"
        self.json_path = Path(json_path)
        self.scale = 10
        self.radar_w = 105 * self.scale
        self.radar_h = 68 * self.scale
        self.homographies: dict[int, np.ndarray] = {}
        self.current_matrix_inv: np.ndarray | None = None

        with open(self.json_path, "r") as f:
            data = json.load(f)
        for item in data["homographies"]:
            self.homographies[item["frame"]] = np.array(item["homography"], dtype=np.float64)
        self.available_frames = sorted(self.homographies.keys())
        logger.info("Tactical Radar loaded %d dynamic camera matrices from %s", len(self.available_frames), self.json_path)

    def update_camera_angle(self, current_frame_idx: int) -> None:
        """Find the closest dynamic homography for the current frame and set inverse (image -> pitch)."""
        if not self.available_frames:
            return
        closest_frame = min(self.available_frames, key=lambda x: abs(x - current_frame_idx))
        matrix = self.homographies[closest_frame]
        try:
            self.current_matrix_inv = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            self.current_matrix_inv = None

    def map_to_2d(self, bbox: np.ndarray) -> tuple[int, int] | None:
        """
        Map image bbox (feet) to radar pixel (x, y). Uses current_matrix_inv (image -> pitch).
        SoccerNet pitch is centered at (0,0) in meters; we shift to top-left origin then scale.
        """
        if self.current_matrix_inv is None:
            return None
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = bbox[3]
        point = np.array([[[x_center, y_bottom]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.current_matrix_inv)
        sn_x_meters = float(mapped[0][0][0])
        sn_y_meters = float(mapped[0][0][1])
        radar_x_meters = sn_x_meters + (105.0 / 2.0)
        radar_y_meters = sn_y_meters + (68.0 / 2.0)
        radar_x = int(radar_x_meters * self.scale)
        radar_y = int(radar_y_meters * self.scale)
        if not (0 <= radar_x < self.radar_w and 0 <= radar_y < self.radar_h):
            return None
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
    radar = TacticalRadar()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %dx%d @ %d fps, %d frames", width, height, fps, total_frames)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # CRITICAL MAC FIX: Force 'avc1' (H.264) codec; QuickTime corrupts mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))
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
                out.write(frame)
                frame_idx += 1
                continue
            r0 = results[0]
            detections = sv.Detections.from_ultralytics(r0)
            detections = tracker.update_with_detections(detections)

            # --- PASS 1: Collect frame data ---
            tracker_ids = getattr(detections, "tracker_id", None)
            frame_data: list[dict[str, Any]] = []
            for i in range(len(detections)):
                tid = None
                if tracker_ids is not None and i < len(tracker_ids):
                    t = tracker_ids[i]
                    tid = int(t) if t is not None else None
                frame_data.append({
                    "id": tid,
                    "bbox": detections.xyxy[i],
                    "cid": int(detections.class_id[i]),
                })

            # Global Draft: one prediction per tracker per frame
            role_mapping = classifier.predict_frame(frame, frame_data, frame_idx)

            # Dynamic homography: use nearest frame's matrix for image -> pitch
            radar.update_camera_angle(frame_idx)

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

            # Tactical radar overlay: only team_0 / team_1 (GK/Ref excluded per spec)
            radar_canvas = np.zeros((radar.radar_h, radar.radar_w, 3), dtype=np.uint8)
            radar_canvas[:] = (40, 40, 40)
            cv2.rectangle(radar_canvas, (0, 0), (radar.radar_w - 1, radar.radar_h - 1), (80, 80, 80), 1)
            for i, data in enumerate(frame_data):
                prediction = role_mapping.get(data["id"], "unknown")
                if prediction not in ("team_0", "team_1"):
                    continue
                pt = radar.map_to_2d(data["bbox"])
                if pt is None:
                    continue
                color = TEAM_COLORS[0] if prediction == "team_0" else TEAM_COLORS[1]
                cv2.circle(radar_canvas, pt, 4, color, -1)
            # Overlay radar on top-right of frame
            overlay_h, overlay_w = min(radar.radar_h, height // 2), min(radar.radar_w, width // 2)
            radar_small = cv2.resize(radar_canvas, (overlay_w, overlay_h))
            x0 = width - overlay_w - 10
            y0 = 10
            annotated[y0 : y0 + overlay_h, x0 : x0 + overlay_w] = radar_small

            out.write(annotated)
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info("Processed %d/%d frames", frame_idx, total_frames)

    finally:
        cap.release()
        out.release()
        logger.info("Saved output to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
