"""
Team-aware tracking: YOLO + ByteTrack + team classification with soft lock and goalkeeper detection.

Uses HSV dominant color, 90-frame sliding window for auto-correction (no permanent mistake trap),
and distance-based outlier detection to isolate goalkeepers. Global K-Means fits when 10+ players
have 15+ color observations; players stay gray until 15 frames observed.

Setup: pip install scikit-learn supervision ultralytics opencv-python

Execution: python backend/scripts/track_teams.py
"""

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
    Classify players into Team 0, Team 1, or Goalkeeper (2) using HSV dominant color.
    90-frame sliding window enables auto-correction; distance threshold isolates goalkeepers.
    """

    def __init__(self) -> None:
        self.player_colors: defaultdict[int, list[np.ndarray]] = defaultdict(list)
        self.max_history = 90
        self.global_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.is_kmeans_fitted = False
        self.known_gks: set[int] = set()

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

    def update_and_predict(
        self, tracker_id: int, image: np.ndarray, bbox: np.ndarray
    ) -> int | tuple[int, bool]:
        """
        Extract dominant color, update 90-frame sliding window, then predict.
        Returns -1 (Unknown) or (team_id, is_gk). GK capped at 2 via known_gks; distance > 55.0 + cap.
        """
        color = self.get_dominant_color(image, bbox)
        if color is not None:
            history = self.player_colors[tracker_id]
            history.append(color)
            if len(history) > self.max_history:
                history.pop(0)
            self.player_colors[tracker_id] = history

        # Rule 1: Fit global model once when 10+ players have 15+ colors each
        if not self.is_kmeans_fitted:
            eligible = [
                tid for tid, colors in self.player_colors.items()
                if len(colors) >= MIN_COLORS_BEFORE_PREDICT
            ]
            if len(eligible) >= MIN_PLAYERS_TO_FIT:
                gathered = []
                for tid in eligible:
                    arr = np.array(self.player_colors[tid], dtype=np.float64)
                    median_hs = np.median(arr, axis=0)
                    gathered.append(median_hs)
                X = np.array(gathered, dtype=np.float64)
                self.global_kmeans.fit(X)
                self.is_kmeans_fitted = True
                logger.info(
                    "TeamClassifier: fitted global K-Means on %d player median colors",
                    X.shape[0],
                )

        # Rule 2: Initial wait (buffer)
        history = self.player_colors[tracker_id]
        if len(history) < MIN_COLORS_BEFORE_PREDICT or not self.is_kmeans_fitted:
            return -1

        # Predict base team and distance to team center
        arr = np.array(history, dtype=np.float64)
        player_median_color = np.median(arr, axis=0)
        if player_median_color.ndim == 0:
            player_median_color = np.atleast_1d(player_median_color)
        median_2d = np.reshape(player_median_color, (1, -1))
        team_id = int(self.global_kmeans.predict(median_2d)[0])
        team_center = self.global_kmeans.cluster_centers_[team_id]
        distance = float(np.linalg.norm(player_median_color - team_center))

        # GK logic: extreme color difference AND cap at 2 GKs
        is_gk = False
        if tracker_id in self.known_gks:
            is_gk = True
        elif distance > 55.0 and len(self.known_gks) < 2:
            self.known_gks.add(tracker_id)
            is_gk = True

        return (team_id, is_gk)


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

            # Per-detection: None (non-player), -1 (unknown), or (team_id, is_gk)
            team_ids: list[int | tuple[int, bool] | None] = [None] * len(detections)
            tracker_ids = getattr(detections, "tracker_id", None)

            for i in range(len(detections)):
                cls_id = int(detections.class_id[i])
                if cls_id != CLASS_PLAYER:
                    continue
                bbox = detections.xyxy[i]
                x1, y1, x2, y2 = map(int, bbox)
                frame_h, frame_w = frame.shape[:2]
                # Ignore detections in top 5% or bottom 10% (touchline staff)
                if y1 < (frame_h * 0.05) or y2 > (frame_h * 0.90):
                    continue
                tid = None
                if tracker_ids is not None and i < len(tracker_ids):
                    t = tracker_ids[i]
                    tid = int(t) if t is not None else None
                if tid is None:
                    team_ids[i] = -1
                    continue
                prediction = classifier.update_and_predict(tid, frame, bbox)
                if prediction == -1:
                    team_id, is_gk = -1, False
                else:
                    team_id, is_gk = prediction
                if team_id == -1 and not is_gk:
                    team_ids[i] = -1
                else:
                    team_ids[i] = (team_id, is_gk)

            annotated = annotate_frame(frame, detections, team_ids, tracker_ids)
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
