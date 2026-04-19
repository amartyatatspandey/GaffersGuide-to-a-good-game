"""
Team-aware tracking: YOLO + ByteTrack + team classification with soft lock and goalkeeper detection.

Uses HSV dominant color, 90-frame sliding window for auto-correction (no permanent mistake trap),
and distance-based outlier detection to isolate goalkeepers. Global K-Means fits when 10+ players
have 15+ color observations; players stay gray until 15 frames observed.

Setup: pip install scikit-learn supervision ultralytics opencv-python

Execution: ``cd backend && python -m scripts.pipeline_core.track_teams``
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
_backend_root = SCRIPT_DIR.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))
# Ensure sibling modules (e.g. reid_healer) resolve when run as a script or from another cwd.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from services.pipeline_paths import (  # noqa: E402
    BACKEND_ROOT,
    format_tracking_model_missing_reason,
    tracking_model_weights_path,
)
from services.homography_provider import default_homography_provider  # noqa: E402

from .track_teams_constants import (  # noqa: E402
    CLASS_BALL,
    CLASS_PLAYER,
    CLASS_REF,
    COLOR_BALL,
    COLOR_GOALKEEPER,
    COLOR_REF,
    COLOR_TEAM_0,
    COLOR_TEAM_1,
    COLOR_UNKNOWN,
    HSV_GREEN_H_HIGH,
    HSV_GREEN_H_LOW,
    HSV_GREEN_S_MIN,
    MIN_COLORS_BEFORE_PREDICT,
    MIN_PLAYERS_TO_FIT,
    TEAM_COLORS,
    cosine_similarity,
)
from .track_teams_reid_hybrid import HybridIDHealer  # noqa: E402

MODEL_PATH = tracking_model_weights_path()
VIDEO_PATH = BACKEND_ROOT / "data" / "match_test.mp4"
OUTPUT_PATH = BACKEND_ROOT / "output" / "tracking_teams.mp4"



from .team_classifier import TeamClassifier  # noqa: E402
from .tactical_radar import TacticalRadar  # noqa: E402

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
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. {format_tracking_model_missing_reason(MODEL_PATH)}"
        )
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

    homography_json = default_homography_provider().ensure_homography_json(VIDEO_PATH)
    radar = TacticalRadar(json_path=homography_json, video_res=(width, height))

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
