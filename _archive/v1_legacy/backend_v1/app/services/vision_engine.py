"""
Vision engine: YOLOv8 + ByteTrack, pitch homography, team classification, SkillCorner JSONL.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.services.team_classifier import TeamClassifier
from app.utils import homography
from app.utils import visualizer

logger = logging.getLogger(__name__)

RADAR_SCALE: int = 5
COLOR_HOME: Tuple[int, int, int] = (0, 0, 255)
COLOR_AWAY: Tuple[int, int, int] = (255, 0, 0)


class VisionEngine:
    PERSON_CLASS: int = 0
    BALL_CLASS: int = 32

    def __init__(
        self,
        source_path: str | Path,
        output_path: str | Path,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.confidence_threshold = confidence_threshold
        self.player_model = YOLO("yolov8x.pt")
        backend_root = Path(__file__).resolve().parent.parent.parent
        pitch_weights = backend_root / "models" / "pitch_pose_v1" / "weights" / "best.pt"
        self.pitch_model: YOLO | None = None
        if pitch_weights.exists():
            self.pitch_model = YOLO(str(pitch_weights))
        else:
            logger.warning("Pitch model not found at %s", pitch_weights)
        self.byte_track = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.classifier = TeamClassifier()
        self.calibration_crops: List[np.ndarray] = []
        self._teams_fitted: bool = False
        (backend_root / "data").mkdir(parents=True, exist_ok=True)
        self.json_file = open(backend_root / "data" / "generated_match.jsonl", "w")

    def _get_pitch_corners(self, frame: np.ndarray) -> np.ndarray | None:
        if self.pitch_model is None:
            return None
        results = self.pitch_model(frame, verbose=False)[0]
        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return None
        kp = results.keypoints.xy[0].cpu().numpy()
        if kp.shape[0] < 4:
            return None
        return np.asarray(kp[:4], dtype=np.float32)

    def _get_feet_points(self, results: Any) -> List[Tuple[float, float]]:
        feet: List[Tuple[float, float]] = []
        if results.boxes is None:
            return feet
        xyxy = results.boxes.xyxy.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()
        for i, c in enumerate(cls):
            if int(c) != self.PERSON_CLASS:
                continue
            x1, y1, x2, y2 = xyxy[i]
            feet.append(((x1 + x2) / 2.0, float(y2)))
        return feet

    def _format_timestamp(self, frame_idx: int, fps: float) -> str:
        total_sec = frame_idx / fps if fps > 0 else 0.0
        minutes = int(total_sec // 60)
        secs = total_sec % 60
        return f"00:{minutes:02d}:{secs:06.3f}"

    def _get_person_crops_feet_tracker(
        self, frame: np.ndarray, results: Any, detections: sv.Detections
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[int]]:
        crops: List[np.ndarray] = []
        feet_px: List[Tuple[float, float]] = []
        tracker_ids: List[int] = []
        if detections.tracker_id is None:
            return crops, feet_px, tracker_ids
        for i in range(len(detections)):
            if int(detections.class_id[i]) != self.PERSON_CLASS:
                continue
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(frame[y1:y2, x1:x2].copy())
            feet_px.append(( (float(x1) + float(x2)) / 2.0, float(y2) ))
            tracker_ids.append(int(detections.tracker_id[i]))
        return crops, feet_px, tracker_ids

    def process_video(self) -> None:
        cap: cv2.VideoCapture | None = None
        out: cv2.VideoWriter | None = None
        try:
            cap = cv2.VideoCapture(str(self.source_path))
            if not cap.isOpened():
                logger.error("Could not open source video: %s", self.source_path)
                return

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            radar_h = visualizer.PITCH_WIDTH * RADAR_SCALE
            radar_w = visualizer.PITCH_LENGTH * RADAR_SCALE
            stacked_w = w + int(radar_w * h / radar_h)
            stacked_h = h
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (stacked_w, stacked_h))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                corners = self._get_pitch_corners(frame)
                H = homography.calculate_homography(corners) if corners is not None else None

                player_results = self.player_model(
                    frame,
                    classes=[self.PERSON_CLASS, self.BALL_CLASS],
                    conf=self.confidence_threshold,
                    verbose=False,
                )[0]
                detections = sv.Detections.from_ultralytics(player_results)
                detections = self.byte_track.update_with_detections(detections)
                crops, feet_px, tracker_ids = self._get_person_crops_feet_tracker(
                    frame, player_results, detections
                )

                if frame_idx <= 60:
                    self.calibration_crops.extend(crops)
                    if frame_idx == 60:
                        self.classifier.fit(self.calibration_crops)
                        self._teams_fitted = True
                        logger.info("Teams Classified")

                feet_m: List[Tuple[float, float]] = []
                if H is not None and feet_px:
                    pts_m = homography.transform_points(feet_px, H)
                    feet_m = [tuple(float(x) for x in row) for row in pts_m]

                team_colors: List[Tuple[int, int, int]] = []
                if frame_idx >= 61:
                    player_list = []
                    if self._teams_fitted and crops and len(feet_m) == len(crops):
                        for i, (crop, tid) in enumerate(zip(crops, tracker_ids)):
                            team_id = self.classifier.predict(crop)
                            group_name = "home" if team_id == 0 else "away"
                            team_colors.append(COLOR_HOME if team_id == 0 else COLOR_AWAY)
                            real_x = feet_m[i][0] if i < len(feet_m) else 0.0
                            real_y = feet_m[i][1] if i < len(feet_m) else 0.0
                            player_list.append({
                                "trackable_object": tid,
                                "group_name": group_name,
                                "x": real_x,
                                "y": real_y,
                            })
                    else:
                        team_colors = [COLOR_HOME] * len(feet_m) if feet_m else []
                    self.json_file.write(json.dumps({
                        "timestamp": self._format_timestamp(frame_idx, fps),
                        "data": player_list,
                    }) + "\n")
                else:
                    team_colors = [COLOR_HOME] * len(feet_m) if feet_m else []

                pitch_img = visualizer.draw_pitch(scale=RADAR_SCALE)
                radar_img = visualizer.draw_radar(pitch_img, feet_m, color=team_colors, scale=RADAR_SCALE)
                radar_resized = cv2.resize(
                    radar_img, (int(radar_w * h / radar_h), h), interpolation=cv2.INTER_LINEAR
                )
                stacked = cv2.hconcat([frame, radar_resized])
                out.write(stacked)
                frame_idx += 1

            logger.info("Saved V2 radar video to %s", self.output_path)
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            if hasattr(self, "json_file") and self.json_file and not self.json_file.closed:
                self.json_file.close()
            logger.info("Resources released and video file closed successfully.")


if __name__ == "__main__":
    backend_root = Path(__file__).resolve().parent.parent.parent
    source = backend_root / "data" / "match_test.mp4"
    output = backend_root / "output" / "gaffer_v2_radar.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = VisionEngine(source_path=source, output_path=output, confidence_threshold=0.3)
    engine.process_video()
