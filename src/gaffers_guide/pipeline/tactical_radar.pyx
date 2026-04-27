# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

"""Tactical radar projection module extracted from track_teams."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class TacticalRadar:
    """
    Panning-aware 2D tactical map using dynamic homographies.
    Unified Architecture: 720p Normalization, Anti-Vortex, and Fail-Safe Clamping.
    """

    def __init__(self, json_path: str | Path | None = None, video_res: tuple[int, int] = (640, 360)) -> None:
        if json_path is None:
            env_h = os.getenv("GAFFERS_HOMOGRAPHY_JSON", "").strip()
            if env_h:
                json_path = Path(env_h).expanduser()
            else:
                raise ValueError(
                    "TacticalRadar requires json_path=... or set GAFFERS_HOMOGRAPHY_JSON. "
                    "Default per-video file: backend/output/{video_stem}_homographies.json "
                    "(generate with scripts/run_calibrator_on_video.py)."
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

    def map_many_to_2d(
        self,
        bboxes_xyxy: np.ndarray | list[np.ndarray],
        *,
        frame_idx: int | None = None,
    ) -> list[tuple[int, int] | None]:
        """
        Vectorized projection for many bboxes in one frame.

        Safety guarantees:
        - Empty input returns [].
        - OpenCV input shape is (N, 1, 2), dtype float32.
        - Output index order matches input order exactly.
        """
        if len(bboxes_xyxy) == 0:
            return []
        if not self.available_frames:
            return [None] * len(bboxes_xyxy)

        f_idx = self.current_frame_idx if frame_idx is None else int(frame_idx)
        scale_x = self.calib_w / self.video_w
        scale_y = self.calib_h / self.video_h

        b = np.asarray(bboxes_xyxy, dtype=np.float32)
        if b.ndim != 2 or b.shape[1] != 4:
            return [None] * len(bboxes_xyxy)

        x_center = ((b[:, 0] + b[:, 2]) / 2.0) * scale_x
        y_bottom = b[:, 3] * scale_y
        # OpenCV shape rule: (N, 1, 2) float32
        points = np.stack([x_center, y_bottom], axis=1).astype(np.float32).reshape(-1, 1, 2)
        if len(points) == 0:
            return []

        def project_many(inv_matrix: np.ndarray) -> np.ndarray:
            try:
                mapped = cv2.perspectiveTransform(points, inv_matrix)
                # Flatten back to (N, 2)
                return mapped.reshape(-1, 2).astype(np.float64)
            except cv2.error:
                return np.full((len(points), 2), -9999.0, dtype=np.float64)

        if f_idx in self.inv_homographies:
            mapped_xy = project_many(self.inv_homographies[f_idx])
        else:
            before = [f for f in self.available_frames if f < f_idx]
            after = [f for f in self.available_frames if f > f_idx]
            if not before:
                mapped_xy = project_many(self.inv_homographies[after[0]])
            elif not after:
                mapped_xy = project_many(self.inv_homographies[before[-1]])
            else:
                f0, f1 = before[-1], after[0]
                if f1 - f0 > 10:
                    mapped_xy = project_many(self.inv_homographies[f0])
                else:
                    xy0 = project_many(self.inv_homographies[f0])
                    xy1 = project_many(self.inv_homographies[f1])
                    weight = (f_idx - f0) / float(f1 - f0)
                    mapped_xy = xy0 + (xy1 - xy0) * weight

        out: list[tuple[int, int] | None] = []
        for i in range(len(mapped_xy)):
            x_raw, y_raw = float(mapped_xy[i][0]), float(mapped_xy[i][1])
            if x_raw == -9999.0:
                out.append(None)
                continue
            radar_x = int(round((x_raw + 52.5) * self.scale))
            radar_y = int(round((y_raw + 34.0) * self.scale))
            radar_x = max(0, min(self.radar_w, radar_x))
            radar_y = max(0, min(self.radar_h, radar_y))
            out.append((radar_x, radar_y))
        return out


