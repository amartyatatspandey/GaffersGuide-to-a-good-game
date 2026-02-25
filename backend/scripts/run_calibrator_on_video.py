"""
Run DynamicPitchCalibrator on a video file (e.g. match_test.mp4).

Samples frames, computes homography per frame, and prints/writes results.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Project layout: run from repo root or backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.dynamic_homography import DynamicPitchCalibrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VIDEO_PATH = BACKEND_ROOT / "data" / "match_test.mp4"
WEIGHTS_DIR = BACKEND_ROOT / "references" / "sn-calibration" / "resources"
OUTPUT_JSON = BACKEND_ROOT / "output" / "match_test_homographies.json"

# Sample every N frames (1 = every frame; 30 ≈ 1 per second at 30fps)
SAMPLE_EVERY = 30


def run(video_path: Path, weights_dir: Path, sample_every: int, output_json: Path | None) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")

    calibrator = DynamicPitchCalibrator(weights_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info("Video: %s | frames=%s fps=%.1f | sample_every=%d", video_path.name, total_frames, fps, sample_every)

    results: list[dict] = []
    frame_idx = 0
    success_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        H = calibrator.get_homography(frame)
        if H is not None:
            success_count += 1
            results.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 2),
                "homography": H.tolist(),
            })
            logger.info("Frame %d (t=%.1fs): homography OK", frame_idx, frame_idx / fps)
        else:
            logger.debug("Frame %d: no homography", frame_idx)

        frame_idx += 1

    cap.release()

    logger.info("Done. Success: %d / %d sampled frames", success_count, len(results) if results else 0)

    if output_json and results:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({"video": str(video_path), "sample_every": sample_every, "homographies": results}, f, indent=2)
        logger.info("Wrote %s", output_json)


if __name__ == "__main__":
    run(
        video_path=VIDEO_PATH,
        weights_dir=WEIGHTS_DIR,
        sample_every=SAMPLE_EVERY,
        output_json=OUTPUT_JSON,
    )
