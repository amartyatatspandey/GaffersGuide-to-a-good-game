"""
Debug script: visualize raw pitch landmark keypoints (goals, half-line, corners)
for homography/calibration stability. Helps diagnose disappearing players on radar.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIDENCE = 0.3
LOG_EVERY_N_FRAMES = 100
CIRCLE_RADIUS = 18
CIRCLE_COLOR = (0, 255, 255)  # BGR yellow
TEXT_COLOR = (255, 255, 255)
FAIL_COLOR = (0, 0, 255)  # BGR red


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    backend_root = script_dir.parent
    input_path = backend_root / "data" / "match_test.mp4"
    model_path = backend_root / "models" / "keypoint_model.pt"
    if not model_path.exists():
        model_path = backend_root / "models" / "pitch_pose_v1" / "weights" / "best.pt"
    output_path = backend_root / "output" / "debug_pitch_landmarks.mp4"

    if not model_path.exists():
        logger.error("Model not found. Tried keypoint_model.pt and pitch_pose_v1/weights/best.pt")
        return
    if not input_path.exists():
        logger.error("Input video not found: %s", input_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", input_path)
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None
    keypoint_counts: list[int] = []

    try:
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=CONFIDENCE, verbose=False)[0]
            n_keypoints = 0
            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                # One detection (pitch) with multiple keypoints
                xy = results.keypoints.xy[0].cpu().numpy()
                conf = (
                    results.keypoints.conf[0].cpu().numpy()
                    if results.keypoints.conf is not None
                    else np.ones(len(xy))
                )
                n_keypoints = len(xy)
                for i in range(n_keypoints):
                    x, y = int(xy[i, 0]), int(xy[i, 1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(frame, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, 2)
                        label = f"{i} {conf[i]:.2f}" if i < len(conf) else str(i)
                        cv2.putText(
                            frame, label, (x + CIRCLE_RADIUS, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA
                        )

            keypoint_counts.append(n_keypoints)
            cv2.putText(
                frame, f"Keypoints Found: {n_keypoints}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 2, cv2.LINE_AA
            )
            if n_keypoints < 4:
                cv2.putText(
                    frame, "HOMOGRAPHY FAIL", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, FAIL_COLOR, 2, cv2.LINE_AA
                )
            writer.write(frame)
            if (frame_idx + 1) % LOG_EVERY_N_FRAMES == 0:
                avg = sum(keypoint_counts[-LOG_EVERY_N_FRAMES:]) / LOG_EVERY_N_FRAMES
                logger.info("Frames %d–%d: avg keypoints = %.2f", frame_idx - LOG_EVERY_N_FRAMES + 1, frame_idx + 1, avg)
            frame_idx += 1
        logger.info("Wrote %d frames to %s", frame_idx, output_path)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info("Video writer released.")


if __name__ == "__main__":
    main()
