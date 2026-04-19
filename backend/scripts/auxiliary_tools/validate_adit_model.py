"""
Validate Adit Jain's pre-trained model on match_test.mp4.

Runs inference to verify detection of Players (0), Ball (1), and Referees (2).
Visualizes bounding boxes with class labels and saves the full validation video.
Uses logging; prints detection counts for the first frame as required.
"""

import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Resolve backend root: script lives in backend/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.pipeline_paths import tracking_model_weights_path  # noqa: E402

MODEL_PATH = tracking_model_weights_path()
VIDEO_PATH = BACKEND_ROOT / "data" / "match_test.mp4"
OUTPUT_DIR = BACKEND_ROOT / "output"
OUTPUT_VIDEO = OUTPUT_DIR / "validation_adit.mp4"

# Class mapping: 0=Player, 1=Ball, 2=Referee
CLASS_NAMES = {0: "Player", 1: "Ball", 2: "Referee"}
CLASS_COLORS = {
    0: (0, 255, 0),  # Green for Players
    1: (255, 0, 0),  # Blue for Ball
    2: (255, 165, 0),  # Orange for Referees
}


def load_model(model_path: Path) -> YOLO:
    """
    Load the pre-trained YOLO model from the specified path.

    Args:
        model_path: Path to the .pt model file.

    Returns:
        Loaded YOLO model instance.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    logger.info("Loading model from %s", model_path)
    return YOLO(str(model_path))


def draw_detections(
    frame: cv2.typing.MatLike, results: list, frame_number: int
) -> tuple[cv2.typing.MatLike, Counter]:
    """
    Draw bounding boxes and labels on frame from YOLO results.

    Args:
        frame: OpenCV frame (numpy array).
        results: YOLO inference results (list of Results objects).
        frame_number: Current frame number for logging.

    Returns:
        Tuple of (annotated_frame, class_counter) where class_counter counts
        detections per class for this frame.
    """
    annotated_frame = frame.copy()
    class_counter = Counter()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract box coordinates and class
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            # Update counter
            class_counter[cls_id] += 1

            # Get class name and color
            class_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )

            # Draw label with confidence
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(int(y1), label_size[1] + 10)

            # Background rectangle for text readability
            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0], int(y1)),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (int(x1), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    return annotated_frame, class_counter


def process_video(
    model: YOLO, video_path: Path, output_path: Path, max_seconds: int | None = None
) -> None:
    """
    Process video, run inference, visualize detections, and save output video.

    Args:
        model: Loaded YOLO model.
        video_path: Path to input video file.
        output_path: Path to save output video.
        max_seconds: Maximum seconds to process (None = process entire video).
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at {video_path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = min(max_seconds * fps, total_frames) if max_seconds else total_frames
    logger.info(
        "Video: %dx%d @ %d fps, %d total frames. Processing %d frames.",
        width,
        height,
        fps,
        total_frames,
        max_frames,
    )

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_number = 0
    first_frame_detections: Counter | None = None

    try:
        while frame_number < max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.warning("End of video reached at frame %d", frame_number)
                break

            # Run inference
            results = model(frame, verbose=False)

            # Draw detections
            annotated_frame, class_counter = draw_detections(
                frame, results, frame_number
            )

            # Store first frame detections for console output
            if frame_number == 0:
                first_frame_detections = class_counter

            # Write frame
            out.write(annotated_frame)
            frame_number += 1

            if frame_number % 100 == 0:
                logger.info("Processed %d/%d frames", frame_number, max_frames)

    finally:
        cap.release()
        out.release()
        logger.info("Saved validation video to %s", output_path)

    # Print first frame detections (required console output)
    if first_frame_detections:
        players = first_frame_detections.get(0, 0)
        ball = first_frame_detections.get(1, 0)
        referees = first_frame_detections.get(2, 0)
        print(
            f"Frame 1: {players} Players, {ball} Ball, {referees} Referee"
        )


def main() -> None:
    """Load model, process video, and generate validation output."""
    try:
        model = load_model(MODEL_PATH)
        process_video(model, VIDEO_PATH, OUTPUT_VIDEO)
        logger.info("Validation complete. Check %s", OUTPUT_VIDEO)
    except FileNotFoundError as e:
        logger.error("%s", e)
        raise
    except Exception as e:
        logger.error("Validation failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
