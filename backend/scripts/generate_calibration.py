"""Generate TacticalRadar homographies from video using Spiideo soccersegcal.

CLI defaults point at the repo dev clip and matching output filename; pass --video
and --output for real matches (same convention as ``output/{{stem}}_homographies.json``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Decoupled import path for Spiideo's architecture.
# Assumes the repo is cloned at backend/references/soccersegcal.
SPIIDEO_PATH = Path(__file__).resolve().parent.parent / "references" / "soccersegcal"
if str(SPIIDEO_PATH) not in sys.path:
    sys.path.insert(0, str(SPIIDEO_PATH))
SPIIDEO_MODULE_PATH = SPIIDEO_PATH / "soccersegcal"
if str(SPIIDEO_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SPIIDEO_MODULE_PATH))

from sncalib.baseline_cameras import Camera
from soccersegcal.pose import segs2cam
from soccersegcal.train import LitSoccerFieldSegmentation


def parse_args() -> Namespace:
    """Parse CLI arguments for calibration batch generation."""
    parser = ArgumentParser(
        description="Generate homography JSON from match video using Spiideo."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("backend/data/match_test.mp4"),
        help="Input video path (default is dev sample; override per match).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/output/match_test_homographies.json"),
        help="Output JSON (default matches dev sample stem; use output/{stem}_homographies.json for other videos).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.getenv(
                "SPIIDEO_CHECKPOINT", "backend/references/soccersegcal/snapshot.ckpt"
            )
        ),
        help="Path to Spiideo segmentation checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Sample every Nth frame for calibration.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Inference width for Spiideo model (height uses 16:9).",
    )
    return parser.parse_args()


def frame_to_model_tensor(frame_bgr: np.ndarray, width: int) -> torch.Tensor:
    """Convert OpenCV BGR frame into normalized model tensor (C,H,W)."""
    height = (width * 9) // 16
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(torch.float32).div(255.0)
    return resize(image, [height, width], antialias=True)


def camera_to_homography(cam: Camera) -> np.ndarray:
    """Build world->image homography from Spiideo camera parameters."""
    world_points = np.array(
        [
            [-52.5, -34.0, 1.0],
            [52.5, -34.0, 1.0],
            [52.5, 34.0, 1.0],
            [-52.5, 34.0, 1.0],
        ],
        dtype=np.float32,
    )
    image_points = np.zeros((4, 2), dtype=np.float32)

    for idx, wp in enumerate(world_points):
        proj = cam.project_point(
            np.array([wp[0], wp[1], 0.0], dtype=np.float64), distort=False
        )
        image_points[idx] = np.array([proj[0], proj[1]], dtype=np.float32)

    homography = cv2.getPerspectiveTransform(world_points[:, :2], image_points)
    return homography.astype(np.float64)


def main() -> None:
    """Run Spiideo camera estimation pass and export strict TacticalRadar JSON."""
    args = parse_args()
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Spiideo checkpoint not found at {args.checkpoint}. "
            "Download snapshot.ckpt from Spiideo releases or pass --checkpoint."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        logger.error("Failed to open video: %s", args.video)
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Spiideo segs2cam currently requires CUDA in upstream code. "
            "No CUDA device is available in this environment."
        )
    device = torch.device("cuda")
    logger.info("Loading Spiideo checkpoint from %s on %s", args.checkpoint, device)
    segmentation_model = LitSoccerFieldSegmentation.load_from_checkpoint(
        str(args.checkpoint)
    )
    segmentation_model = segmentation_model.to(
        memory_format=torch.channels_last, device=device
    )
    segmentation_model.eval()

    output_data: dict[str, list[dict[str, Any]]] = {"homographies": []}
    frame_idx = 0
    prev_cam: Camera | None = None
    world_scale = 100

    logger.info("Starting Spiideo SOTA Calibration Pass...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.sample_every == 0:
            matrix = None
            try:
                image_tensor = frame_to_model_tensor(frame, width=args.width).to(device)
                with torch.no_grad():
                    segs = torch.sigmoid_(segmentation_model(image_tensor[None]))[
                        0
                    ].cpu()

                ptz_model = segs2cam(segs, world_scale, prev_cam, show=False)
                if ptz_model is not None:
                    ptz_model = ptz_model.cpu()
                    smallest_side = min(segs.shape[2], segs.shape[1])
                    focal = smallest_side / 2.0 / float(ptz_model.camera_focal.item())

                    cam = Camera(int(segs.shape[2]), int(segs.shape[1]))
                    cam.from_json_parameters(
                        {
                            "position_meters": (
                                ptz_model.camera_position.detach().numpy() * world_scale
                            ).tolist(),
                            "principal_point": cam.principal_point,
                            "x_focal_length": focal,
                            "y_focal_length": focal,
                            "pan_degrees": float(
                                np.rad2deg(ptz_model.camera_pan.item())
                            ),
                            "tilt_degrees": float(
                                np.rad2deg(ptz_model.camera_tilt.item())
                            ),
                            "roll_degrees": float(
                                np.rad2deg(ptz_model.camera_roll.item())
                            ),
                            "radial_distortion": np.zeros(6, dtype=np.float64).tolist(),
                            "tangential_distortion": np.zeros(
                                2, dtype=np.float64
                            ).tolist(),
                            "thin_prism_distortion": np.zeros(
                                4, dtype=np.float64
                            ).tolist(),
                        }
                    )
                    matrix = camera_to_homography(cam)
                    prev_cam = cam
            except Exception as exc:  # noqa: BLE001
                logger.warning("Calibration failed at frame %d: %s", frame_idx, exc)

            if matrix is None and prev_cam is not None:
                matrix = camera_to_homography(prev_cam)
            if matrix is None:
                matrix = np.eye(3, dtype=np.float64)

            output_data["homographies"].append(
                {"frame": frame_idx, "homography": matrix.tolist()}
            )
            logger.info("Calibrated frame %d", frame_idx)

        frame_idx += 1

    cap.release()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    logger.info("Calibration complete! Saved to %s", args.output)


if __name__ == "__main__":
    main()
