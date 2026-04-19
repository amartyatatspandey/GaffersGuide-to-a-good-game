"""Generate TacticalRadar homographies from video using Spiideo soccersegcal.

AUXILIARY / NON-PRODUCTION ONLY: this module is **not** on the ``run_e2e_cloud`` /
``LocalCVRunner`` dependency chain. Spiideo stack is loaded **only** from ``main()`` so
importing this file does not mutate ``sys.path`` (Rule 1 P2).

Preferred checkout: ``backend/calibration/soccersegcal_vendor`` (see README there), or set
``SPIIDEO_SEGCAL_ROOT``. Legacy fallback: ``backend/references/soccersegcal`` if the vendor
tree is absent.

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

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> Namespace:
    """Parse CLI arguments for calibration batch generation."""
    parser = ArgumentParser(
        description="Generate homography JSON from match video using Spiideo.",
        epilog=(
            "Standalone auxiliary tool only — not wired into production E2E. "
            "For production homographies use scripts.pipeline_core.run_calibrator_on_video "
            "and vendored sn-calibration under backend/calibration/."
        ),
    )
    parser.add_argument(
        "--spiideo-root",
        type=Path,
        default=None,
        help="Root of Spiideo soccersegcal checkout (default: SPIIDEO_SEGCAL_ROOT or calibration/soccersegcal_vendor).",
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
    default_ckpt = BACKEND_ROOT / "models" / "calibration" / "spiideo" / "snapshot.ckpt"
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.getenv("SPIIDEO_CHECKPOINT", str(default_ckpt))),
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


def _resolve_spiideo_root(cli_root: Path | None) -> Path:
    env = os.getenv("SPIIDEO_SEGCAL_ROOT", "").strip()
    if cli_root is not None:
        return cli_root.expanduser().resolve()
    if env:
        return Path(env).expanduser().resolve()
    vendor = BACKEND_ROOT / "calibration" / "soccersegcal_vendor"
    if vendor.is_dir() and any(vendor.iterdir()):
        return vendor.resolve()
    legacy = BACKEND_ROOT / "references" / "soccersegcal"
    return legacy.resolve()


def _activate_spiideo_import_path(root: Path) -> tuple[Path, Path]:
    """Insert Spiideo repo roots on ``sys.path`` (CLI/runtime only)."""
    spiideo_path = root
    module_path = root / "soccersegcal"
    if not spiideo_path.is_dir():
        raise FileNotFoundError(
            f"Spiideo root not found at {spiideo_path}. "
            "Clone into backend/calibration/soccersegcal_vendor or set SPIIDEO_SEGCAL_ROOT."
        )
    if str(spiideo_path) not in sys.path:
        sys.path.insert(0, str(spiideo_path))
    if module_path.is_dir() and str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))
    return spiideo_path, module_path


def frame_to_model_tensor(frame_bgr: np.ndarray, width: int) -> torch.Tensor:
    """Convert OpenCV BGR frame into normalized model tensor (C,H,W)."""
    height = (width * 9) // 16
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(torch.float32).div(255.0)
    return resize(image, [height, width], antialias=True)


def main() -> None:
    """Run Spiideo camera estimation pass and export strict TacticalRadar JSON."""
    args = parse_args()
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Spiideo checkpoint not found at {args.checkpoint}. "
            "Place snapshot.ckpt under backend/models/calibration/spiideo/ or pass --checkpoint."
        )

    root = _resolve_spiideo_root(args.spiideo_root)
    _activate_spiideo_import_path(root)

    from sncalib.baseline_cameras import Camera
    from soccersegcal.pose import segs2cam
    from soccersegcal.train import LitSoccerFieldSegmentation

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
    segmentation_model = LitSoccerFieldSegmentation.load_from_checkpoint(str(args.checkpoint))
    segmentation_model = segmentation_model.to(memory_format=torch.channels_last, device=device)
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
                    segs = torch.sigmoid_(segmentation_model(image_tensor[None]))[0].cpu()

                ptz_model = segs2cam(segs, world_scale, prev_cam, show=False)
                if ptz_model is not None:
                    ptz_model = ptz_model.cpu()
                    smallest_side = min(segs.shape[2], segs.shape[1])
                    focal = smallest_side / 2.0 / float(ptz_model.camera_focal.item())

                    cam = Camera(int(segs.shape[2]), int(segs.shape[1]))
                    cam.from_json_parameters(
                        {
                            "position_meters": (ptz_model.camera_position.detach().numpy() * world_scale).tolist(),
                            "principal_point": cam.principal_point,
                            "x_focal_length": focal,
                            "y_focal_length": focal,
                            "pan_degrees": float(np.rad2deg(ptz_model.camera_pan.item())),
                            "tilt_degrees": float(np.rad2deg(ptz_model.camera_tilt.item())),
                            "roll_degrees": float(np.rad2deg(ptz_model.camera_roll.item())),
                            "radial_distortion": np.zeros(6, dtype=np.float64).tolist(),
                            "tangential_distortion": np.zeros(2, dtype=np.float64).tolist(),
                            "thin_prism_distortion": np.zeros(4, dtype=np.float64).tolist(),
                        }
                    )
                    matrix = _camera_to_homography(cam)
                    prev_cam = cam
            except Exception as exc:  # noqa: BLE001
                logger.warning("Calibration failed at frame %d: %s", frame_idx, exc)

            if matrix is None and prev_cam is not None:
                matrix = _camera_to_homography(prev_cam)
            if matrix is None:
                matrix = np.eye(3, dtype=np.float64)

            output_data["homographies"].append({"frame": frame_idx, "homography": matrix.tolist()})
            logger.info("Calibrated frame %d", frame_idx)

        frame_idx += 1

    cap.release()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    logger.info("Calibration complete! Saved to %s", args.output)


def _camera_to_homography(cam: Any) -> np.ndarray:
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
        proj = cam.project_point(np.array([wp[0], wp[1], 0.0], dtype=np.float64), distort=False)
        image_points[idx] = np.array([proj[0], proj[1]], dtype=np.float32)

    homography = cv2.getPerspectiveTransform(world_points[:, :2], image_points)
    return homography.astype(np.float64)


if __name__ == "__main__":
    main()
