"""TVCalib-based keypoint extraction for camera calibration.

Outputs `calibration_data.json` for your existing `calibrate.cpp`.

JSON schema (minimal):
{
  "frames": [
    {
      "frame_idx": <int>,
      "keypoints": {
        "<semantic intersection key>": [x, y]
      }
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _setup_import_paths() -> None:
    """Ensure TVCalib + its submodules are importable."""
    tvcalib_root = Path(__file__).resolve().parent / "tvcalib"
    if str(tvcalib_root) not in sys.path:
        sys.path.insert(0, str(tvcalib_root))


def _load_segmentation_model(checkpoint_path: Path, device: str) -> Any:
    """Load TVCalib segmentation model."""
    from tvcalib.inference import InferenceSegmentationModel

    return InferenceSegmentationModel(checkpoint=checkpoint_path, device=device)


def _extract_keypoints_from_frame(
    frame_bgr: np.ndarray,
    *,
    model_seg: Any,
    device: str,
    image_width: int,
    image_height: int,
    seg_width: int,
    seg_height: int,
    radius: int,
    maxdist: int,
    num_points_lines: int,
    num_points_circles: int,
) -> dict[str, list[float]]:
    """Run TVCalib segmentation + point selection, then return semantic intersection keypoints."""
    from SoccerNet.Evaluation.utils_calibration import SoccerPitch
    from sn_segmentation.src.custom_extremities import (
        generate_class_synthesis,
        get_line_extremities,
    )

    # Match TVCalib's expected source image resolution (used later for scaling keypoints).
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    # Mirror TVCalib's InferenceDatasetSegmentation transforms: Resize(shorter_edge=256), normalize.
    tfms = T.Compose(
        [
            T.Resize(256),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    from PIL import Image

    pil = Image.fromarray(frame_resized)
    img_t = tfms(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        sem_lines = model_seg.inference(img_t)  # [1, seg_height, seg_width]

    sem_lines_np = sem_lines[0].detach().cpu().numpy().astype(np.uint8)

    # 1) semantic -> per-class skeleton points
    buckets = generate_class_synthesis(sem_lines_np, radius=radius)
    # 2) skeleton points -> extremities / tangent points (normalized to [0,1])
    extremities = get_line_extremities(
        buckets,
        maxdist=maxdist,
        width=seg_width,
        height=seg_height,
        num_points_lines=num_points_lines,
        num_points_circles=num_points_circles,
    )

    def _endpoints_for_line(class_name: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return (p1, p2) pixel endpoints defining the detected line; None if missing/unusable."""
        pts = extremities.get(class_name)
        if pts is None or len(pts) < 2:
            return None
        p1 = pts[0]
        p2 = pts[-1]
        x1 = float(p1["x"]) * float(image_width - 1)
        y1 = float(p1["y"]) * float(image_height - 1)
        x2 = float(p2["x"]) * float(image_width - 1)
        y2 = float(p2["y"]) * float(image_height - 1)
        # Avoid degenerate/near-identical endpoints.
        if (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) < 25.0:  # (5px)^2
            return None
        return (x1, y1), (x2, y2)

    def _line_from_points(p1: tuple[float, float], p2: tuple[float, float]) -> np.ndarray:
        # Homogeneous line l = p1 x p2, where p=(x,y,1).
        x1, y1 = p1
        x2, y2 = p2
        v1 = np.array([x1, y1, 1.0], dtype=np.float64)
        v2 = np.array([x2, y2, 1.0], dtype=np.float64)
        return np.cross(v1, v2)  # shape (3,)

    def _intersection_of_lines(l1: np.ndarray, l2: np.ndarray) -> list[float] | None:
        # Intersection point p = l1 x l2 in homogeneous coords.
        p = np.cross(l1, l2)
        if abs(float(p[2])) < 1e-9:
            return None
        x = float(p[0] / p[2])
        y = float(p[1] / p[2])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return [x, y]

    def _intersection_by_line_names(
        top_key: str,
        left_class_name: str,
        right_class_name: str,
        *,
        image_line_override: dict[str, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> list[float] | None:
        # Unused; keeps helper signature flexible for future.
        _ = top_key
        _ = image_line_override
        e1 = _endpoints_for_line(left_class_name)
        e2 = _endpoints_for_line(right_class_name)
        if e1 is None or e2 is None:
            return None
        l1 = _line_from_points(e1[0], e1[1])
        l2 = _line_from_points(e2[0], e2[1])
        return _intersection_of_lines(l1, l2)

    # Stable intersection keypoints derived from line crossings in the image plane.
    # We use SoccerNet's `SoccerPitch.line_extremities_keys` to discover which
    # semantic intersection points are shared between which two semantic lines.
    #
    # NOTE: We intentionally avoid exporting raw extremity points for "moving target"
    # stability; we export only the intersections between line equations.
    pitch = SoccerPitch()
    point_to_lines: dict[str, set[str]] = {}
    for line_name, (point_a, point_b) in pitch.line_extremities_keys.items():
        point_to_lines.setdefault(point_a, set()).add(line_name)
        point_to_lines.setdefault(point_b, set()).add(line_name)

    out: dict[str, list[float]] = {}
    for point_key, lines in point_to_lines.items():
        if len(lines) < 2:
            continue
        # SoccerPitch's shared points have exactly 2 lines in this dataset.
        line_list = sorted(lines)
        line_a = line_list[0]
        line_b = line_list[1]

        e1 = _endpoints_for_line(line_a)
        e2 = _endpoints_for_line(line_b)
        if e1 is None or e2 is None:
            continue

        l1 = _line_from_points(e1[0], e1[1])
        l2 = _line_from_points(e2[0], e2[1])
        pt = _intersection_of_lines(l1, l2)
        if pt is None:
            continue

        out[point_key] = pt

    # Additional semantic intersections that aren't guaranteed to be shared
    # endpoints in `SoccerPitch.line_extremities_keys`, but are well-defined
    # intersections of detected semantic lines.
    extra_defs: dict[str, tuple[str, str]] = {
        "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION": ("Side line top", "Middle line"),
        "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION": ("Side line bottom", "Middle line"),
    }
    for semantic_key, (line_a, line_b) in extra_defs.items():
        e1 = _endpoints_for_line(line_a)
        e2 = _endpoints_for_line(line_b)
        if e1 is None or e2 is None:
            continue
        l1 = _line_from_points(e1[0], e1[1])
        l2 = _line_from_points(e2[0], e2[1])
        pt = _intersection_of_lines(l1, l2)
        if pt is None:
            continue
        out[semantic_key] = pt

    return out


def main() -> None:
    """Run keypoint extraction on `match_video.mp4` every 10th frame."""
    _setup_import_paths()

    refs_dir = Path(__file__).resolve().parent
    video_path = refs_dir / "match_video.mp4"
    json_out_path = refs_dir / "calibration_data.json"

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}. Expected a placeholder at backend/references/match_video.mp4."
        )

    checkpoint_path = refs_dir / "tvcalib" / "data" / "segment_localization" / "train_59.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"TVCalib checkpoint not found: {checkpoint_path}. Download it into tvcalib/data/segment_localization."
        )

    # TVCalib notebook uses these source resolutions.
    image_width = 1280
    image_height = 720
    seg_width = 455
    seg_height = 256

    sample_every = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available; TVCalib segmentation will be slow on CPU.")

    model_seg = _load_segmentation_model(checkpoint_path, device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: list[dict[str, Any]] = []
    frame_idx = 0
    radius = 4
    maxdist = 30
    # We only need two endpoints per line to define a stable line equation.
    num_points_lines = 2
    num_points_circles = 2

    logger.info("Starting TVCalib keypoint extraction (every %d frames)...", sample_every)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            keypoints_xy = _extract_keypoints_from_frame(
                frame,
                model_seg=model_seg,
                device=device,
                image_width=image_width,
                image_height=image_height,
                seg_width=seg_width,
                seg_height=seg_height,
                radius=radius,
                maxdist=maxdist,
                num_points_lines=num_points_lines,
                num_points_circles=num_points_circles,
            )

            frames.append({"frame_idx": int(frame_idx), "keypoints": keypoints_xy})
            logger.info("Extracted keypoints for frame %d", frame_idx)

        frame_idx += 1

    cap.release()

    payload: dict[str, Any] = {"frames": frames}
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote %d frames to %s", len(frames), json_out_path)


if __name__ == "__main__":
    main()

