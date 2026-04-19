"""
Run pitch calibrator on a video file (e.g. match_test.mp4).

By default uses ``AdvancedPitchCalibrator`` (V2: LM refinement, 1280×720 H lock).
Pass ``use_advanced_calibration=False`` to use legacy ``DynamicPitchCalibrator``.

Samples frames, computes homography per frame, and prints/writes results.
CLI defaults match legacy paths; use --video / --output for per-match files
that align with cloud_batch_processor.py ({stem}_homographies.json).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

# Project layout: run from repo root or backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VIDEO_PATH = BACKEND_ROOT / "data" / "match_test.mp4"
def _default_weights_dir() -> Path:
    from services.pipeline_paths import sn_calibration_resources_dir

    return sn_calibration_resources_dir()


DEFAULT_WEIGHTS_DIR = _default_weights_dir()
DEFAULT_SAMPLE_EVERY = 30


def run(
    video_path: Path,
    weights_dir: Path,
    sample_every: int,
    output_json: Path | None,
    *,
    use_advanced_calibration: bool = True,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")

    if use_advanced_calibration:
        from scripts.pipeline_core.advanced_pitch_calibration import AdvancedPitchCalibrator

        calibrator = AdvancedPitchCalibrator(weights_dir)
        logger.info("Using AdvancedPitchCalibrator (V2, pitch→1280×720)")
    else:
        from scripts.pipeline_core.dynamic_homography import DynamicPitchCalibrator

        calibrator = DynamicPitchCalibrator(weights_dir)
        logger.info("Using DynamicPitchCalibrator (V1 legacy, pitch→native resolution)")
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
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"video": str(video_path), "sample_every": sample_every, "homographies": results}, f, indent=2)
        logger.info("Wrote %s", output_json)
    elif output_json and not results:
        logger.warning("No homography samples; not writing %s (empty file would be skipped by cloud batch)", output_json)


def ensure_homography_json_for_video(
    video_path: Path,
    *,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    use_advanced_calibration: bool = True,
) -> Path:
    """
    Resolve homography JSON for ``video_path`` (env override or per-stem under ``output/``).

    If the file is missing and vendored weights exist under ``backend/models/calibration/``,
    runs the calibrator synchronously. Otherwise raises ``FileNotFoundError`` with a CLI hint.

    :param use_advanced_calibration: If True (default), generate matrices with
        ``AdvancedPitchCalibrator`` (1280×720 pitch→image). If False, use V1 native-resolution H.
    """
    from services.homography_resolution import ensure_homography_json_for_video as _ensure

    return _ensure(
        video_path,
        sample_every=sample_every,
        use_advanced_calibration=use_advanced_calibration,
    )


def _default_output_for_video(video_path: Path) -> Path:
    return BACKEND_ROOT / "output" / f"{video_path.stem}_homographies.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TacticalRadar homography JSON from broadcast video.")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=f"Input video (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help=f"SoccerNet calibration resources dir (default: {DEFAULT_WEIGHTS_DIR})",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
        help="Sample every N frames (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: backend/output/{video_stem}_homographies.json",
    )
    parser.add_argument(
        "--legacy-v1-calibrator",
        action="store_true",
        help="Use DynamicPitchCalibrator (native-res H) instead of AdvancedPitchCalibrator (1280×720).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    video_path = args.video.expanduser().resolve()
    weights_dir = args.weights_dir.expanduser().resolve()
    out = args.output.expanduser().resolve() if args.output else _default_output_for_video(video_path)
    run(
        video_path=video_path,
        weights_dir=weights_dir,
        sample_every=args.sample_every,
        output_json=out,
        use_advanced_calibration=not args.legacy_v1_calibrator,
    )
