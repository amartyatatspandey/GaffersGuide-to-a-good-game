from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_ROOT.parent
TRAINING_SAMPLES_DIR = BACKEND_ROOT / "data" / "training_samples"
# Override for cloud: e.g. GAFFERS_VIDEO_INPUT_DIR=/workspace/videos
_VIDEO_INPUT_ENV = os.getenv("GAFFERS_VIDEO_INPUT_DIR", "").strip()
VIDEO_INPUT_DIR = (
    Path(_VIDEO_INPUT_ENV).expanduser().resolve()
    if _VIDEO_INPUT_ENV
    else TRAINING_SAMPLES_DIR
)
OUTPUT_DIR = BACKEND_ROOT / "output"
CLOUD_RESULTS_DIR = OUTPUT_DIR / "cloud_results"
RUN_E2E_SCRIPT = SCRIPT_DIR / "run_e2e_cloud.py"

_HOMOGRAPHY_DIR_ENV = os.getenv("GAFFERS_HOMOGRAPHY_DIR", "").strip()
HOMOGRAPHY_DIR = (
    Path(_HOMOGRAPHY_DIR_ENV).expanduser().resolve()
    if _HOMOGRAPHY_DIR_ENV
    else OUTPUT_DIR
)

DEFAULT_REPORT_NAME = "test_mp4_report.json"
DEFAULT_METRICS_NAME = "tactical_metrics_e2e.json"
DEFAULT_OVERLAY_NAME = "test_mp4_tracking_overlay.mp4"


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _unique_target_path(base_name: str, suffix: str) -> Path:
    candidate = CLOUD_RESULTS_DIR / f"{base_name}{suffix}"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = CLOUD_RESULTS_DIR / f"{base_name}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _copy_artifact(src: Path, match_name: str, suffix: str) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Expected artifact missing: {src}")
    dest = _unique_target_path(match_name, suffix)
    shutil.copy2(src, dest)
    return dest


def _homography_path_for_video(video_path: Path) -> Path:
    """Per-match file: {video_stem}_homographies.json under GAFFERS_HOMOGRAPHY_DIR or output/."""
    return HOMOGRAPHY_DIR / f"{video_path.stem}_homographies.json"


def _validate_homography_json(path: Path) -> tuple[bool, str]:
    """
    Return (ok, reason). Requires readable JSON with non-empty homographies list
    (TacticalRadar needs real matrices for meaningful radar).
    """
    if not path.is_file():
        return False, f"not a file: {path}"
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON: {exc}"
    if not isinstance(data, dict):
        return False, "root must be a JSON object"
    homographies = data.get("homographies")
    if not isinstance(homographies, list):
        return False, "missing or invalid 'homographies' array"
    if len(homographies) == 0:
        return False, "'homographies' is empty (no calibration samples)"
    return True, ""


def _log_skip_homography(video_path: Path, homography_path: Path, reason: str) -> None:
    border = "=" * 72
    LOGGER.error(border)
    LOGGER.error(
        "MISSING OR INVALID HOMOGRAPHY — SKIPPING THIS MATCH (no wrong-map fallback)"
    )
    LOGGER.error("Video:              %s", video_path)
    LOGGER.error("Expected JSON path: %s", homography_path.resolve())
    LOGGER.error("Reason:             %s", reason)
    LOGGER.error(
        "Generate with: scripts/run_calibrator_on_video.py --video <path> "
        "(writes {stem}_homographies.json under output/)"
    )
    LOGGER.error(border)


def _run_single_video(video_path: Path) -> bool:
    """
    Run E2E for one video with GAFFERS_HOMOGRAPHY_JSON set to that match's calibration file.

    Returns:
        True if processed successfully, False if skipped (missing/invalid homography).
    """
    match_name = _safe_stem(video_path.stem)
    homography_path = _homography_path_for_video(video_path)

    ok, reason = _validate_homography_json(homography_path)
    if not ok:
        _log_skip_homography(video_path, homography_path, reason)
        return False

    LOGGER.info(
        "Processing match: %s (homography: %s)", video_path.name, homography_path
    )

    env = os.environ.copy()
    env["GAFFERS_HOMOGRAPHY_JSON"] = str(homography_path.resolve())

    cmd = [sys.executable, str(RUN_E2E_SCRIPT), str(video_path)]
    subprocess.run(cmd, cwd=str(BACKEND_ROOT), env=env, check=True)

    report_src = OUTPUT_DIR / DEFAULT_REPORT_NAME
    metrics_src = OUTPUT_DIR / DEFAULT_METRICS_NAME
    overlay_src = OUTPUT_DIR / DEFAULT_OVERLAY_NAME

    report_dest = _copy_artifact(report_src, match_name, "_report.json")
    metrics_dest = _copy_artifact(metrics_src, match_name, "_metrics.json")
    overlay_dest = _copy_artifact(overlay_src, match_name, "_tracking_overlay.mp4")

    LOGGER.info("Saved report to %s", report_dest)
    LOGGER.info("Saved metrics to %s", metrics_dest)
    LOGGER.info("Saved overlay to %s", overlay_dest)
    return True


def main() -> None:
    CLOUD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HOMOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)

    # Tactical threshold alignment marker with production threshold.
    try:
        from scripts.e2e_shared import MIN_BALL_CONFIDENCE

        LOGGER.info(
            "Threshold alignment verified: MIN_BALL_CONFIDENCE=%.2f",
            MIN_BALL_CONFIDENCE,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not import threshold from run_e2e.py: %s", exc)

    videos = sorted(VIDEO_INPUT_DIR.glob("*.mp4"))
    if not videos:
        LOGGER.warning("No .mp4 files found in %s", VIDEO_INPUT_DIR)
        return

    LOGGER.info(
        "Found %d videos under %s; homography dir: %s",
        len(videos),
        VIDEO_INPUT_DIR,
        HOMOGRAPHY_DIR,
    )
    processed = 0
    skipped = 0
    for video_path in videos:
        if _run_single_video(video_path):
            processed += 1
        else:
            skipped += 1

    LOGGER.info(
        "Cloud batch finished: processed=%d skipped=%d total=%d",
        processed,
        skipped,
        len(videos),
    )


if __name__ == "__main__":
    main()
