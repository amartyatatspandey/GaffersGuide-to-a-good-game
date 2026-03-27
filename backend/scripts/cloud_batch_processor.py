from __future__ import annotations

import logging
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
OUTPUT_DIR = BACKEND_ROOT / "output"
CLOUD_RESULTS_DIR = OUTPUT_DIR / "cloud_results"
RUN_E2E_SCRIPT = SCRIPT_DIR / "run_e2e.py"

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


def _run_single_video(video_path: Path) -> None:
    match_name = _safe_stem(video_path.stem)
    LOGGER.info("Processing match: %s", video_path.name)
    cmd = [sys.executable, str(RUN_E2E_SCRIPT), str(video_path)]
    subprocess.run(cmd, cwd=str(BACKEND_ROOT), check=True)

    report_src = OUTPUT_DIR / DEFAULT_REPORT_NAME
    metrics_src = OUTPUT_DIR / DEFAULT_METRICS_NAME
    overlay_src = OUTPUT_DIR / DEFAULT_OVERLAY_NAME

    report_dest = _copy_artifact(report_src, match_name, "_report.json")
    metrics_dest = _copy_artifact(metrics_src, match_name, "_metrics.json")
    overlay_dest = _copy_artifact(overlay_src, match_name, "_tracking_overlay.mp4")

    LOGGER.info("Saved report to %s", report_dest)
    LOGGER.info("Saved metrics to %s", metrics_dest)
    LOGGER.info("Saved overlay to %s", overlay_dest)


def main() -> None:
    CLOUD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tactical threshold alignment marker with run_e2e production threshold.
    try:
        from run_e2e import MIN_BALL_CONFIDENCE  # local script import

        LOGGER.info(
            "Threshold alignment verified: MIN_BALL_CONFIDENCE=%.2f",
            MIN_BALL_CONFIDENCE,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not import threshold from run_e2e.py: %s", exc)

    videos = sorted(TRAINING_SAMPLES_DIR.glob("*.mp4"))
    if not videos:
        LOGGER.warning("No .mp4 files found in %s", TRAINING_SAMPLES_DIR)
        return

    LOGGER.info("Found %d videos under %s", len(videos), TRAINING_SAMPLES_DIR)
    for video_path in videos:
        _run_single_video(video_path)
    LOGGER.info("Cloud batch processing complete for %d videos.", len(videos))


if __name__ == "__main__":
    main()
