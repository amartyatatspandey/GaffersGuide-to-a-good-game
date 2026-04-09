"""
Convert vision engine output to H.264 for QuickTime (macOS) compatibility.

Uses ffmpeg: libx264 + yuv420p + aac. Input: gaffer_v2_radar.mp4 → Output: gaffer_v2_radar_qt.mp4.
"""
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    backend_root = script_dir.parent
    input_path = backend_root / "output" / "gaffer_v2_radar.mp4"
    output_path = backend_root / "output" / "gaffer_v2_radar_qt.mp4"

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(output_path),
        "-y",
    ]
    logger.info("Conversion started...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg failed: %s", result.stderr[-500:] if result.stderr else "no stderr")
        return 1
    logger.info("Success: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
