"""SoccerNet pitch-segmentation weight file layout (shared by V1/V2 calibrators)."""

from __future__ import annotations

from pathlib import Path


def require_sn_calibration_weight_files(weights_dir: Path) -> tuple[Path, Path, Path]:
    """Return ``(pth, mean_npy, std_npy)`` paths or raise ``FileNotFoundError``."""
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    pth = weights_dir / "soccer_pitch_segmentation.pth"
    mean_npy = weights_dir / "mean.npy"
    std_npy = weights_dir / "std.npy"
    for f in (pth, mean_npy, std_npy):
        if not f.is_file():
            raise FileNotFoundError(
                f"Missing required file: {f}. "
                "Download weights from sn-calibration README and place under resources dir."
            )
    return pth, mean_npy, std_npy
