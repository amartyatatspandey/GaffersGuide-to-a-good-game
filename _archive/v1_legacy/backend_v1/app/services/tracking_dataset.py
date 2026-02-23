"""
SoccerNet tracking dataset: video clips and tracklet labels.

Discovers clips under backend/data/soccernet/tracking-2023/<split>/ and
exposes clip paths + label paths for use with YOLO/supervision. Missing
labels are logged; the training loop can skip or use empty annotations.
"""
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TRACKING_TASK_DIR = "tracking-2023"

# Video extensions to discover clips.
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov")


def build_tracking_sample_list(root_dir: Path, split: str) -> list[tuple[Path, Path | None]]:
    """Discover tracking clips under root_dir/tracking-2023/<split>/.

    Returns list of (clip_dir, labels_path). clip_dir is the folder containing
    the video; labels_path is the JSON label file if found, else None.
    """
    base = root_dir / TRACKING_TASK_DIR / split
    if not base.is_dir():
        return []
    samples: list[tuple[Path, Path | None]] = []
    for clip_dir in base.iterdir():
        if not clip_dir.is_dir():
            continue
        # Prefer labels.json or <clip_name>_labels.json.
        labels_path = clip_dir / "labels.json"
        if not labels_path.is_file():
            labels_path = clip_dir / f"{clip_dir.name}_labels.json"
        if not labels_path.is_file():
            # Any JSON in the clip dir.
            jsons = list(clip_dir.glob("*.json"))
            labels_path = jsons[0] if jsons else None
        # Count if there is at least one video in the clip dir.
        has_video = any(
            f.suffix.lower() in VIDEO_EXTENSIONS for f in clip_dir.iterdir() if f.is_file()
        )
        if has_video:
            label_file = labels_path if (labels_path is not None and labels_path.is_file()) else None
            samples.append((clip_dir, label_file))
    return samples


def _parse_tracklet_labels(labels_path: Path) -> list[dict[str, Any]]:
    """Parse SoccerNet-style tracklet JSON into list of frame annotations."""
    try:
        with open(labels_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not parse %s: %s", labels_path, e)
        return []
    # Common keys: "annotations", "frames", "tracklets", or root is list of frames.
    if isinstance(data, list):
        return data
    for key in ("annotations", "frames", "tracklets"):
        if isinstance(data.get(key), list):
            return data[key]
    return []


class SoccerNetTrackingDataset(Dataset[tuple[Path, list[dict[str, Any]]] | tuple[None, None]]):
    """Dataset of tracking clips: (clip_dir, list of frame annotations)."""

    def __init__(self, root_dir: str | Path, split: str) -> None:
        """Build index of (clip_dir, labels_path) for the given split.

        Args:
            root_dir: Path to backend/data/soccernet.
            split: One of 'train', 'test'.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples = build_tracking_sample_list(self.root_dir, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[Path, list[dict[str, Any]]] | tuple[None, None]:
        """Return (clip_dir, frame_annotations) or (None, None) on error."""
        clip_dir, labels_path = self.samples[idx]
        if not clip_dir.is_dir():
            logger.warning("Missing clip dir: %s", clip_dir)
            return (None, None)
        if labels_path is None or not labels_path.is_file():
            logger.warning("No labels for clip: %s", clip_dir)
            return (clip_dir, [])  # Valid clip, empty annotations.
        annotations = _parse_tracklet_labels(labels_path)
        return (clip_dir, annotations)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(__file__).resolve().parent.parent.parent / "data" / "soccernet"
    dataset = SoccerNetTrackingDataset(root_dir=root, split="test")
    logger.info("Tracking dataset split=test len=%d", len(dataset))
    if len(dataset) > 0:
        clip_dir, anns = dataset[0]
        logger.info("First clip: %s, annotations count=%d", clip_dir, len(anns))
