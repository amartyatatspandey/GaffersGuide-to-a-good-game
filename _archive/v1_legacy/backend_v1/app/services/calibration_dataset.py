"""
SoccerNet calibration dataset: images + 3x3 homography labels.

Loads images and corresponding JSON camera-parameter files from
backend/data/soccernet (calibration task). Homography is flattened to (9,)
for regression. Missing files are logged and return None so the training
loop can skip them without crashing.
"""
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Subfolder name for calibration task (matches SoccerNet downloadDataTask).
CALIBRATION_TASK_DIR = "calibration-2023"

# Common JSON keys for a 3x3 homography/camera matrix (row-major or list of rows).
# SoccerNet calibration may use camera_matrix, K, or homography depending on challenge.
HOMOGRAPHY_KEYS = (
    "homography",
    "Homography",
    "H",
    "homography_matrix",
    "camera_matrix",
    "K",
    "intrinsics",
)

# ImageNet normalization for pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_homography_3x3(data: dict[str, Any]) -> list[float] | None:
    """Extract a 3x3 homography from a dict (e.g. JSON). Flatten row-major to 9 values."""
    for key in HOMOGRAPHY_KEYS:
        if key not in data:
            continue
        raw = data[key]
        if isinstance(raw, list):
            if len(raw) == 9:
                return [float(x) for x in raw]
            if len(raw) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in raw):
                return [float(x) for row in raw for x in row]
        if hasattr(raw, "tolist"):
            arr = raw.tolist()
            if len(arr) == 3 and all(len(r) == 3 for r in arr):
                return [float(x) for row in arr for x in row]
    return None


def _build_sample_list(root_dir: Path, split: str) -> list[tuple[Path, Path]]:
    """Scan root_dir/calibration-2023/<split>/ for images and pair with same-stem JSON."""
    base = root_dir / CALIBRATION_TASK_DIR / split
    if not base.is_dir():
        return []
    samples: list[tuple[Path, Path]] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in base.rglob(ext):
            json_path = img_path.with_suffix(".json")
            if json_path.is_file():
                samples.append((img_path, json_path))
    return samples


class SoccerNetCalibrationDataset(Dataset[tuple[torch.Tensor, torch.Tensor] | tuple[None, None]]):
    """Dataset of calibration images and 3x3 homography (flattened to 9-D)."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        image_size: tuple[int, int] = (224, 224),
        normalize_mean: tuple[float, ...] = IMAGENET_MEAN,
        normalize_std: tuple[float, ...] = IMAGENET_STD,
    ) -> None:
        """Build index of (image_path, json_path) under root_dir for the given split.

        Args:
            root_dir: Path to backend/data/soccernet (contains calibration-2023/).
            split: One of 'train', 'test', 'valid'.
            image_size: (H, W) for Resize transform.
            normalize_mean: Channel mean for Normalize.
            normalize_std: Channel std for Normalize.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples = _build_sample_list(self.root_dir, split)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Load image and homography; return (image_tensor, homography_tensor) or (None, None) on error."""
        img_path, json_path = self.samples[idx]

        if not img_path.is_file():
            logger.warning("Missing image: %s", img_path)
            return (None, None)
        if not json_path.is_file():
            logger.warning("Missing label: %s", json_path)
            return (None, None)

        # Load image (BGR -> RGB for consistency with pretrained models).
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read image: %s", img_path)
            return (None, None)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load JSON and extract 3x3 homography.
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not parse %s: %s", json_path, e)
            return (None, None)

        flat_h = _find_homography_3x3(data)
        if flat_h is None:
            logger.warning("No 3x3 homography found in %s (keys: %s)", json_path, list(data.keys()))
            return (None, None)

        # Apply transforms: ToTensor expects (H, W, C) uint8.
        tensor = self.transform(image)
        homography = torch.tensor(flat_h, dtype=torch.float32)
        return (tensor, homography)


def _collate_skip_none(
    batch: list[tuple[torch.Tensor | None, torch.Tensor | None]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate only non-None samples so we can run one batch in __main__."""
    valid = [(img, h) for img, h in batch if img is not None and h is not None]
    if not valid:
        raise ValueError("No valid samples in batch (all missing or invalid).")
    imgs = torch.stack([x[0] for x in valid])
    homographies = torch.stack([x[1] for x in valid])
    return imgs, homographies


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Default root: backend/data/soccernet relative to this file.
    _root = Path(__file__).resolve().parent.parent.parent / "data" / "soccernet"
    dataset = SoccerNetCalibrationDataset(root_dir=_root, split="test")
    logger.info("Dataset split=test len=%d", len(dataset))

    if len(dataset) == 0:
        logger.warning("No samples found under %s. Run soccernet_loader and unzip calibration test split.", _root)
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_skip_none,
        )
        images, homographies = next(iter(loader))
        logger.info("One batch: images %s, homographies %s", images.shape, homographies.shape)
