"""Video I/O utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class VideoReader:
    """Simple OpenCV-based frame reader."""

    path: Path

    def __iter__(self) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
                idx += 1
        finally:
            cap.release()
