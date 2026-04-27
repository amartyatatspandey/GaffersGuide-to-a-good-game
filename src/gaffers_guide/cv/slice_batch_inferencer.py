from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
from ultralytics import YOLO


@dataclass(frozen=True)
class SliceBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class SliceCandidate:
    xyxy: np.ndarray
    confidence: float
    source: str


class SliceBatchInferencer:
    """Run one batched forward pass over generated slices."""

    def __init__(
        self,
        model: YOLO,
        *,
        ball_class_ids: set[int],
        conf: float,
        device: str | None,
        use_half: bool,
    ) -> None:
        self._model = model
        self._ball_class_ids = ball_class_ids
        self._conf = float(conf)
        self._device = device
        self._use_half = bool(use_half)

    def infer(
        self, frame_bgr: np.ndarray, slices: list[SliceBox]
    ) -> list[SliceCandidate]:
        if not slices:
            return []
        crops: list[np.ndarray] = []
        valid_slices: list[SliceBox] = []
        for s in slices:
            if s.x2 <= s.x1 or s.y2 <= s.y1:
                continue
            crop = frame_bgr[s.y1 : s.y2, s.x1 : s.x2]
            if crop.size <= 0:
                continue
            crops.append(crop)
            valid_slices.append(s)
        if not crops:
            return []

        kwargs: dict[str, Any] = {"conf": self._conf, "verbose": False}
        if self._device:
            kwargs["device"] = self._device
        if self._use_half:
            kwargs["half"] = True

        results = self._model(crops, **kwargs)
        candidates: list[SliceCandidate] = []
        for idx, result in enumerate(results):
            if idx >= len(valid_slices):
                break
            s = valid_slices[idx]
            detections = sv.Detections.from_ultralytics(result)
            det_conf = getattr(detections, "confidence", None)
            for i in range(len(detections)):
                cid = int(detections.class_id[i])
                if cid not in self._ball_class_ids:
                    continue
                score = (
                    float(det_conf[i])
                    if det_conf is not None and i < len(det_conf)
                    else 0.0
                )
                local = detections.xyxy[i]
                global_xyxy = np.array(
                    [
                        float(local[0]) + float(s.x1),
                        float(local[1]) + float(s.y1),
                        float(local[2]) + float(s.x1),
                        float(local[3]) + float(s.y1),
                    ],
                    dtype=np.float32,
                )
                candidates.append(
                    SliceCandidate(
                        xyxy=global_xyxy,
                        confidence=score,
                        source="sahi_slice",
                    )
                )
        return candidates

