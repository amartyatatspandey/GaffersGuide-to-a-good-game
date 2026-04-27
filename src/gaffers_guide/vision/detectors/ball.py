"""Ball detector wrapper with lazy model imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from gaffers_guide.core.types import BBoxDetection, FrameDetections
from gaffers_guide.vision._models import get_ultralytics_yolo
from gaffers_guide.vision.detectors.base import BaseDetector


@dataclass
class BallDetector(BaseDetector):
    model_path: str
    device: str = "cpu"
    confidence_threshold: float = 0.3
    _model: Optional[Any] = None

    def warmup(self) -> None:
        if self._model is None:
            yolo_cls = get_ultralytics_yolo()
            self._model = yolo_cls(self.model_path)

    def detect(self, frame: NDArray[np.uint8]) -> FrameDetections:
        self.warmup()
        if self._model is None:
            raise RuntimeError("Detector model failed to initialize")
        results = self._model(frame, verbose=False, conf=self.confidence_threshold)
        detections: list[BBoxDetection] = []
        if results:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                names = getattr(self._model, "names", {})
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].tolist()
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    detections.append(
                        BBoxDetection(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=str(names.get(cls_id, "unknown")),
                        )
                    )
        return FrameDetections(
            frame_idx=0,
            timestamp=0.0,
            detections=detections,
            frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
        )
