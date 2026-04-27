"""Detector protocols for plug-in CV backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray

from gaffers_guide.core.types import FrameDetections


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: NDArray) -> FrameDetections:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError
