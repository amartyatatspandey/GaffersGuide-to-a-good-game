"""
Team classifier: assign player crops to one of two teams via shirt-color clustering.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class TeamClassifier:
    def __init__(self) -> None:
        self._kmeans: KMeans | None = None

    def _top_half_mean_color(self, crop: np.ndarray) -> np.ndarray:
        h = crop.shape[0]
        end = max(1, h // 2)
        top = crop[0:end, :, :]
        return np.mean(top, axis=(0, 1)).astype(np.float64)

    def fit(self, crops: List[np.ndarray]) -> None:
        if not crops:
            return
        features = np.array([self._top_half_mean_color(c) for c in crops], dtype=np.float64)
        self._kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        self._kmeans.fit(features)

    def predict(self, crop: np.ndarray) -> int:
        if self._kmeans is None:
            raise RuntimeError("TeamClassifier.predict called before fit()")
        color = self._top_half_mean_color(crop).reshape(1, -1)
        return int(self._kmeans.predict(color)[0])
