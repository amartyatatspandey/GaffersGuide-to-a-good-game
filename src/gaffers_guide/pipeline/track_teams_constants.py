"""Shared constants and small math helpers for team tracking (Rule 4 modularization)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Class IDs: 0=Player, 1=Ball, 2=Referee
CLASS_PLAYER = 0
CLASS_BALL = 1
CLASS_REF = 2

# Team box colors (BGR for OpenCV)
COLOR_TEAM_0 = (0, 0, 255)  # Red (Team A)
COLOR_TEAM_1 = (255, 0, 0)  # Blue (Team B)
COLOR_UNKNOWN = (128, 128, 128)  # Gray (waiting for buffer)
COLOR_GOALKEEPER = (0, 255, 255)  # Yellow for GK (team-associated)
COLOR_BALL = (255, 255, 0)  # Cyan
COLOR_REF = (0, 165, 255)  # Orange
TEAM_COLORS = {0: COLOR_TEAM_0, 1: COLOR_TEAM_1}

# Soft lock: min frames before assigning; max history for auto-correction
MIN_COLORS_BEFORE_PREDICT = 15
MIN_PLAYERS_TO_FIT = 10
# HSV green range (OpenCV H 0–180): pitch background
HSV_GREEN_H_LOW, HSV_GREEN_H_HIGH = 35, 85
HSV_GREEN_S_MIN = 40

# Hybrid healer: cosine ReID + tactical radar gate (scaled pitch pixels ~= meters * 10)
REID_COSINE_THRESHOLD = 0.85
RADAR_DISTANCE_HEAL_PX = 150.0  # ~15 m on radar at scale=10


def cosine_similarity(v1: np.ndarray | Sequence[float], v2: np.ndarray | Sequence[float]) -> float:
    """Cosine similarity between two vectors (e.g. 512-D ReID embeddings)."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    norm1 = float(np.linalg.norm(a))
    norm2 = float(np.linalg.norm(b))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm1 * norm2))
