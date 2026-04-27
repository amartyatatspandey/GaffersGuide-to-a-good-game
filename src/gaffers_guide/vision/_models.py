"""Lazy model loaders for heavy CV dependencies."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_ultralytics_yolo():
    """Lazily import and return ultralytics.YOLO class."""
    from ultralytics import YOLO

    return YOLO
