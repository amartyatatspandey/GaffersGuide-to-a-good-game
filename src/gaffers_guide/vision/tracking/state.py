"""Tracking state contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackingState:
    next_id: int = 1
