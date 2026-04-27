"""Parsers for tracking and report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_tracking_json(path: Path) -> dict[str, Any]:
    """Parse a JSON tracking artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
