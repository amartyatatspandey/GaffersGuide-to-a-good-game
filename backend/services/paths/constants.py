"""Repository roots for path resolution."""

from __future__ import annotations

from pathlib import Path

# backend/services/paths/constants.py -> parents[2] == backend/
BACKEND_ROOT = Path(__file__).resolve().parents[2]
