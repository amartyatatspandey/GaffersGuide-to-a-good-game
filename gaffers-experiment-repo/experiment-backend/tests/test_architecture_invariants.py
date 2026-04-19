from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_architecture_invariants import main as verify_main  # noqa: E402


def test_architecture_invariants() -> None:
    assert verify_main() == 0
