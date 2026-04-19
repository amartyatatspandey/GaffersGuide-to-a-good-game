from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from scripts.auxiliary_tools import verify_architecture_invariants  # noqa: E402


def test_architecture_invariants_script() -> None:
    """CI guardrail for decoupling boundaries."""
    rc = verify_architecture_invariants.main()
    assert rc == 0

