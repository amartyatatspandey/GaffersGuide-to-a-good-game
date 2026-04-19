"""Lightweight checks for Rule 1 P2 (generate_calibration import hygiene)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    gen = root / "scripts" / "auxiliary_tools" / "generate_calibration.py"
    if not gen.is_file():
        print("skip: generate_calibration.py not found")
        return 0
    text = gen.read_text(encoding="utf-8")
    head, _, rest = text.partition("def parse_args")
    if "sys.path.insert" in head:
        print("FAIL: sys.path.insert appears before parse_args (import-time path mutation risk)")
        return 1
    print("OK: generate_calibration defers sys.path.insert to runtime helpers / main()")

    e2e = (root / "scripts" / "pipeline_core" / "e2e_shared.py").read_text(encoding="utf-8")
    cloud = (root / "scripts" / "pipeline_core" / "run_e2e_cloud.py").read_text(encoding="utf-8")
    if "legacy.run_e2e_legacy" in e2e or "legacy.run_e2e_legacy" in cloud:
        print("FAIL: modern pipeline still references legacy.run_e2e_legacy")
        return 1
    print("OK: modern pipeline has no direct legacy bridge import")
    return 0


if __name__ == "__main__":
    sys.exit(main())
