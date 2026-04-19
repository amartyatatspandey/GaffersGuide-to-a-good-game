from __future__ import annotations

import logging
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_GLOBS = ("main.py", "worker_main.py", "services/**/*.py")
FORBIDDEN_IMPORT_TOKENS = ("legacy", "references", "run_e2e_legacy")
LOGGER = logging.getLogger(__name__)


def _runtime_files() -> list[Path]:
    files: list[Path] = []
    for pattern in RUNTIME_GLOBS:
        files.extend(BACKEND_ROOT.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file() and p.suffix == ".py"})


def _has_forbidden_runtime_import(text: str) -> bool:
    normalized = text.replace('"', "'")
    for token in FORBIDDEN_IMPORT_TOKENS:
        if f"from {token}" in normalized or f"import {token}" in normalized:
            return True
        if f"/{token}/" in normalized:
            return True
    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    failures: list[str] = []
    for file_path in _runtime_files():
        text = file_path.read_text(encoding="utf-8")
        rel = file_path.relative_to(BACKEND_ROOT)
        if "sys.path.insert(" in text or "sys.path.append(" in text:
            failures.append(f"{rel}: runtime sys.path mutation is forbidden")
        if _has_forbidden_runtime_import(text):
            failures.append(f"{rel}: forbidden legacy/reference import token detected")
        if rel.parts[:1] == ("services",) and ("from scripts." in text or "import scripts." in text):
            failures.append(f"{rel}: services layer must not import scripts layer")
    if failures:
        for item in failures:
            LOGGER.error("FAIL: %s", item)
        return 1
    LOGGER.info("OK: experiment runtime has no sys.path mutations")
    LOGGER.info("OK: experiment runtime has no legacy/reference bridge imports")
    LOGGER.info("OK: services layer does not import scripts layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
