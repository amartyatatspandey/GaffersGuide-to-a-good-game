"""
Reference setup for transfer learning: clone Soccer_Analysis repo and extract .pt weights.

Ensures backend/references/external and backend/models/pretrained exist,
clones Adit Jain's Soccer_Analysis into external, and copies any .pt (YOLO) weights
to backend/models/pretrained. Uses logging; no hardcoded secrets.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Resolve backend root: script lives in backend/scripts/auxiliary_tools/
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent.parent
REFERENCES_EXTERNAL = BACKEND_ROOT / "references" / "external"
PRETRAINED_DIR = BACKEND_ROOT / "models" / "pretrained"
REPO_URL = "https://github.com/Adit-jain/Soccer_Analysis.git"
REPO_CLONE_PATH = REFERENCES_EXTERNAL / "Soccer_Analysis"


def ensure_directories() -> None:
    """Create references/external and models/pretrained if they do not exist."""
    REFERENCES_EXTERNAL.mkdir(parents=True, exist_ok=True)
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directories ready: %s, %s", REFERENCES_EXTERNAL, PRETRAINED_DIR)


def clone_or_pull_repo() -> None:
    """Clone Soccer_Analysis into references/external, or pull if already present."""
    if REPO_CLONE_PATH.exists() and (REPO_CLONE_PATH / ".git").exists():
        logger.info("Repo already exists at %s; pulling updates.", REPO_CLONE_PATH)
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=REPO_CLONE_PATH,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("git pull failed (non-fatal): %s", e.stderr or e)
    else:
        logger.info("Cloning %s into %s", REPO_URL, REPO_CLONE_PATH)
        subprocess.run(
            ["git", "clone", REPO_URL, str(REPO_CLONE_PATH)],
            check=True,
            cwd=str(REFERENCES_EXTERNAL),
        )


def find_pt_files(root: Path) -> list[Path]:
    """Return all .pt files under root (YOLO weight files)."""
    return list(root.rglob("*.pt"))


def copy_pt_to_pretrained() -> bool:
    """
    Scan cloned repo for .pt files and copy them to backend/models/pretrained.
    Returns True if at least one file was copied.
    """
    if not REPO_CLONE_PATH.exists():
        logger.warning("Clone path does not exist: %s", REPO_CLONE_PATH)
        return False
    pt_files = find_pt_files(REPO_CLONE_PATH)
    copied = False
    for src in pt_files:
        dest = PRETRAINED_DIR / src.name
        if dest.resolve() == src.resolve():
            continue
        try:
            shutil.copy2(src, dest)
            logger.info("Copied %s -> %s", src.name, dest)
            copied = True
        except OSError as e:
            logger.warning("Failed to copy %s: %s", src, e)
    if not pt_files:
        logger.info("No .pt files found in %s", REPO_CLONE_PATH)
    return copied


def main() -> None:
    """Run directory setup, clone/pull repo, and extract .pt weights."""
    ensure_directories()
    clone_or_pull_repo()
    found = copy_pt_to_pretrained()
    if not found:
        readme = REPO_CLONE_PATH / "README.md"
        print(
            "Repo cloned. Check backend/references/external/Soccer_Analysis/README.md "
            "for Google Drive links if the .pt files were not found automatically."
        )
        if readme.exists():
            logger.info("README path: %s", readme)


if __name__ == "__main__":
    main()
