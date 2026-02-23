"""
Create folder structure for football computer vision backend.
Paths: backend/data, backend/scripts, backend/output
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIRS = ("data", "scripts", "output")


def create_backend_structure() -> None:
    """Create backend/data, backend/scripts, backend/output."""
    backend_root = BASE_DIR / "backend"
    for name in BACKEND_DIRS:
        path = backend_root / name
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Created: %s", path)


if __name__ == "__main__":
    create_backend_structure()
