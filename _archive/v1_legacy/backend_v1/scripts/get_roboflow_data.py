"""
Download football-field-detection dataset from Roboflow Cloud (YOLOv8 format).

Stores data under backend/data/roboflow_pitch.

Requires ROBOFLOW_API_KEY in env. If you get a workspace 404, set ROBOFLOW_WORKSPACE_ID
to your workspace slug from the app URL (app.roboflow.com/<workspace_id>/...).
Optional: ROBOFLOW_WORKSPACE_ID, ROBOFLOW_VERSION (default 1), ROBOFLOW_PROJECT_ID.
"""
import logging
import os
import shutil
from pathlib import Path

from roboflow import Roboflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set")

    workspace_id = os.getenv("ROBOFLOW_WORKSPACE_ID", "amartyas-workshop")
    project_id = os.getenv("ROBOFLOW_PROJECT_ID", "football-field-detection-f07vi-fjs4v")
    version_num = int(os.getenv("ROBOFLOW_VERSION", "1"))

    rf = Roboflow(api_key=api_key)
    try:
        dataset = (
            rf.workspace(workspace_id)
            .project(project_id)
            .version(version_num)
            .download("yolov8")
        )
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg or "does not exist" in err_msg or "missing permissions" in err_msg:
            raise RuntimeError(
                "Roboflow workspace not found or no permission. Set ROBOFLOW_WORKSPACE_ID to your workspace slug "
                "(from app.roboflow.com/<workspace_id>/...). For Universe projects, use the workspace shown on the dataset page."
            ) from e
        if "Version number" in err_msg and "is not found" in err_msg:
            raise RuntimeError(
                f"Roboflow version {version_num} not found for this project. Set ROBOFLOW_VERSION to a valid version number from the dataset page."
            ) from e
        raise

    # Library may create folder like "football-field-detection-f07vi-15" or "football-field-detection-15"; move into backend/data/roboflow_pitch
    script_dir = Path(__file__).resolve().parent
    backend_root = script_dir.parent
    dest_dir = backend_root / "data" / "roboflow_pitch"
    cwd = Path.cwd()
    for candidate in (
        hasattr(dataset, "location") and dataset.location and Path(dataset.location),
        cwd / f"{project_id}-{version_num}",
        backend_root / f"{project_id}-{version_num}",
        cwd / f"football-field-detection-{version_num}",
        backend_root / f"football-field-detection-{version_num}",
    ):
        if candidate and Path(candidate).is_dir():
            src_dir = Path(candidate)
            break
    else:
        raise FileNotFoundError(f"Expected download folder not found for {project_id} v{version_num}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        dest_item = dest_dir / item.name
        if dest_item.exists():
            if dest_item.is_dir():
                shutil.rmtree(dest_item)
            else:
                dest_item.unlink()
        shutil.move(str(item), str(dest_item))
    src_dir.rmdir()

    logger.info("✅ Data successfully pulled from Roboflow Cloud")


if __name__ == "__main__":
    main()
