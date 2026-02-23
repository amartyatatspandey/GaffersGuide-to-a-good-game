"""
Datasets API: list available SoccerNet splits and sample counts.

Uses calibration and tracking services to report dataset info without loading
full samples. Async for FastAPI compliance.
"""
import logging
from pathlib import Path

from fastapi import APIRouter

from backend.app.schemas.datasets import DatasetInfo, DatasetListResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _soccernet_root() -> Path:
    """Resolve backend/data/soccernet from this file."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "soccernet"


@router.get("", response_model=DatasetListResponse)
async def list_datasets() -> DatasetListResponse:
    """List available SoccerNet datasets (calibration, tracking) and sample counts per split."""
    root = _soccernet_root()
    datasets: list[DatasetInfo] = []

    # Calibration: report sample count per split via SoccerNetCalibrationDataset.
    try:
        from backend.app.services.calibration_dataset import (
            CALIBRATION_TASK_DIR,
            SoccerNetCalibrationDataset,
        )

        for split in ("train", "valid", "test"):
            ds = SoccerNetCalibrationDataset(root_dir=root, split=split)
            if len(ds) > 0:
                datasets.append(
                    DatasetInfo(
                        name=CALIBRATION_TASK_DIR,
                        split=split,
                        num_samples=len(ds),
                        root_dir=str(root),
                    )
                )
    except Exception as e:
        logger.warning("Calibration dataset discovery failed: %s", e)

    # Tracking: report per split using tracking dataset discovery.
    try:
        from backend.app.services.tracking_dataset import (
            TRACKING_TASK_DIR,
            build_tracking_sample_list,
        )

        for split in ("train", "test"):
            samples = build_tracking_sample_list(root, split)
            if samples:
                datasets.append(
                    DatasetInfo(
                        name=TRACKING_TASK_DIR,
                        split=split,
                        num_samples=len(samples),
                        root_dir=str(root),
                    )
                )
    except Exception as e:
        logger.warning("Tracking dataset discovery failed: %s", e)

    return DatasetListResponse(datasets=datasets)
