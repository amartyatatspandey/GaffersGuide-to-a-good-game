"""Pydantic models for dataset-related API responses."""

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Metadata for a single dataset (e.g. calibration test split)."""

    name: str = Field(..., description="Dataset identifier (e.g. calibration-2023)")
    split: str = Field(..., description="Split: train, valid, test, or challenge")
    num_samples: int = Field(..., ge=0, description="Number of indexed samples")
    root_dir: str = Field(..., description="Path to dataset root")


class DatasetListResponse(BaseModel):
    """List of dataset infos returned by the datasets router."""

    datasets: list[DatasetInfo] = Field(default_factory=list, description="Available datasets")
