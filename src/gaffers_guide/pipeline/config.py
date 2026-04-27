"""Pipeline configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PipelineConfig:
    video: Path
    output_dir: Path
    quality_profile: Literal["fast", "balanced", "high_res", "sahi"] = "balanced"
