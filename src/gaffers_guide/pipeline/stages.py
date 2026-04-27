"""Stage-level orchestration helpers."""

from __future__ import annotations

from pathlib import Path

from gaffers_guide.profiles import ProfileConfig
from gaffers_guide.runtime.run_e2e_cloud import run_e2e_cloud


def run_full_pipeline(*, video: Path, output_dir: Path, profile: ProfileConfig) -> Path:
    """Execute the current end-to-end runtime and return report path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_e2e_cloud(
        video=video,
        output_prefix=output_dir.name,
        profile=profile,
    )
