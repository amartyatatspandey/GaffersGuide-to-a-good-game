from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gaffers_guide.profiles import ProfileConfig


@dataclass(slots=True)
class TacticalPipeline:
    """Thin orchestration wrapper for CLI-driven pipeline execution."""

    profile: ProfileConfig

    def run(
        self,
        *,
        video: Path,
        output: Path,
    ) -> Path:
        from gaffers_guide.runtime.run_e2e_cloud import run_e2e_cloud

        output.mkdir(parents=True, exist_ok=True)
        return run_e2e_cloud(
            video=video,
            output_prefix=output.name,
            profile=self.profile,
        )
