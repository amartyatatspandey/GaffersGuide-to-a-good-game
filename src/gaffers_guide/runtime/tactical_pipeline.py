from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gaffers_guide.pipeline.config import PipelineConfig
from gaffers_guide.pipeline.e2e import MatchAnalysisPipeline
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
        pipeline = MatchAnalysisPipeline(profile=self.profile)
        return pipeline.run(
            PipelineConfig(
                video=video,
                output_dir=output,
                quality_profile=self.profile.name,
            )
        )
