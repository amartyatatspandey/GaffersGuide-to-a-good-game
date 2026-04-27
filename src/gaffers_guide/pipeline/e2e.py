"""High-level modular pipeline entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gaffers_guide.pipeline.config import PipelineConfig
from gaffers_guide.pipeline.stages import run_full_pipeline
from gaffers_guide.profiles import ProfileConfig, resolve_profile


@dataclass
class MatchAnalysisPipeline:
    """Public pipeline API for end-to-end match analysis."""

    profile: ProfileConfig

    @classmethod
    def from_profile(cls, profile_name: str) -> "MatchAnalysisPipeline":
        return cls(profile=resolve_profile(profile_name))

    def run(self, config: PipelineConfig) -> Path:
        return run_full_pipeline(
            video=config.video,
            output_dir=config.output_dir,
            profile=self.profile,
        )
