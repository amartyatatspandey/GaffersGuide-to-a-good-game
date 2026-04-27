from __future__ import annotations

import importlib

import pytest


MODULES = [
    "gaffers_guide.cli",
    "gaffers_guide.profiles",
    "gaffers_guide.runtime.tactical_pipeline",
    "gaffers_guide.runtime.run_e2e",
    "gaffers_guide.runtime.run_calibrator_on_video",
    "gaffers_guide.runtime.run_e2e_cloud",
    "gaffers_guide.pipeline.e2e_shared",
    "gaffers_guide.pipeline.e2e_shared_impl",
    "gaffers_guide.pipeline.track_teams",
    "gaffers_guide.pipeline.dynamic_homography",
    "gaffers_guide.pipeline.generate_analytics",
    "gaffers_guide.cv.optimized_sahi_wrapper",
    "gaffers_guide.cv.slice_batch_inferencer",
    "gaffers_guide.cv.temporal_ball_prior",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_major_modules_import(module_name: str) -> None:
    imported = importlib.import_module(module_name)
    assert imported is not None
