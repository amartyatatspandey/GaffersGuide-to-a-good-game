"""Cython extension build configuration for protected modules.

This file is intentionally separate from package metadata. It can be run
directly for local extension builds and is imported by setup.py for wheel
builds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py


CV_MODULES = [
    "temporal_ball_prior",
    "pitch_roi_provider",
    "ball_candidate_fuser",
    "slice_batch_inferencer",
    "optimized_sahi_wrapper",
]

PIPELINE_MODULES = [
    "advanced_pitch_calibration",
    "dynamic_homography",
    "track_teams",
    "track_teams_reid_hybrid",
    "reid_healer",
    "global_refiner",
    "e2e_shared_impl",
    "tactical_radar",
    "generate_analytics",
]

COMPILED_MODULES = {
    *(f"gaffers_guide.cv.{name}" for name in CV_MODULES),
    *(f"gaffers_guide.pipeline.{name}" for name in PIPELINE_MODULES),
}
COMPILED_SOURCE_NAMES = {
    *(f"{name}.py" for name in CV_MODULES),
    *(f"{name}.pyx" for name in CV_MODULES),
    *(f"{name}.c" for name in CV_MODULES),
    *(f"{name}.py" for name in PIPELINE_MODULES),
    *(f"{name}.pyx" for name in PIPELINE_MODULES),
    *(f"{name}.c" for name in PIPELINE_MODULES),
}


def _extension(module_name: str, source: str) -> Extension:
    return Extension(
        module_name,
        [source],
        include_dirs=[np.get_include()],
    )


def get_extensions() -> list[Extension]:
    extensions: list[Extension] = []
    for name in CV_MODULES:
        extensions.append(
            _extension(
                f"gaffers_guide.cv.{name}",
                f"src/gaffers_guide/cv/{name}.pyx",
            )
        )
    for name in PIPELINE_MODULES:
        extensions.append(
            _extension(
                f"gaffers_guide.pipeline.{name}",
                f"src/gaffers_guide/pipeline/{name}.pyx",
            )
        )
    return extensions


class build_py_without_compiled_sources(_build_py):
    """Exclude readable sources for modules that ship as binary extensions."""

    def find_package_modules(self, package: str, package_dir: str) -> list[tuple[str, str, str]]:
        modules = super().find_package_modules(package, package_dir)
        return [
            item
            for item in modules
            if f"{item[0]}.{item[1]}" not in COMPILED_MODULES
        ]

    def find_data_files(self, package: str, src_dir: str) -> list[str]:
        files = super().find_data_files(package, src_dir)
        if package not in {"gaffers_guide.cv", "gaffers_guide.pipeline"}:
            return files
        return [
            filename
            for filename in files
            if Path(filename).name not in COMPILED_SOURCE_NAMES
        ]


def build_setup_kwargs() -> dict[str, object]:
    return {
        "ext_modules": cythonize(
            get_extensions(),
            compiler_directives={
                "language_level": "3",
                "embedsignature": True,
            },
            annotate=False,
        ),
        "cmdclass": {"build_py": build_py_without_compiled_sources},
    }


if __name__ == "__main__":
    # Ensure direct local builds behave like PEP 517 wheel builds.
    setup(**build_setup_kwargs())
