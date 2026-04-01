from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.gpu_runtime import select_gpu_runtime


def test_nvidia_runtime_selection_uses_gpu_backend() -> None:
    cfg = select_gpu_runtime(runtime_target="nvidia", hardware_profile="l4", cv_engine="cloud")
    assert cfg.backend in ("tensorrt", "onnxruntime-gpu")
    assert cfg.batch_size >= 1


def test_mps_runtime_selection_falls_back_when_unavailable() -> None:
    cfg = select_gpu_runtime(runtime_target="apple_mps", hardware_profile="mps", cv_engine="local")
    assert cfg.backend in ("apple-mps", "cpu-fallback")
