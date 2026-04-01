from __future__ import annotations

import os
from dataclasses import dataclass

from services.model_profiles import resolve_profile


@dataclass(slots=True)
class GpuRuntimeConfig:
    backend: str
    batch_size: int
    precision: str
    concurrency_cap: int


def select_gpu_runtime(
    *,
    runtime_target: str = "nvidia",
    hardware_profile: str = "l4",
    cv_engine: str = "cloud",
    prefer_tensorrt: bool = True,
) -> GpuRuntimeConfig:
    profile = resolve_profile(hardware_profile)
    if runtime_target == "apple_mps":
        return _select_mps_runtime()
    if runtime_target == "cpu_fallback" or cv_engine != "cloud":
        return GpuRuntimeConfig(
            backend="cpu", batch_size=1, precision="fp32", concurrency_cap=1
        )
    if prefer_tensorrt:
        return GpuRuntimeConfig(
            backend="tensorrt",
            batch_size=profile.batch_size,
            precision="fp16",
            concurrency_cap=profile.concurrency_cap,
        )
    return GpuRuntimeConfig(
        backend="onnxruntime-gpu",
        batch_size=max(1, profile.batch_size // 2),
        precision="fp16",
        concurrency_cap=profile.concurrency_cap,
    )


def _select_mps_runtime() -> GpuRuntimeConfig:
    # Keep import optional so Linux/NVIDIA builds do not require torch.
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return GpuRuntimeConfig(
            backend="cpu-fallback", batch_size=1, precision="fp32", concurrency_cap=1
        )

    if not torch.backends.mps.is_available():
        return GpuRuntimeConfig(
            backend="cpu-fallback", batch_size=1, precision="fp32", concurrency_cap=1
        )
    batch_size = int(os.getenv("EXP_MPS_BATCH_SIZE", "4"))
    concurrency = int(os.getenv("EXP_MPS_CONCURRENCY_CAP", "2"))
    return GpuRuntimeConfig(
        backend="apple-mps",
        batch_size=max(1, batch_size),
        precision="fp16",
        concurrency_cap=max(1, concurrency),
    )
