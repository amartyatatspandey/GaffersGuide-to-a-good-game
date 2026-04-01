from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    gpu_sku: str
    batch_size: int
    concurrency_cap: int


MODEL_PROFILES: dict[str, ModelProfile] = {
    "l4": ModelProfile(gpu_sku="l4", batch_size=8, concurrency_cap=2),
    "a10": ModelProfile(gpu_sku="a10", batch_size=12, concurrency_cap=3),
    "a100": ModelProfile(gpu_sku="a100", batch_size=16, concurrency_cap=4),
}


def resolve_profile(gpu_sku: str) -> ModelProfile:
    return MODEL_PROFILES.get(gpu_sku.lower(), MODEL_PROFILES["l4"])
