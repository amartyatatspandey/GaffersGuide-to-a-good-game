from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from models import (
    ChunkingPolicy,
    DecoderMode,
    EngineMode,
    HardwareProfile,
    QualityMode,
    RuntimeTarget,
    SlaTier,
)


@dataclass(slots=True)
class TaskPayload:
    job_id: str
    video_path: Path
    cv_engine: EngineMode
    llm_engine: EngineMode
    decoder_mode: DecoderMode
    runtime_target: RuntimeTarget
    hardware_profile: HardwareProfile
    quality_mode: QualityMode
    chunking_policy: ChunkingPolicy
    max_parallel_chunks: int
    target_sla_tier: SlaTier
    enqueued_at_epoch_ms: float


class TaskBackend(Protocol):
    def enqueue(self, task: TaskPayload) -> None: ...
    def dequeue(self) -> TaskPayload | None: ...
