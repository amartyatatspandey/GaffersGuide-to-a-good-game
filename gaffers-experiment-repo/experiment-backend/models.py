from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DecoderMode = Literal["opencv", "pyav"]
EngineMode = Literal["local", "cloud"]
JobStatusValue = Literal["pending", "processing", "done", "error"]
QualityMode = Literal["fast", "balanced", "high"]
ChunkingPolicy = Literal["none", "fixed", "auto"]
SlaTier = Literal["tier_10m", "tier_5m"]
RuntimeTarget = Literal["nvidia", "apple_mps", "cpu_fallback"]
HardwareProfile = Literal["l4", "a10", "a100", "mps", "cpu"]


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatusValue
    cv_engine: EngineMode
    llm_engine: EngineMode
    decoder_mode: DecoderMode
    runtime_target: RuntimeTarget
    hardware_profile: HardwareProfile


class JobCreateOptions(BaseModel):
    runtime_target: RuntimeTarget = "nvidia"
    hardware_profile: HardwareProfile = "l4"
    quality_mode: QualityMode = "balanced"
    chunking_policy: ChunkingPolicy = "fixed"
    max_parallel_chunks: int = Field(default=2, ge=1, le=32)
    target_sla_tier: SlaTier = "tier_10m"


class ChatRequest(BaseModel):
    message: str = Field(description="User coaching question.")
    job_id: str | None = Field(default=None, description="Optional completed job id.")


class ChatResponse(BaseModel):
    reply: str


class ReportEntry(BaseModel):
    job_id: str
    created_at: str
    report_filename: str


class ReportsResponse(BaseModel):
    reports: list[ReportEntry]


class AdviceItem(BaseModel):
    frame_idx: int
    team: str
    flaw: str
    severity: str
    evidence: str
    matched_philosophy_author: str
    tactical_instruction: str
    tactical_instruction_steps: list[str]


class AdviceResponse(BaseModel):
    generated_at: str
    pipeline: dict[str, str]
    advice_items: list[AdviceItem]


class StageTelemetry(BaseModel):
    queue_wait_ms: float = 0.0
    decode_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0
    frames_processed: int = 0
    effective_fps: float = 0.0
    reid_invocations: int = 0
    reid_ms: float = 0.0
    id_switch_rate: float = 0.0
