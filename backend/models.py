from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChunkTacticalInsight(BaseModel):
    """
    Aggregated tactical flaw over an entire video chunk (macro-trend).

    frequency_pct is the percent of frames in which the rule was violated.
    """

    team_id: Literal["team_0", "team_1", "global"]
    flaw: str
    severity: str
    frequency_pct: float = Field(..., ge=0.0, le=100.0)
    evidence: str
    ball_data_quality: Literal["sufficient", "insufficient"] = "sufficient"
    confidence_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    confidence_reason: str | None = None


class JobStatus(BaseModel):
    """Job tracking for asynchronous video processing."""

    job_id: str
    status: Literal["pending", "processing", "done", "error"]
    current_step: str
    result_path: str | None = None
    error: str | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "done", "error"]
    cv_engine: Literal["local", "cloud"] = "cloud"
    llm_engine: Literal["local", "cloud"] = "cloud"
    quality_profile: str = "balanced"
    chunking_interval: str = "15-minute intervals"


class ChatRequest(BaseModel):
    message: str = Field(description="User coaching question / request.")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Optional conversation history for multi-turn context.",
    )
    job_id: str | None = Field(
        default=None,
        description="Optional job id whose tactical insights should be used as context.",
    )
    llm_engine: Literal["local", "cloud"] | None = Field(
        default=None,
        description="Optional override to route LLM to local Ollama or cloud API.",
    )


class EvidenceAttachment(BaseModel):
    clips: list[dict]               # ClipRecord dicts
    top_threats: list[dict]         # PlayerThreatProfile dicts (top 3)
    observation: str                # Text that triggered retrieval


class ChatResponse(BaseModel):
    reply: str
    evidence: EvidenceAttachment | None = None  # NEW — opt-in, None by default


class EnrichedCoachingCard(BaseModel):
    """Extends raw LLM coaching card with evidence fields."""
    model_config = {"extra": "allow"}

    job_id: str
    flaw: str
    severity: str
    evidence: str
    team: str
    frequency_pct: float
    evidence_clips: list[dict] = []
    threat_context: dict = {}
    event_count_summary: dict[str, int] = {}
    # Add these missing fields:
    frame_idx: Optional[int] = None
    tactical_instruction: Optional[str] = None
    matched_philosophy_author: Optional[str] = None
    matched_quote_excerpt: Optional[str] = None
    fc_role_recommendations: Optional[list] = None
    confidence_pct: Optional[float] = None
    confidence_reason: Optional[str] = None
    llm_error: Optional[str] = None
    minute: Optional[str] = None
    timestamp: Optional[str] = None
    ball_data_quality: Optional[str] = None
    llm_prompt: Optional[str] = None


class ReportEntry(BaseModel):
    job_id: str
    created_at: str
    report_filename: str


class ReportsResponse(BaseModel):
    reports: list[ReportEntry]


class DatasetInfo(BaseModel):
    """One dataset folder under DATASETS_ROOT (optional API for tooling UIs)."""

    name: str
    split: str
    num_samples: int
    root_dir: str


class DatasetsListResponse(BaseModel):
    datasets: list[DatasetInfo]


class EngineSelection(BaseModel):
    cv_engine: Literal["local", "cloud"] = "cloud"
    llm_engine: Literal["local", "cloud"] = "cloud"


class ErrorResponse(BaseModel):
    code: str
    message: str
    hint: str | None = None

