from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChunkTacticalInsight(BaseModel):
    """
    Aggregated tactical flaw over an entire video chunk (macro-trend).

    frequency_pct is the percent of frames in which the rule was violated.
    """

    team_id: Literal["team_0", "team_1"]
    flaw: str
    severity: str
    frequency_pct: float = Field(..., ge=0.0, le=100.0)
    evidence: str
    ball_data_quality: Literal["sufficient", "insufficient"] = "sufficient"


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


class ChatRequest(BaseModel):
    message: str = Field(description="User coaching question / request.")
    job_id: str | None = Field(
        default=None,
        description="Optional job id whose tactical insights should be used as context.",
    )
    llm_engine: Literal["local", "cloud"] | None = Field(
        default=None,
        description="Optional override to route LLM to local Ollama or cloud API.",
    )


class ChatResponse(BaseModel):
    reply: str


class ReportEntry(BaseModel):
    job_id: str
    created_at: str
    report_filename: str


class ReportsResponse(BaseModel):
    reports: list[ReportEntry]


class EngineSelection(BaseModel):
    cv_engine: Literal["local", "cloud"] = "cloud"
    llm_engine: Literal["local", "cloud"] = "cloud"


class ErrorResponse(BaseModel):
    code: str
    message: str
    hint: str | None = None


class JobArtifactsResponse(BaseModel):
    """Artifact paths returned after a job completes."""

    job_id: str
    status: Literal["pending", "processing", "done", "error"]
    report_path: str | None = None
    tracking_overlay_path: str | None = None
    tracking_data_path: str | None = None
    report_state: Literal["ready", "not_ready"] = "not_ready"
    tracking_state: Literal["ready", "not_ready"] = "not_ready"
    overlay_state: Literal["ready", "not_ready", "unavailable"] = "not_ready"
    overlay_reason: str | None = None


class BetaJobResponse(BaseModel):
    """Full record returned by GET /api/v1beta/jobs/{job_id}."""

    job_id: str
    status: Literal["pending", "processing", "done", "error"]
    current_step: str
    result_path: str | None = None
    tracking_overlay_path: str | None = None
    tracking_data_path: str | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class JobProgressMessage(BaseModel):
    """WebSocket progress frame for both v1 and v1beta."""

    job_id: str
    status: Literal["pending", "processing", "done", "error"]
    current_step: str
    result_path: str | None = None
    tracking_overlay_path: str | None = None
    tracking_data_path: str | None = None
    error: str | None = None


class LocalLlmPreflightResponse(BaseModel):
    configured_base_url: str
    configured_model: str
    daemon_reachable: bool
    model_present: bool
    generation_ok: bool
    error: str | None = None
    hint: str | None = None

