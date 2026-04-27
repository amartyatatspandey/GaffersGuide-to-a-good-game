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
