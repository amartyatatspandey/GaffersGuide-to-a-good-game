"""
benchmarks/models.py — Pydantic data models for all benchmark structures.

Schema version: bench.v1
All models are strictly typed. No service layer imports.
"""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Schema constants ─────────────────────────────────────────────────────────

SCHEMA_VERSION = "bench.v1"
PERF_SCHEMA_VERSION = "perf.v1"


# ── Sub-models ───────────────────────────────────────────────────────────────

class VideoProfile(BaseModel):
    filename: str
    sha256: str
    dataset_version: str | None = None
    duration_s: float
    fps: float
    width_px: int
    height_px: int
    total_frames: int
    file_size_bytes: int


class HardwareProfile(BaseModel):
    profile_id: str
    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    gpu_model: str | None = None
    vram_total_gb: float | None = None
    device_type: Literal["cpu", "cuda", "mps"]
    os_version: str
    python_version: str
    torch_version: str

    @staticmethod
    def detect() -> "HardwareProfile":
        """Auto-detect hardware profile from the running system."""
        import hashlib
        import sys

        cpu_model = platform.processor() or platform.machine() or "unknown"
        cpu_cores = 1
        try:
            import psutil
            cpu_cores = psutil.cpu_count(logical=True) or 1
        except ImportError:
            import os
            cpu_cores = os.cpu_count() or 1

        ram_total_gb = 0.0
        try:
            import psutil
            ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except ImportError:
            pass

        gpu_model: str | None = None
        vram_total_gb: float | None = None
        device_type: Literal["cpu", "cuda", "mps"] = "cpu"

        try:
            import torch
            torch_version = torch.__version__
            if torch.cuda.is_available():
                device_type = "cuda"
                gpu_model = torch.cuda.get_device_name(0)
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_total_gb = round(vram_bytes / (1024 ** 3), 2)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_type = "mps"
                gpu_model = "Apple Silicon (MPS)"
        except ImportError:
            torch_version = "not_installed"

        os_version = f"{platform.system()} {platform.release()}"
        python_version = sys.version.split()[0]

        # Stable fingerprint for the hardware
        fingerprint = f"{cpu_model}|{gpu_model}|{ram_total_gb}|{os_version}"
        profile_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]

        return HardwareProfile(
            profile_id=profile_id,
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            ram_total_gb=ram_total_gb,
            gpu_model=gpu_model,
            vram_total_gb=vram_total_gb,
            device_type=device_type,
            os_version=os_version,
            python_version=python_version,
            torch_version=torch_version,
        )


class StageResult(BaseModel):
    stage_id: str
    stage_name: str
    module_id: str
    duration_ms: float
    status: Literal["ok", "error", "skipped"] = "ok"
    chunk_idx: int | None = None
    worker_pid: int | None = None
    frames_processed: int | None = None
    fps_achieved: float | None = None
    ram_start_mb: float | None = None
    ram_end_mb: float | None = None
    ram_delta_mb: float | None = None
    vram_peak_mb: float | None = None
    cpu_pct_mean: float | None = None
    gpu_pct_mean: float | None = None
    io_read_bytes: int | None = None
    io_write_bytes: int | None = None
    sample_rate: int | None = None
    error_message: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModuleSummary(BaseModel):
    module_id: str
    total_duration_ms: float
    pct_of_total: float
    stage_count: int
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None


class LLMCallRecord(BaseModel):
    call_id: str
    job_id: str
    chunk_idx: int | None = None
    engine: Literal["local", "cloud"]
    provider: Literal["gemini", "openai", "ollama", "unknown"] = "unknown"
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float
    retries: int = 0
    outcome: Literal["success", "error", "timeout"] = "success"
    data_guard_active: bool = False
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FrameProfileRecord(BaseModel):
    frame_idx: int
    chunk_idx: int
    sampled: bool = True
    yolo_ms: float
    tracker_ms: float
    homography_ms: float
    optical_flow_ms: float
    detection_count: int
    ball_detected: bool
    homography_conf: float
    fallback_used: bool


class QualityMetrics(BaseModel):
    ball_visibility_ratio: float
    optical_flow_pct: float
    interpolated_ball_pct: float
    total_frames_processed: int
    total_chunks: int
    raw_ball_detections: int
    interpolated_ball_frames: int
    optical_flow_frames: int
    standard_homography_frames: int


class AggregateMetrics(BaseModel):
    # Timing (ms)
    end_to_end_ms: float
    calibration_ms: float = 0.0
    parallel_cv_ms: float = 0.0
    chunk_merge_ms: float = 0.0
    spatial_math_ms: float = 0.0
    metrics_timeline_ms: float = 0.0
    event_intelligence_ms: float = 0.0
    llm_total_ms: float = 0.0
    report_assembly_ms: float = 0.0
    gcs_total_ms: float = 0.0

    # Throughput
    overall_fps: float = 0.0
    cv_fps: float = 0.0
    yolo_fps: float = 0.0

    # Resource peaks
    peak_ram_mb: float = 0.0
    peak_vram_mb: float | None = None
    peak_cpu_pct: float = 0.0

    # I/O totals
    total_io_read_bytes: int = 0
    total_io_write_bytes: int = 0
    gcs_upload_bytes: int = 0
    gcs_download_bytes: int = 0

    # LLM aggregate
    total_llm_calls: int = 0
    total_llm_latency_ms: float = 0.0
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None

    # Hardware context
    hardware_profile_id: str = ""
    pipeline_version: str = ""


class RegressionFlag(BaseModel):
    metric_id: str
    stage_id: str | None = None
    baseline_value: float
    current_value: float
    delta_pct: float
    tolerance_pct: float
    severity: Literal["WARNING", "CRITICAL"]
    description: str


class BenchmarkConfig(BaseModel):
    benchmark_mode: bool = False
    frame_profiling: bool = False
    enable_cprofile: bool = False
    sample_rate: int = 10
    gpu_sample_interval_s: float = 2.0
    dataset_tier: Literal["smoke", "standard", "full"] | None = None
    comparison_baseline_path: str | None = None

    @classmethod
    def from_env(cls) -> "BenchmarkConfig":
        import os
        return cls(
            benchmark_mode=os.getenv("BENCHMARK_MODE", "false").lower() in ("true", "1", "yes"),
            frame_profiling=os.getenv("FRAME_PROFILING", "false").lower() in ("true", "1", "yes"),
            enable_cprofile=os.getenv("ENABLE_CPROFILE", "false").lower() in ("true", "1", "yes"),
        )


class BenchmarkReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    benchmark_run_id: str
    job_id: str
    pipeline_version: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hardware_profile: HardwareProfile
    video_profile: VideoProfile | None = None
    stages: list[StageResult] = Field(default_factory=list)
    module_summary: list[ModuleSummary] = Field(default_factory=list)
    aggregate: AggregateMetrics
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    quality_metrics: QualityMetrics | None = None
    regression_flags: list[RegressionFlag] = Field(default_factory=list)
    pipeline_dag: dict[str, list[str]] = Field(default_factory=dict)
    benchmark_config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @property
    def has_regressions(self) -> bool:
        return len(self.regression_flags) > 0

    @property
    def has_critical_regressions(self) -> bool:
        return any(f.severity == "CRITICAL" for f in self.regression_flags)


class DatasetEntry(BaseModel):
    version: str
    tier: Literal["smoke", "standard", "full"]
    gcs_uri: str
    sha256: str
    duration_s: float
    fps: float
    width: int
    height: int
    registered_at: datetime
    registered_by: str


# ── Canonical pipeline DAG (stage dependency map) ───────────────────────────

PIPELINE_DAG: dict[str, list[str]] = {
    "stage.upload": ["stage.calibration"],
    "stage.calibration": ["stage.parallel_cv"],
    "stage.parallel_cv": ["stage.chunk_merge"],
    "stage.worker_init": [],
    "stage.worker_tracking": [],
    "stage.chunk_merge": ["stage.spatial_math"],
    "stage.spatial_math": ["stage.metrics_timeline"],
    "stage.metrics_timeline": ["stage.event_intelligence"],
    "stage.event_intelligence": ["stage.llm_advice"],
    "stage.llm_advice": ["stage.report_assembly"],
    "stage.report_assembly": ["stage.gcs_upload"],
    "stage.gcs_upload": [],
}

# Module attribution: stage_id -> module_id
STAGE_MODULE_MAP: dict[str, str] = {
    "stage.upload": "MOD-UPL",
    "stage.calibration": "MOD-CAL",
    "stage.parallel_cv": "MOD-PAR",
    "stage.worker_init": "MOD-DET",
    "stage.worker_tracking": "MOD-TRK",
    "stage.chunk_merge": "MOD-SPI",
    "stage.spatial_math": "MOD-SPI",
    "stage.metrics_timeline": "MOD-TAC",
    "stage.event_intelligence": "MOD-EVT",
    "stage.llm_advice": "MOD-AIR",
    "stage.report_assembly": "MOD-RPT",
    "stage.gcs_upload": "MOD-RPT",
}
