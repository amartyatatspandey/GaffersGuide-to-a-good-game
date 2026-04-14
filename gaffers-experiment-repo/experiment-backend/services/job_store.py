from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentJob:
    job_id: str
    status: str
    current_step: str
    cv_engine: str
    llm_engine: str
    decoder_mode: str
    source_video_path: str
    runtime_target: str = "nvidia"
    hardware_profile: str = "l4"
    source_video_uri: str | None = None
    quality_mode: str = "balanced"
    chunking_policy: str = "fixed"
    max_parallel_chunks: int = 2
    target_sla_tier: str = "tier_10m"
    stage: str = "queued"
    result_path: str | None = None
    result_uri: str | None = None
    tracking_data_path: str | None = None
    tracking_data_uri: str | None = None
    chunks: list[dict[str, object]] | None = None
    queue_wait_ms: float = 0.0
    decode_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0
    frames_processed: int = 0
    effective_fps: float = 0.0
    reid_invocations: int = 0
    reid_ms: float = 0.0
    id_switch_rate: float = 0.0
    frames_with_homography: int = 0
    frames_without_homography: int = 0
    fallback_frames: int = 0
    calibration_latency_ms: float = 0.0
    error: str | None = None
    idempotency_key: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ExperimentJobStore:
    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._lock = threading.Lock()
        self._jobs: dict[str, ExperimentJob] = {}
        self._idempotency: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for job_id, payload in raw.get("jobs", {}).items():
            self._jobs[job_id] = ExperimentJob(**payload)
        self._idempotency = {
            str(key): str(value) for key, value in raw.get("idempotency", {}).items()
        }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "jobs": {job_id: asdict(job) for job_id, job in self._jobs.items()},
            "idempotency": self._idempotency,
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def create(self, job: ExperimentJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            if job.idempotency_key:
                self._idempotency[job.idempotency_key] = job.job_id
            self._save()

    def update(self, job_id: str, **fields: Any) -> ExperimentJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            self._save()
            return job

    def get(self, job_id: str) -> ExperimentJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[ExperimentJob]:
        with self._lock:
            return list(self._jobs.values())

    def find_by_idempotency(self, key: str) -> ExperimentJob | None:
        with self._lock:
            job_id = self._idempotency.get(key)
            if not job_id:
                return None
            return self._jobs.get(job_id)
