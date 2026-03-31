from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BetaJobRecord:
    job_id: str
    status: str
    current_step: str
    cv_engine: str
    llm_engine: str
    source_video_path: str
    result_path: str | None = None
    tracking_overlay_path: str | None = None
    tracking_data_path: str | None = None
    error: str | None = None
    idempotency_key: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class BetaJobStore:
    """Thread-safe JSON-backed persistent store for beta jobs."""

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path
        self._lock = threading.Lock()
        self._jobs: dict[str, BetaJobRecord] = {}
        self._idempotency_index: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._store_path.is_file():
            return
        with self._store_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        jobs_raw = raw.get("jobs", {})
        idx_raw = raw.get("idempotency_index", {})
        for job_id, payload in jobs_raw.items():
            self._jobs[job_id] = BetaJobRecord(**payload)
        self._idempotency_index = {str(k): str(v) for k, v in idx_raw.items()}

    def _persist(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "jobs": {job_id: asdict(rec) for job_id, rec in self._jobs.items()},
            "idempotency_index": self._idempotency_index,
        }
        with self._store_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def create(self, rec: BetaJobRecord) -> BetaJobRecord:
        with self._lock:
            self._jobs[rec.job_id] = rec
            if rec.idempotency_key:
                self._idempotency_index[rec.idempotency_key] = rec.job_id
            self._persist()
            return rec

    def find_by_idempotency(self, key: str) -> BetaJobRecord | None:
        with self._lock:
            job_id = self._idempotency_index.get(key)
            if not job_id:
                return None
            return self._jobs.get(job_id)

    def get(self, job_id: str) -> BetaJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> BetaJobRecord | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            for key, value in fields.items():
                if hasattr(rec, key):
                    setattr(rec, key, value)
            self._persist()
            return rec

    def delete(self, job_id: str) -> None:
        with self._lock:
            rec = self._jobs.pop(job_id, None)
            if rec and rec.idempotency_key:
                self._idempotency_index.pop(rec.idempotency_key, None)
            self._persist()
