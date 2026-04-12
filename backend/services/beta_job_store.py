from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


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


def _record_from_payload(job_id: str, payload: dict[str, Any]) -> BetaJobRecord:
    """Build a record from JSON, dropping keys not on BetaJobRecord (schema drift)."""
    allowed = {f.name for f in fields(BetaJobRecord)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        LOGGER.warning(
            "Beta job store %s: dropping unknown keys %s",
            job_id,
            unknown,
        )
    filtered = {k: payload[k] for k in payload if k in allowed}
    return BetaJobRecord(**filtered)


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
        try:
            with self._store_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.error(
                "Beta job store: failed to read %s (%s). Starting with empty store.",
                self._store_path,
                exc,
            )
            return
        jobs_raw = raw.get("jobs", {}) if isinstance(raw, dict) else {}
        idx_raw = raw.get("idempotency_index", {}) if isinstance(raw, dict) else {}
        loaded = 0
        skipped = 0
        for job_id, payload in jobs_raw.items():
            if not isinstance(payload, dict):
                LOGGER.warning(
                    "Beta job store: skipping job %r (not an object)",
                    job_id,
                )
                skipped += 1
                continue
            try:
                self._jobs[job_id] = _record_from_payload(job_id, payload)
                loaded += 1
            except TypeError as exc:
                LOGGER.warning(
                    "Beta job store: skipping job %r (record build failed: %s)",
                    job_id,
                    exc,
                )
                skipped += 1
        self._idempotency_index = {str(k): str(v) for k, v in idx_raw.items()}
        if skipped:
            LOGGER.warning(
                "Beta job store: loaded %d jobs, skipped %d malformed records from %s",
                loaded,
                skipped,
                self._store_path,
            )
        else:
            LOGGER.info(
                "Beta job store: loaded %d jobs from %s",
                loaded,
                self._store_path,
            )

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
