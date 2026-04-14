from __future__ import annotations

import json
import socket
import time
from pathlib import Path

from services.task_backend import TaskBackend, TaskPayload


class RedisTaskBackend(TaskBackend):
    def __init__(self, redis_url: str, queue_key: str = "exp:task_queue") -> None:
        try:
            import redis  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Redis backend requires `redis` package. Install with `pip install redis`."
            ) from exc
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._queue_key = queue_key
        self._dead_letter_key = f"{queue_key}:dead_letter"
        self._worker_id = socket.gethostname()
        self._processing_key = f"{queue_key}:processing:{self._worker_id}"
        self._max_retries = 3
        self._stale_seconds = 300

    def enqueue(self, task: TaskPayload) -> None:
        payload = {
            "job_id": task.job_id,
            "video_path": str(task.video_path),
            "cv_engine": task.cv_engine,
            "llm_engine": task.llm_engine,
            "decoder_mode": task.decoder_mode,
            "runtime_target": task.runtime_target,
            "hardware_profile": task.hardware_profile,
            "quality_mode": task.quality_mode,
            "chunking_policy": task.chunking_policy,
            "max_parallel_chunks": task.max_parallel_chunks,
            "target_sla_tier": task.target_sla_tier,
            "enqueued_at_epoch_ms": task.enqueued_at_epoch_ms,
            "homography_weights_dir": str(task.homography_weights_dir) if task.homography_weights_dir else None,
            "enqueued_at": time.time(),
            "worker_id": self._worker_id,
            "retry_count": 0,
        }
        self._client.rpush(self._queue_key, json.dumps(payload))

    def dequeue(self) -> TaskPayload | None:
        self._sweep_stale_processing()
        row = self._client.blmove(self._queue_key, self._processing_key, 30, "LEFT", "RIGHT")
        if not row:
            return None
        payload = json.loads(row)
        return TaskPayload(
            job_id=str(payload["job_id"]),
            video_path=Path(str(payload["video_path"])),
            cv_engine=str(payload["cv_engine"]),  # type: ignore[arg-type]
            llm_engine=str(payload["llm_engine"]),  # type: ignore[arg-type]
            decoder_mode=str(payload["decoder_mode"]),  # type: ignore[arg-type]
            runtime_target=str(payload.get("runtime_target", "nvidia")),  # type: ignore[arg-type]
            hardware_profile=str(payload.get("hardware_profile", "l4")),  # type: ignore[arg-type]
            quality_mode=str(payload["quality_mode"]),  # type: ignore[arg-type]
            chunking_policy=str(payload["chunking_policy"]),  # type: ignore[arg-type]
            max_parallel_chunks=int(payload["max_parallel_chunks"]),
            target_sla_tier=str(payload["target_sla_tier"]),  # type: ignore[arg-type]
            enqueued_at_epoch_ms=float(payload["enqueued_at_epoch_ms"]),
            homography_weights_dir=Path(str(payload["homography_weights_dir"])) if payload.get("homography_weights_dir") else None,
            _receipt_processing_key=self._processing_key,
            _receipt_payload_json=row,
        )

    def ack(self, task: TaskPayload) -> None:
        if task._receipt_processing_key and task._receipt_payload_json:
            self._client.lrem(task._receipt_processing_key, 1, task._receipt_payload_json)

    def fail(self, task: TaskPayload, error: str) -> None:
        if task._receipt_processing_key and task._receipt_payload_json:
            self._client.lrem(task._receipt_processing_key, 1, task._receipt_payload_json)
            payload = json.loads(task._receipt_payload_json)
            payload["last_error"] = error
            payload["failed_at"] = time.time()
            self._client.rpush(self._dead_letter_key, json.dumps(payload))

    def _sweep_stale_processing(self) -> None:
        now = time.time()
        rows = self._client.lrange(self._processing_key, 0, -1)
        for row in rows:
            try:
                payload = json.loads(row)
                enq = float(payload.get("enqueued_at", now))
                retry_count = int(payload.get("retry_count", 0))
                if now - enq < self._stale_seconds:
                    continue
                self._client.lrem(self._processing_key, 1, row)
                if retry_count >= self._max_retries:
                    payload["last_error"] = "stale_processing_timeout"
                    payload["failed_at"] = now
                    self._client.rpush(self._dead_letter_key, json.dumps(payload))
                else:
                    payload["retry_count"] = retry_count + 1
                    payload["enqueued_at"] = now
                    self._client.rpush(self._queue_key, json.dumps(payload))
            except Exception:
                continue
