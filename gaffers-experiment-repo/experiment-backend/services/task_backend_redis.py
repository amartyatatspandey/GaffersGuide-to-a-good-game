from __future__ import annotations

import json
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
        }
        self._client.rpush(self._queue_key, json.dumps(payload))

    def dequeue(self) -> TaskPayload | None:
        row = self._client.lpop(self._queue_key)
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
        )
