from __future__ import annotations

import json
import threading
from pathlib import Path

from services.task_backend import TaskBackend, TaskPayload


class LocalFileTaskBackend(TaskBackend):
    def __init__(self, queue_path: Path) -> None:
        self._queue_path = queue_path
        self._lock = threading.Lock()
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._queue_path.is_file():
            self._queue_path.write_text("[]", encoding="utf-8")

    def _read(self) -> list[dict[str, object]]:
        return json.loads(self._queue_path.read_text(encoding="utf-8"))

    def _write(self, rows: list[dict[str, object]]) -> None:
        self._queue_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def enqueue(self, task: TaskPayload) -> None:
        with self._lock:
            rows = self._read()
            rows.append(
                {
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
            )
            self._write(rows)

    def dequeue(self) -> TaskPayload | None:
        with self._lock:
            rows = self._read()
            if not rows:
                return None
            first = rows.pop(0)
            self._write(rows)
        return TaskPayload(
            job_id=str(first["job_id"]),
            video_path=Path(str(first["video_path"])),
            cv_engine=str(first["cv_engine"]),  # type: ignore[arg-type]
            llm_engine=str(first["llm_engine"]),  # type: ignore[arg-type]
            decoder_mode=str(first["decoder_mode"]),  # type: ignore[arg-type]
            runtime_target=str(first.get("runtime_target", "nvidia")),  # type: ignore[arg-type]
            hardware_profile=str(first.get("hardware_profile", "l4")),  # type: ignore[arg-type]
            quality_mode=str(first["quality_mode"]),  # type: ignore[arg-type]
            chunking_policy=str(first["chunking_policy"]),  # type: ignore[arg-type]
            max_parallel_chunks=int(first["max_parallel_chunks"]),
            target_sla_tier=str(first["target_sla_tier"]),  # type: ignore[arg-type]
            enqueued_at_epoch_ms=float(first["enqueued_at_epoch_ms"]),
            homography_weights_dir=Path(str(first["homography_weights_dir"])) if first.get("homography_weights_dir") else None,
        )

    def ack(self, task: TaskPayload) -> None:
        _ = task

    def fail(self, task: TaskPayload, error: str) -> None:
        _ = (task, error)
