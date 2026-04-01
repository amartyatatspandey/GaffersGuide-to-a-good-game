from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from services.cv_pipeline import DecoderMode, process_video
from services.job_store import ExperimentJobStore
from services.observability import MetricsRegistry
from services.task_backend import TaskPayload


@dataclass(slots=True)
class QueueItem:
    job_id: str
    video_path: Path
    decoder_mode: DecoderMode
    cv_engine: str = "local"
    llm_engine: str = "local"
    runtime_target: str = "nvidia"
    hardware_profile: str = "l4"
    quality_mode: str = "balanced"
    chunking_policy: str = "fixed"
    max_parallel_chunks: int = 2
    target_sla_tier: str = "tier_10m"
    enqueued_at_epoch_ms: float = 0.0


class ExperimentQueue:
    def __init__(self, store: ExperimentJobStore, metrics: MetricsRegistry, output_dir: Path) -> None:
        self._store = store
        self._metrics = metrics
        self._output_dir = output_dir
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker())

    async def enqueue(self, item: QueueItem) -> None:
        await self._queue.put(item)
        self._metrics.incr("exp.jobs.enqueued")

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            await self.process_item(item)
            self._queue.task_done()

    async def process_task(self, task: TaskPayload) -> None:
        await self.process_item(
            QueueItem(
                job_id=task.job_id,
                video_path=task.video_path,
                decoder_mode=task.decoder_mode,
                cv_engine=task.cv_engine,
                llm_engine=task.llm_engine,
                runtime_target=task.runtime_target,
                hardware_profile=task.hardware_profile,
                quality_mode=task.quality_mode,
                chunking_policy=task.chunking_policy,
                max_parallel_chunks=task.max_parallel_chunks,
                target_sla_tier=task.target_sla_tier,
                enqueued_at_epoch_ms=task.enqueued_at_epoch_ms,
            )
        )

    async def process_item(self, item: QueueItem) -> None:
        now = datetime.now(timezone.utc).isoformat()
        queue_wait_ms = max(0.0, (time.time() * 1000.0) - item.enqueued_at_epoch_ms)
        self._store.update(
            item.job_id,
            status="processing",
            current_step="Decoding",
            stage="decode",
            queue_wait_ms=round(queue_wait_ms, 2),
            updated_at=now,
        )
        try:
            with self._metrics.timed("exp.job.total_ms"):
                artifacts = await asyncio.to_thread(
                    process_video,
                    item.video_path,
                    output_dir=self._output_dir,
                    output_prefix=item.job_id,
                    decoder_mode=item.decoder_mode,
                    cv_engine=item.cv_engine,
                    runtime_target=item.runtime_target,
                    hardware_profile=item.hardware_profile,
                    quality_mode=item.quality_mode,
                    chunking_policy=item.chunking_policy,
                    max_parallel_chunks=item.max_parallel_chunks,
                )
            self._store.update(
                item.job_id,
                status="done",
                current_step="Completed",
                stage="complete",
                result_path=str(artifacts.report_path),
                result_uri=f"file://{artifacts.report_path}",
                tracking_data_path=str(artifacts.tracking_path),
                tracking_data_uri=f"file://{artifacts.tracking_path}",
                decode_ms=round(artifacts.decode_ms, 2),
                infer_ms=round(artifacts.infer_ms, 2),
                post_ms=round(artifacts.post_ms, 2),
                frames_processed=artifacts.frames_processed,
                effective_fps=round(artifacts.effective_fps, 2),
                reid_invocations=artifacts.reid_invocations,
                reid_ms=round(artifacts.reid_ms, 2),
                id_switch_rate=artifacts.id_switch_rate,
                chunks=artifacts.chunks,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._metrics.incr("exp.jobs.succeeded")
        except Exception as exc:  # noqa: BLE001
            self._store.update(
                item.job_id,
                status="error",
                current_step="Error",
                stage="error",
                error=str(exc),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._metrics.incr("exp.jobs.failed")
