from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from services.beta_job_store import BetaJobStore
from services.cv_router import CVRouterFactory
from services.errors import EngineRoutingError
from services.observability import PipelineMetricsRegistry

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BetaQueueItem:
    job_id: str
    video_path: Path
    cv_engine: str


class BetaPipelineQueue:
    """Queue-backed execution model for beta job isolation."""

    def __init__(self, job_store: BetaJobStore, metrics: PipelineMetricsRegistry) -> None:
        self._job_store = job_store
        self._metrics = metrics
        self._queue: asyncio.Queue[BetaQueueItem] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def enqueue(self, item: BetaQueueItem) -> None:
        await self._queue.put(item)
        self._metrics.incr("beta.jobs.enqueued")

    async def _worker_loop(self) -> None:
        while True:
            item = await self._queue.get()
            self._metrics.incr("beta.jobs.dequeued")
            started = datetime.now(timezone.utc).isoformat()
            self._job_store.update(
                item.job_id,
                status="processing",
                current_step="Tracking Players",
                updated_at=started,
            )
            try:
                runner = CVRouterFactory.get(item.cv_engine)  # type: ignore[arg-type]

                def progress(step: str) -> None:
                    self._job_store.update(
                        item.job_id,
                        current_step=step,
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )

                with self._metrics.timed("beta.job.e2e.total_ms"):
                    report_path = await runner.run(
                        job_id=item.job_id,
                        video_path=item.video_path,
                        progress_callback=progress,
                    )
                output_dir = Path(__file__).resolve().parent.parent / "output"
                overlay = output_dir / f"{item.job_id}_tracking_overlay.mp4"
                tracking = output_dir / f"{item.job_id}_tracking_data.json"
                self._job_store.update(
                    item.job_id,
                    status="done",
                    current_step="Completed",
                    result_path=str(report_path),
                    tracking_overlay_path=str(overlay) if overlay.is_file() else None,
                    tracking_data_path=str(tracking) if tracking.is_file() else None,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                self._metrics.incr("beta.jobs.succeeded")
            except EngineRoutingError as exc:
                self._job_store.update(
                    item.job_id,
                    status="error",
                    current_step="Error",
                    error=f"{exc.code}: {exc.message}",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                self._metrics.incr("beta.jobs.failed")
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Beta queue worker failed for %s", item.job_id)
                self._job_store.update(
                    item.job_id,
                    status="error",
                    current_step="Error",
                    error=str(exc),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                self._metrics.incr("beta.jobs.failed")
            finally:
                self._queue.task_done()
