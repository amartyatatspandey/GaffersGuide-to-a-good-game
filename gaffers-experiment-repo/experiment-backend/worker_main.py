from __future__ import annotations

import asyncio
import os
import shutil

from services.job_store import ExperimentJobStore
from services.observability import MetricsRegistry
from services.paths import OUTPUT_ROOT, STORE_PATH
from services.queue import ExperimentQueue
from services.task_backend_factory import build_task_backend

OUTPUT_DIR = OUTPUT_ROOT


def ensure_runtime_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preflight_check() -> None:
    ensure_runtime_directories()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for worker runtime.")


async def run() -> None:
    preflight_check()
    store = ExperimentJobStore(STORE_PATH)
    metrics = MetricsRegistry()
    queue = ExperimentQueue(store, metrics, OUTPUT_DIR)
    backend = build_task_backend()
    idle_sleep_seconds = float(os.getenv("EXP_WORKER_IDLE_SLEEP_SECONDS", "0.25"))
    while True:
        task = backend.dequeue()
        if task is None:
            await asyncio.sleep(max(0.05, idle_sleep_seconds))
            continue
        try:
            await queue.process_task(task)
            backend.ack(task)
        except Exception as exc:  # noqa: BLE001
            backend.fail(task, str(exc))


if __name__ == "__main__":
    asyncio.run(run())
