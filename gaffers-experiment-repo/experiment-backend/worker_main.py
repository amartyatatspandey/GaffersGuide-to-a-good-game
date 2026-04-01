from __future__ import annotations

import asyncio
import os
from pathlib import Path

from services.job_store import ExperimentJobStore
from services.observability import MetricsRegistry
from services.queue import ExperimentQueue
from services.task_backend_local import LocalFileTaskBackend
from services.task_backend_redis import RedisTaskBackend

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "exp"
STORE_PATH = OUTPUT_DIR / "exp_jobs_store.json"
TASK_QUEUE_PATH = OUTPUT_DIR / "exp_task_queue.json"


async def run() -> None:
    store = ExperimentJobStore(STORE_PATH)
    metrics = MetricsRegistry()
    queue = ExperimentQueue(store, metrics, OUTPUT_DIR)
    backend_name = os.getenv("EXP_TASK_BACKEND", "local").lower()
    if backend_name == "redis":
        redis_url = os.getenv("EXP_REDIS_URL", "redis://127.0.0.1:6379/0")
        queue_key = os.getenv("EXP_REDIS_QUEUE_KEY", "exp:task_queue")
        backend = RedisTaskBackend(redis_url, queue_key=queue_key)
    else:
        backend = LocalFileTaskBackend(TASK_QUEUE_PATH)
    while True:
        task = backend.dequeue()
        if task is None:
            await asyncio.sleep(0.25)
            continue
        await queue.process_task(task)


if __name__ == "__main__":
    asyncio.run(run())
