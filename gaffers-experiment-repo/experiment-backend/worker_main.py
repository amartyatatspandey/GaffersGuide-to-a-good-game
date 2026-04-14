from __future__ import annotations

import asyncio
import logging
import os
import shutil
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
LOGGER = logging.getLogger(__name__)


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
    cloud_mode = os.getenv("EXP_CLOUD_MODE", "0") == "1"
    backend_name = os.getenv("EXP_TASK_BACKEND", "redis").lower()
    if cloud_mode and backend_name != "redis":
        raise RuntimeError("EXP_CLOUD_MODE=1 requires EXP_TASK_BACKEND=redis.")
    if backend_name == "redis":
        redis_url = os.getenv("EXP_REDIS_URL", "redis://127.0.0.1:6379/0")
        queue_key = os.getenv("EXP_REDIS_QUEUE_KEY", "exp:task_queue")
        try:
            backend = RedisTaskBackend(redis_url, queue_key=queue_key)
        except Exception as exc:  # noqa: BLE001
            fallback_default = "0" if cloud_mode else "1"
            if os.getenv("EXP_ALLOW_LOCAL_BACKEND_FALLBACK", fallback_default) == "1":
                LOGGER.warning("Redis backend unavailable, falling back to local task backend: %s", exc)
                backend = LocalFileTaskBackend(TASK_QUEUE_PATH)
            else:
                raise
    else:
        backend = LocalFileTaskBackend(TASK_QUEUE_PATH)
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
