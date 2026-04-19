from __future__ import annotations

import logging
import os

from services.paths import TASK_QUEUE_PATH
from services.task_backend import TaskBackend
from services.task_backend_local import LocalFileTaskBackend
from services.task_backend_redis import RedisTaskBackend

LOGGER = logging.getLogger(__name__)


def build_task_backend() -> TaskBackend:
    cloud_mode = os.getenv("EXP_CLOUD_MODE", "0") == "1"
    backend_name = os.getenv("EXP_TASK_BACKEND", "redis").lower()
    if cloud_mode and backend_name != "redis":
        raise RuntimeError("EXP_CLOUD_MODE=1 requires EXP_TASK_BACKEND=redis.")
    if backend_name == "redis":
        redis_url = os.getenv("EXP_REDIS_URL", "redis://127.0.0.1:6379/0")
        queue_key = os.getenv("EXP_REDIS_QUEUE_KEY", "exp:task_queue")
        try:
            return RedisTaskBackend(redis_url, queue_key=queue_key)
        except Exception as exc:  # noqa: BLE001
            fallback_default = "0" if cloud_mode else "1"
            if os.getenv("EXP_ALLOW_LOCAL_BACKEND_FALLBACK", fallback_default) == "1":
                LOGGER.warning("Redis backend unavailable, falling back to local task backend: %s", exc)
                return LocalFileTaskBackend(TASK_QUEUE_PATH)
            raise
    return LocalFileTaskBackend(TASK_QUEUE_PATH)
