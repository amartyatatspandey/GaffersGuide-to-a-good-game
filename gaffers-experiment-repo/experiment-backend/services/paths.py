from __future__ import annotations

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BACKEND_ROOT / "output" / "exp"
UPLOAD_ROOT = BACKEND_ROOT / "data" / "uploads"
TASK_QUEUE_PATH = OUTPUT_ROOT / "exp_task_queue.json"
STORE_PATH = OUTPUT_ROOT / "exp_jobs_store.json"
SN_CALIBRATION_RESOURCES_DIR = BACKEND_ROOT / "resources" / "sn-calibration"
BENCHMARK_TMP_DIR = BACKEND_ROOT / "output" / "bench_tmp"
