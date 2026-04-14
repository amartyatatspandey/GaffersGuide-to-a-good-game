from __future__ import annotations

import sys
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as api_main  # noqa: E402
from main import app  # noqa: E402


def test_create_job_with_idempotency() -> None:
    client = TestClient(app)
    files = {"file": ("x.mp4", b"abc", "video/mp4")}
    data = {
        "cv_engine": "local",
        "llm_engine": "local",
        "decoder_mode": "pyav",
        "runtime_target": "apple_mps",
        "hardware_profile": "mps",
        "quality_mode": "fast",
        "chunking_policy": "auto",
        "max_parallel_chunks": "3",
        "target_sla_tier": "tier_5m",
        "idempotency_key": f"exp-idem-{uuid.uuid4().hex}",
    }
    first = client.post("/api/exp/jobs", files=files, data=data)
    assert first.status_code == 200
    second = client.post("/api/exp/jobs", files=files, data=data)
    assert second.status_code == 200
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["decoder_mode"] == "pyav"
    assert second.json()["runtime_target"] == "apple_mps"


def test_rejects_v1_routes() -> None:
    client = TestClient(app)
    assert client.get("/api/v1/reports").status_code == 404
    assert client.post("/api/v1/jobs").status_code == 404


def test_metrics_and_health_routes_exist() -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/exp/metrics").status_code == 200
