from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import main as api_main  # noqa: E402
from main import app  # noqa: E402
from services.beta_job_store import BetaJobRecord  # noqa: E402


def test_beta_job_create_uses_idempotency_key() -> None:
    client = TestClient(app)
    with patch.object(api_main._beta_queue, "enqueue", new_callable=AsyncMock):
        files = {"file": ("x.mp4", b"abc", "video/mp4")}
        data = {
            "cv_engine": "local",
            "llm_engine": "local",
            "idempotency_key": "idem-1",
        }
        first = client.post("/api/v1beta/jobs", files=files, data=data)
        assert first.status_code == 200
        first_job = first.json()["job_id"]

        second = client.post("/api/v1beta/jobs", files=files, data=data)
        assert second.status_code == 200
        assert second.json()["job_id"] == first_job


def test_beta_metrics_endpoint_shape() -> None:
    client = TestClient(app)
    res = client.get("/api/v1beta/metrics")
    assert res.status_code == 200
    body = res.json()
    assert "snapshot" in body
    assert "promotion_gate" in body


def test_beta_job_lookup_endpoint() -> None:
    client = TestClient(app)
    rec = BetaJobRecord(
        job_id="beta_lookup",
        status="pending",
        current_step="Pending",
        cv_engine="local",
        llm_engine="local",
        source_video_path="/tmp/x.mp4",
    )
    api_main._beta_store.create(rec)
    try:
        res = client.get("/api/v1beta/jobs/beta_lookup")
        assert res.status_code == 200
        assert res.json()["job_id"] == "beta_lookup"
    finally:
        api_main._beta_store.delete("beta_lookup")
