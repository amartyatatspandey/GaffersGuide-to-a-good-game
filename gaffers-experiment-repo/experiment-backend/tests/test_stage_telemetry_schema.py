from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402


def test_job_state_contains_stage_telemetry_contract() -> None:
    client = TestClient(app)
    files = {"file": ("x.mp4", b"abc", "video/mp4")}
    data = {
        "cv_engine": "local",
        "llm_engine": "local",
        "decoder_mode": "opencv",
    }
    create = client.post("/api/exp/jobs", files=files, data=data)
    assert create.status_code == 200
    job_id = create.json()["job_id"]
    state = client.get(f"/api/exp/jobs/{job_id}")
    assert state.status_code == 200
    payload = state.json()
    telemetry = payload["telemetry"]
    for key in (
        "queue_wait_ms",
        "decode_ms",
        "infer_ms",
        "post_ms",
        "frames_processed",
        "effective_fps",
        "reid_invocations",
        "reid_ms",
        "id_switch_rate",
    ):
        assert key in telemetry
