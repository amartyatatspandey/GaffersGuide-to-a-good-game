from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Avoid hard dependency on google-generativeai during API unit tests.
if "google" not in sys.modules:
    sys.modules["google"] = types.ModuleType("google")
if "google.generativeai" not in sys.modules:
    mod = types.ModuleType("google.generativeai")
    setattr(mod, "configure", lambda **_: None)
    setattr(mod, "GenerativeModel", lambda *_args, **_kwargs: None)
    sys.modules["google.generativeai"] = mod

import main as api_main  # noqa: E402
from main import JobRecord, app  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_job_artifact_endpoints() -> None:
    client = TestClient(app)
    job_id = "job_artifacts_test"
    report_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_report.json"
    tracking_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_tracking_data.json"
    overlay_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_tracking_overlay.mp4"

    _write_json(report_path, [{"frame_idx": 1, "team": "team_0", "flaw": "test"}])
    _write_json(
        tracking_path,
        {
            "video_path": "/tmp/video.mp4",
            "frames": [
                {
                    "frame_idx": 1,
                    "players": [],
                    "ball_xy": None,
                    "used_optical_flow_fallback": False,
                    "camera_shift_xy": [0.0, 0.0],
                }
            ],
        },
    )
    overlay_path.write_bytes(b"\x00\x00\x00\x18ftypisom")

    with api_main._job_store_lock:
        api_main._job_store[job_id] = JobRecord(
            job_id=job_id,
            status="done",
            current_step="Completed",
            cv_engine="local",
            llm_engine="cloud",
            result_path=str(report_path),
            tracking_overlay_path=str(overlay_path),
            tracking_data_path=str(tracking_path),
        )

    try:
        art = client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert art.status_code == 200
        art_body = art.json()
        assert art_body["report_path"].endswith(f"{job_id}_report.json")
        assert art_body["tracking_overlay_path"].endswith(
            f"{job_id}_tracking_overlay.mp4"
        )
        assert art_body["tracking_data_path"].endswith(f"{job_id}_tracking_data.json")

        tr = client.get(f"/api/v1/jobs/{job_id}/tracking")
        assert tr.status_code == 200
        tr_body = tr.json()
        assert isinstance(tr_body.get("frames"), list)
        assert tr_body["frames"][0]["frame_idx"] == 1
        assert "used_optical_flow_fallback" in tr_body["frames"][0]
        assert "camera_shift_xy" in tr_body["frames"][0]

        ov = client.get(f"/api/v1/jobs/{job_id}/overlay")
        assert ov.status_code == 200
        assert ov.headers.get("content-type", "").startswith("video/mp4")
    finally:
        with api_main._job_store_lock:
            api_main._job_store.pop(job_id, None)
        for p in (report_path, tracking_path, overlay_path):
            if p.exists():
                p.unlink()
