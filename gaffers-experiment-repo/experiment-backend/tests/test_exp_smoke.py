from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as api_main  # noqa: E402
from main import app  # noqa: E402
from services.job_store import ExperimentJob  # noqa: E402


def test_smoke_outputs_for_completed_job() -> None:
    client = TestClient(app)
    job_id = "exp_smoke_job"
    now = datetime.now(timezone.utc).isoformat()
    output_dir = api_main.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{job_id}_report.json"
    tracking_path = output_dir / f"{job_id}_tracking_data.json"

    report_payload = [
        {
            "frame_idx": 0,
            "team": "team_0",
            "flaw": "Spacing",
            "severity": "low",
            "evidence": "synthetic test",
            "matched_philosophy_author": "test",
            "tactical_instruction": "1. Compact\n2. Scan\n3. Press",
        }
    ]
    tracking_payload = {"telemetry": {"total_frames_processed": 10}, "frames": []}
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    tracking_path.write_text(json.dumps(tracking_payload), encoding="utf-8")

    api_main.job_store.create(
        ExperimentJob(
            job_id=job_id,
            status="done",
            current_step="Completed",
            cv_engine="local",
            llm_engine="local",
            decoder_mode="opencv",
            source_video_path="/tmp/x.mp4",
            result_path=str(report_path),
            tracking_data_path=str(tracking_path),
            created_at=now,
            updated_at=now,
        )
    )

    advice = client.get(f"/api/exp/coach/advice?job_id={job_id}")
    tracking = client.get(f"/api/exp/jobs/{job_id}/tracking")
    reports = client.get("/api/exp/reports")
    chat = client.post("/api/exp/chat", json={"job_id": job_id, "message": "help"})

    assert advice.status_code == 200
    assert tracking.status_code == 200
    assert reports.status_code == 200
    assert chat.status_code == 200
    assert isinstance(chat.json().get("reply"), str)
