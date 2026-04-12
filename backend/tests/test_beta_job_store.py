from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services.beta_job_store import BetaJobStore  # noqa: E402


def test_beta_job_store_load_drops_unknown_keys(tmp_path: Path) -> None:
    store_path = tmp_path / "beta_jobs_store.json"
    store_path.write_text(
        json.dumps(
            {
                "jobs": {
                    "j1": {
                        "job_id": "j1",
                        "status": "done",
                        "current_step": "Completed",
                        "cv_engine": "local",
                        "llm_engine": "local",
                        "source_video_path": "/tmp/video.mp4",
                        "decoder_mode": "pyav",
                        "legacy_field": 1,
                    }
                },
                "idempotency_index": {},
            }
        ),
        encoding="utf-8",
    )
    store = BetaJobStore(store_path)
    rec = store.get("j1")
    assert rec is not None
    assert rec.job_id == "j1"
    assert rec.status == "done"
    assert rec.cv_engine == "local"
