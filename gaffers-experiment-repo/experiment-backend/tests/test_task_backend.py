from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.task_backend import TaskPayload
from services.task_backend_local import LocalFileTaskBackend


def test_local_file_task_backend_enqueue_dequeue(tmp_path: Path) -> None:
    backend = LocalFileTaskBackend(tmp_path / "queue.json")
    payload = TaskPayload(
        job_id="job1",
        video_path=tmp_path / "x.mp4",
        cv_engine="cloud",
        llm_engine="local",
        decoder_mode="opencv",
        runtime_target="nvidia",
        hardware_profile="l4",
        quality_mode="balanced",
        chunking_policy="fixed",
        max_parallel_chunks=2,
        target_sla_tier="tier_10m",
        enqueued_at_epoch_ms=10.0,
    )
    backend.enqueue(payload)
    out = backend.dequeue()
    assert out is not None
    assert out.job_id == "job1"
    assert out.runtime_target == "nvidia"
    assert out.max_parallel_chunks == 2
    assert backend.dequeue() is None
