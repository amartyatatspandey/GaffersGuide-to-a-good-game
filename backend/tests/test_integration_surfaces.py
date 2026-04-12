"""
Integration surface tests: beta endpoints, chat beta-job resolution,
coach-advice beta-job resolution, beta WS, and beta media routes.
These tests cover the contract fixes in the integration hardening pass.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Stub optional heavy deps to prevent import-time failures in CI.
for _mod in ("google", "google.generativeai"):
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        setattr(_m, "configure", lambda **_: None)
        setattr(_m, "GenerativeModel", lambda *a, **kw: None)
        sys.modules[_mod] = _m

import main as api_main  # noqa: E402
from main import app  # noqa: E402
from services.beta_job_store import BetaJobRecord  # noqa: E402

CLIENT = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_beta_rec(
    job_id: str,
    status: str = "done",
    report_path: str | None = None,
    source_video_path: str | None = None,
) -> BetaJobRecord:
    return BetaJobRecord(
        job_id=job_id,
        status=status,
        current_step="Completed" if status == "done" else "Pending",
        cv_engine="local",
        llm_engine="local",
        source_video_path=source_video_path or "/tmp/x.mp4",
        result_path=report_path,
    )


# ---------------------------------------------------------------------------
# Phase 1: Contract validation — beta GET uses typed model
# ---------------------------------------------------------------------------

def test_beta_job_get_typed_response() -> None:
    rec = _make_beta_rec("typed_test")
    api_main._beta_store.create(rec)
    try:
        res = CLIENT.get("/api/v1beta/jobs/typed_test")
        assert res.status_code == 200
        body = res.json()
        for key in ("job_id", "status", "current_step", "result_path",
                    "tracking_overlay_path", "tracking_data_path", "error"):
            assert key in body, f"Missing key '{key}' in beta job GET response"
    finally:
        api_main._beta_store.delete("typed_test")


def test_beta_artifacts_typed_response() -> None:
    rec = _make_beta_rec("arts_test")
    api_main._beta_store.create(rec)
    try:
        res = CLIENT.get("/api/v1beta/jobs/arts_test/artifacts")
        assert res.status_code == 200
        body = res.json()
        for key in (
            "job_id",
            "status",
            "report_path",
            "tracking_overlay_path",
            "tracking_data_path",
            "report_state",
            "tracking_state",
            "overlay_state",
        ):
            assert key in body, f"Missing key '{key}' in beta artifacts response"
    finally:
        api_main._beta_store.delete("arts_test")


# ---------------------------------------------------------------------------
# Phase 2: Engine selection propagated correctly
# ---------------------------------------------------------------------------

def test_beta_job_create_propagates_local_engine() -> None:
    with patch.object(api_main._beta_queue, "enqueue", new_callable=AsyncMock):
        files = {"file": ("x.mp4", b"abc", "video/mp4")}
        data = {"cv_engine": "local", "llm_engine": "local"}
        res = CLIENT.post("/api/v1beta/jobs", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["cv_engine"] == "local"
    assert body["llm_engine"] == "local"
    api_main._beta_store.delete(body["job_id"])


def test_beta_job_create_propagates_cloud_engine() -> None:
    with patch.object(api_main._beta_queue, "enqueue", new_callable=AsyncMock):
        files = {"file": ("x.mp4", b"abc", "video/mp4")}
        data = {"cv_engine": "cloud", "llm_engine": "cloud"}
        res = CLIENT.post("/api/v1beta/jobs", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["cv_engine"] == "cloud"
    api_main._beta_store.delete(body["job_id"])


# ---------------------------------------------------------------------------
# Phase 3: Job surface — chat resolves beta job IDs
# ---------------------------------------------------------------------------

def test_chat_resolves_beta_job(tmp_path: Path) -> None:
    report = tmp_path / "chat_beta_test_report.json"
    report.write_text(
        json.dumps(
            [
                {
                    "team": "team_0",
                    "flaw": "too deep",
                    "severity": "medium",
                    "evidence": "x",
                }
            ]
        ),
        encoding="utf-8",
    )
    rec = _make_beta_rec("chat_beta_test", status="done", report_path=str(report))
    api_main._beta_store.create(rec)
    try:
        with patch(
            "main.get_tactical_advice", new_callable=AsyncMock, return_value="reply ok"
        ):
            res = CLIENT.post(
                "/api/v1/chat",
                json={"job_id": "chat_beta_test", "message": "explain?"},
            )
        assert res.status_code == 200, res.text
        assert "reply" in res.json()
    finally:
        api_main._beta_store.delete("chat_beta_test")


def test_chat_returns_404_for_unknown_job() -> None:
    res = CLIENT.post("/api/v1/chat", json={"job_id": "ghost_job", "message": "test"})
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Phase 3: coach advice resolves beta job IDs
# ---------------------------------------------------------------------------

def test_coach_advice_resolves_beta_job(tmp_path: Path) -> None:
    report = tmp_path / "coach_beta_test_report.json"
    report.write_text(
        json.dumps([{
            "frame_idx": 100,
            "team": "team_0",
            "flaw": "f1",
            "severity": "high",
            "evidence": "e",
            "matched_philosophy_author": "tiki",
        }]),
        encoding="utf-8",
    )
    rec = _make_beta_rec("coach_beta_test", status="done", report_path=str(report))
    api_main._beta_store.create(rec)
    try:
        res = CLIENT.get("/api/v1/coach/advice?job_id=coach_beta_test&skip_llm=true")
        assert res.status_code == 200, res.text
        body = res.json()
        assert "advice_items" in body
    finally:
        api_main._beta_store.delete("coach_beta_test")


# ---------------------------------------------------------------------------
# Phase 3: Media routes — beta overlay / tracking 404 when no file
# ---------------------------------------------------------------------------

def test_beta_overlay_404_when_no_file() -> None:
    rec = _make_beta_rec("overlay_missing")
    api_main._beta_store.create(rec)
    try:
        res = CLIENT.get("/api/v1beta/jobs/overlay_missing/overlay")
        assert res.status_code in (404, 425)
    finally:
        api_main._beta_store.delete("overlay_missing")


def test_beta_tracking_404_when_no_file() -> None:
    rec = _make_beta_rec("tracking_missing")
    api_main._beta_store.create(rec)
    try:
        res = CLIENT.get("/api/v1beta/jobs/tracking_missing/tracking")
        assert res.status_code in (404, 425)
    finally:
        api_main._beta_store.delete("tracking_missing")


def test_beta_overlay_404_for_unknown_job() -> None:
    res = CLIENT.get("/api/v1beta/jobs/no_such_job/overlay")
    assert res.status_code == 404


def test_beta_source_video_404_for_unknown_job() -> None:
    res = CLIENT.get("/api/v1beta/jobs/no_such_job/source-video")
    assert res.status_code == 404


def test_beta_source_video_404_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing_upload.mp4"
    rec = _make_beta_rec(
        "src_missing_file",
        source_video_path=str(missing),
    )
    api_main._beta_store.create(rec)
    try:
        res = CLIENT.get("/api/v1beta/jobs/src_missing_file/source-video")
        assert res.status_code == 404
    finally:
        api_main._beta_store.delete("src_missing_file")


def test_beta_source_video_200_when_file_exists(tmp_path: Path) -> None:
    vid = tmp_path / "upload.mp4"
    vid.write_bytes(b"fake mp4 bytes for stream test")
    rec = _make_beta_rec("src_ok", source_video_path=str(vid))
    api_main._beta_store.create(rec)
    try:
        res = CLIENT.get("/api/v1beta/jobs/src_ok/source-video")
        assert res.status_code == 200
        assert res.content == b"fake mp4 bytes for stream test"
        assert "video" in (res.headers.get("content-type") or "")
    finally:
        api_main._beta_store.delete("src_ok")


def test_beta_source_video_200_when_overlay_unavailable(tmp_path: Path) -> None:
    """Source route must work even when annotated overlay artifact is absent."""
    vid = tmp_path / "match.mp4"
    vid.write_bytes(b"x")
    rec = _make_beta_rec("src_overlay_absent", status="done", source_video_path=str(vid))
    api_main._beta_store.create(rec)
    try:
        arts = CLIENT.get("/api/v1beta/jobs/src_overlay_absent/artifacts")
        assert arts.status_code == 200
        assert arts.json()["overlay_state"] in ("unavailable", "not_ready")
        src = CLIENT.get("/api/v1beta/jobs/src_overlay_absent/source-video")
        assert src.status_code == 200
    finally:
        api_main._beta_store.delete("src_overlay_absent")


def test_beta_artifacts_overlay_unavailable_when_done_and_missing_file() -> None:
    rec = _make_beta_rec("overlay_state_test", status="done")
    api_main._beta_store.create(rec)
    try:
        res = CLIENT.get("/api/v1beta/jobs/overlay_state_test/artifacts")
        assert res.status_code == 200
        body = res.json()
        assert body["overlay_state"] in ("unavailable", "not_ready")
    finally:
        api_main._beta_store.delete("overlay_state_test")


# ---------------------------------------------------------------------------
# Phase 5: Startup resilience — store survives corrupt payload
# ---------------------------------------------------------------------------

def test_beta_store_survives_corrupt_json(tmp_path: Path) -> None:
    from services.beta_job_store import BetaJobStore

    store_path = tmp_path / "corrupt.json"
    store_path.write_text("{not valid json{{", encoding="utf-8")
    store = BetaJobStore(store_path)
    assert store.get("anything") is None


def test_beta_store_survives_missing_required_field(tmp_path: Path) -> None:
    from services.beta_job_store import BetaJobStore

    store_path = tmp_path / "missing_field.json"
    store_path.write_text(
        json.dumps({"jobs": {"j1": {"job_id": "j1"}}, "idempotency_index": {}}),
        encoding="utf-8",
    )
    store = BetaJobStore(store_path)
    assert store.get("j1") is None


# ---------------------------------------------------------------------------
# Phase 7: Beta WebSocket progress stream
# ---------------------------------------------------------------------------

def test_beta_ws_progress_unknown_job() -> None:
    with CLIENT.websocket_connect("/ws/v1beta/jobs/nonexistent_ws_job") as ws:
        msg = ws.receive_json()
        assert msg["status"] == "error"
        assert msg["error"] == "job_not_found"


def test_beta_ws_progress_done_job() -> None:
    rec = _make_beta_rec("ws_done_test", status="done")
    api_main._beta_store.create(rec)
    try:
        with CLIENT.websocket_connect("/ws/v1beta/jobs/ws_done_test") as ws:
            msg = ws.receive_json()
            assert msg["status"] == "done"
            assert msg["job_id"] == "ws_done_test"
    finally:
        api_main._beta_store.delete("ws_done_test")


def test_local_llm_preflight_endpoint_shape() -> None:
    fake_payload = {
        "configured_base_url": "http://localhost:11434",
        "configured_model": "llama3",
        "daemon_reachable": True,
        "model_present": True,
        "generation_ok": True,
        "error": None,
        "hint": None,
    }
    with patch("main.run_ollama_preflight_check", new_callable=AsyncMock) as mocked:
        mocked.return_value = fake_payload
        res = CLIENT.get("/api/v1/llm/local/preflight")
    assert res.status_code == 200
    body = res.json()
    assert body["daemon_reachable"] is True
    assert body["model_present"] is True
