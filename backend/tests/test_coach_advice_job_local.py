"""Tests for coach/advice job mode with local LLM refresh."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import main as api_main  # noqa: E402
from main import app  # noqa: E402


def test_job_advice_refreshes_with_ollama_when_local_engine() -> None:
    job_id = "test_local_llm_refresh"
    report_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_report.json"

    card = {
        "frame_idx": 0,
        "team": "team_0",
        "flaw": "Midfield Disconnect",
        "severity": "high",
        "evidence": "Test evidence.",
        "matched_philosophy_author": "Test",
        "fc_role_recommendations": ["False 9"],
        "llm_prompt": "Synthetic prompt for local completion.",
        "tactical_instruction": None,
        "llm_error": (
            "Missing GEMINI_API_KEY or LLM_API_KEY/OPENAI_API_KEY; "
            "skipped cloud completion."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump([card], f)

    client = TestClient(app)
    try:
        with patch.object(
            api_main,
            "ensure_ollama_available",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch.object(
                api_main,
                "get_tactical_advice",
                new_callable=AsyncMock,
                return_value="Ollama filled this insight.",
            ) as mock_advice:
                res = client.get(
                    f"/api/v1/coach/advice?job_id={job_id}&llm_engine=local"
                )
        assert res.status_code == 200
        body = res.json()
        assert body["pipeline"]["llm"] == "ok (local)"
        items = body["advice_items"]
        assert len(items) == 1
        assert items[0]["tactical_instruction"] == "1. Ollama filled this insight."
        assert items[0]["tactical_instruction_steps"] == ["Ollama filled this insight."]
        assert items[0]["llm_error"] is None
        mock_advice.assert_awaited_once()
        call_kw = mock_advice.await_args
        assert call_kw[0][1] == "local"
    finally:
        if report_path.exists():
            report_path.unlink()


def test_job_advice_skips_refresh_when_cloud_engine() -> None:
    job_id = "test_cloud_no_refresh"
    report_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_report.json"

    err_msg = "Missing GEMINI_API_KEY; skipped cloud completion."
    card = {
        "frame_idx": 0,
        "team": "team_0",
        "flaw": "Midfield Disconnect",
        "severity": "high",
        "evidence": "Test evidence.",
        "matched_philosophy_author": "Test",
        "llm_prompt": "Prompt.",
        "tactical_instruction": None,
        "llm_error": err_msg,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump([card], f)

    client = TestClient(app)
    try:
        with patch.object(
            api_main,
            "ensure_ollama_available",
            new_callable=AsyncMock,
        ) as mock_ensure:
            with patch.object(
                api_main,
                "get_tactical_advice",
                new_callable=AsyncMock,
            ) as mock_advice:
                res = client.get(
                    f"/api/v1/coach/advice?job_id={job_id}&llm_engine=cloud"
                )
        assert res.status_code == 200
        body = res.json()
        assert body["pipeline"]["llm"] == "skipped"
        assert body["advice_items"][0]["tactical_instruction_steps"] == []
        assert body["advice_items"][0]["llm_error"] == err_msg
        mock_ensure.assert_not_called()
        mock_advice.assert_not_called()
    finally:
        if report_path.exists():
            report_path.unlink()
