import json
import os
import pytest
from unittest.mock import AsyncMock, patch

from main import app, BACKEND_ROOT, _job_store, JobRecord

# Clear startup/shutdown event handlers to prevent background queue or Ollama daemon startup during tests
app.router.on_startup.clear()
app.router.on_shutdown.clear()

from fastapi.testclient import TestClient
client = TestClient(app)

# Use API key from environment if defined, otherwise the default one
API_KEY = os.getenv("API_KEY", "gG_83a9f4c7b2d5e11496a80275819d4b3f")
HEADERS = {"x-api-key": API_KEY}

def test_get_job_report_enriched():
    job_id = "test_job_enrich_endpoint"
    output_dir = BACKEND_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock job store record
    rec = JobRecord(
        job_id=job_id,
        status="done",
        current_step="Completed",
        cv_engine="cloud",
        llm_engine="cloud"
    )
    from main import _job_store_lock
    with _job_store_lock:
        _job_store[job_id] = rec
        
    # Write mock original report and events
    report_cards = [
        {
            "frame_idx": 100,
            "team": "team_0",
            "flaw": "Suicidal High Line",
            "severity": "High",
            "frequency_pct": 10.0,
            "evidence": "High line at 35m",
            "matched_philosophy_author": "Arrigo Sacchi",
            "matched_quote_excerpt": "...",
            "fc_role_recommendations": [],
            "tactical_instruction": "Drop line.",
            "llm_error": None
        }
    ]
    with open(output_dir / f"{job_id}_report.json", "w") as f:
        json.dump(report_cards, f)
        
    # Mock event index
    from event_layer.models import EventIndex, EventRecord
    events = [
        EventRecord(
            event_type="THR_001",
            event_name="Dangerous Run",
            category="threat",
            team_id="team_0",
            player_id=3,
            start_frame=100,
            end_frame=200,
            start_time_s=4.0,
            end_time_s=8.0,
            duration_s=4.0,
            confidence=0.9,
            importance=0.8,
            clip_start_frame=50,
            clip_end_frame=250
        )
    ]
    index = EventIndex(
        job_id=job_id,
        total_frames=1000,
        fps=25.0,
        events=events,
        generated_at="2026-06-09T00:00:00Z"
    )
    with open(output_dir / f"{job_id}_events.json", "w") as f:
        json.dump(index.model_dump(), f)
        
    # Request enriched report
    response = client.get(f"/api/v1/jobs/{job_id}/report/enriched", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["flaw"] == "Suicidal High Line"
    assert "evidence_clips" in data[0]

def test_chat_evidence_endpoint(monkeypatch):
    # Mock detect_intent and get_tactical_advice
    async def mock_detect_intent(prompt: str):
        return "evidence_request"
        
    async def mock_get_tactical_advice(prompt: str, engine: str):
        return "Here is the evidence you requested."
        
    monkeypatch.setattr("scripts.llm_router.detect_intent", mock_detect_intent)
    monkeypatch.setattr("main.get_tactical_advice", mock_get_tactical_advice)
    
    # We use the same job_id created in the first test
    job_id = "test_job_enrich_endpoint"
    
    chat_payload = {
        "message": "Show me player 3's dangerous runs",
        "job_id": job_id
    }
    
    response = client.post("/api/v1/chat", json=chat_payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["reply"] == "Here is the evidence you requested."
    assert "evidence" in data
    assert data["evidence"] is not None
    assert len(data["evidence"]["clips"]) > 0
    assert data["evidence"]["clips"][0]["highlight_player_ids"] == [3]
