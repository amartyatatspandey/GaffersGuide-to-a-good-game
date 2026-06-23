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
from main import app  # noqa: E402
from services.pdf_service import PDFService  # noqa: E402


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_pdf_generation_flow() -> None:
    client = TestClient(app)
    job_id = "pdf_test_job_id"
    
    # Paths
    report_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_report.json"
    tracking_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_tracking_data.json"
    events_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_events.json"
    threats_path = api_main.BACKEND_ROOT / "output" / f"{job_id}_threat_profiles.json"

    # Dummy reports
    report_payload = {
        "advice_items": [
            {
                "flaw": "Match Summary",
                "severity": "Info",
                "team": "team_0",
                "evidence": "Tactical Power: Red 75.2 vs Blue 68.4 | Win Probability: Red 58.2% | Blue 41.8% | Compactness: Red 65.0 / Blue 62.0 | Transition Speed: Red 4.2 / Blue 3.8",
                "matched_philosophy_author": "System",
                "summary_data": {
                    "team_0": {
                        "tactical_power": 75.2,
                        "compactness": 65.0,
                        "compactness_vertical": 68.0,
                        "compactness_horizontal": 62.0,
                        "compactness_midfield": 64.0,
                        "control": 58.2,
                        "intensity": 70.0,
                        "defensive_shape": 69.0,
                        "transition_speed": 4.2,
                        "win_prob": 58.2
                    },
                    "team_1": {
                        "tactical_power": 68.4,
                        "compactness": 62.0,
                        "compactness_vertical": 60.0,
                        "compactness_horizontal": 64.0,
                        "compactness_midfield": 61.0,
                        "control": 41.8,
                        "intensity": 65.0,
                        "defensive_shape": 62.0,
                        "transition_speed": 3.8,
                        "win_prob": 41.8
                    }
                }
            },
            {
                "flaw": "Low Defensive Block",
                "severity": "Critical",
                "team": "team_0",
                "evidence": "Red team defense dropped below 20 meters.",
                "tactical_instruction": "Shift defensive line forward by 10 meters.",
                "matched_philosophy_author": "Arrigo Sacchi"
            }
        ],
        "quality_profile": "balanced",
        "metadata": {
            "chunking_interval": "15-minute intervals"
        }
    }

    tracking_payload = {
        "frames": [
            {
                "frame_idx": 0,
                "players": [
                    {"player_id": 1, "team": "team_0", "radar_pt": [-10.0, 5.0], "speed_kmh": 25.5},
                    {"player_id": 2, "team": "team_1", "radar_pt": [12.0, -8.0], "speed_kmh": 12.0}
                ]
            },
            {
                "frame_idx": 1,
                "players": [
                    {"player_id": 1, "team": "team_0", "radar_pt": [-9.5, 5.2], "speed_kmh": 26.2},
                    {"player_id": 2, "team": "team_1", "radar_pt": [11.8, -7.5], "speed_kmh": 14.5}
                ]
            }
        ]
    }

    events_payload = {
        "events": [
            {
                "event_name": "Sprint",
                "player_id": 1,
                "start_time_s": 15.5,
                "pitch_zone": "defensive_third",
                "importance": 0.9,
                "description": "Player 1 made a sprint covering 15 meters."
            }
        ]
    }

    threats_payload = [
        {"player_id": 1, "threat_score": 85.0},
        {"player_id": 2, "threat_score": 45.0}
    ]

    try:
        _write_json(report_path, report_payload)
        _write_json(tracking_path, tracking_payload)
        _write_json(events_path, events_payload)
        _write_json(threats_path, threats_payload)

        import os
        headers = {}
        api_key = os.getenv("API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        # Call endpoint via test client
        resp = client.get(f"/api/v1/jobs/{job_id}/report/pdf", headers=headers)
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "application/pdf"
        assert resp.content.startswith(b"%PDF")
    finally:
        for p in (report_path, tracking_path, events_path, threats_path):
            if p.exists():
                p.unlink()
