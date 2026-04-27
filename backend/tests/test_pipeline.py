from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"

# Allow `from main import app` and `from scripts.* import ...` imports.
sys.path.insert(0, str(BACKEND_DIR))

from main import app as fastapi_app  # noqa: E402
from scripts.rag_coach import run as run_rag_synthesizer  # noqa: E402
from scripts.tactical_rule_engine import RuleEngine, run_engine  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_massive_gap_meters(evidence: str) -> float | None:
    """Extract the numeric value from:
    "Team spent {x}% of the time with this flaw, averaging a {y}m gap."
    """
    match = re.search(
        r"averaging a\s+(-?[0-9]+(?:\.[0-9]+)?)m\s+gap\.",
        evidence,
    )
    if not match:
        return None
    return float(match.group(1))


def test_analytics_math() -> None:
    tactical_metrics_path = BACKEND_DIR / "output" / "tactical_metrics.json"
    assert tactical_metrics_path.is_file(), f"Missing {tactical_metrics_path}"

    timeline = _load_json(tactical_metrics_path)
    assert isinstance(timeline, list), "tactical_metrics.json must be a list of frames"
    assert timeline, "tactical_metrics.json must contain at least one frame"

    for frame in timeline:
        assert isinstance(frame, dict), "Each frame must be a dict"
        for team_key in ("team_0", "team_1"):
            team_metrics = frame.get(team_key)
            assert isinstance(team_metrics, dict), f"{team_key} must be a dict"

            team_length_m = team_metrics.get("team_length_m")
            assert isinstance(team_length_m, (int, float))
            assert 20.0 <= float(team_length_m) <= 60.0

            line_gap_def_mid_m = team_metrics.get("line_gap_def_mid_m")
            assert isinstance(line_gap_def_mid_m, (int, float))
            assert 5.0 < float(line_gap_def_mid_m) < 40.0

            pitch_control_pct = team_metrics.get("pitch_control_pct")
            assert pitch_control_pct is not None
            assert isinstance(pitch_control_pct, (int, float))
            assert 0.0 <= float(pitch_control_pct) <= 100.0


def test_rule_engine_logic() -> None:
    engine = RuleEngine()
    triggers = run_engine(write_output=True)
    assert isinstance(triggers, list)

    assert triggers, "Expected at least one chunk-level tactical insight"

    midfield_disconnect_insights: list[dict[str, Any]] = [
        t for t in triggers if t.get("flaw") == "Midfield Disconnect"
    ]
    assert midfield_disconnect_insights, (
        "Expected at least one 'Midfield Disconnect' chunk-level insight"
    )

    for insight in midfield_disconnect_insights:
        evidence = insight.get("evidence")
        assert isinstance(evidence, str) and evidence.strip(), (
            "Evidence must be a non-empty string"
        )

        extracted = _parse_massive_gap_meters(evidence)
        assert extracted is not None, (
            f"Could not parse evidence meters from: {evidence!r}"
        )
        assert extracted > engine.MAX_LINE_GAP, (
            f"Expected evidence gap ({extracted:.1f}m) to exceed MAX_LINE_GAP "
            f"({engine.MAX_LINE_GAP:.1f}m)"
        )


def test_rag_synthesis() -> None:
    # Ensure `backend/output/tactical_triggers.json` matches the refactored rule engine.
    run_engine(write_output=True)
    records = run_rag_synthesizer()
    assert isinstance(records, list)
    assert records, "Expected RAG synthesizer to output at least one prompt record"

    targeted_flaw = "Midfield Disconnect"
    targeted = [r for r in records if getattr(r, "flaw", None) == targeted_flaw]
    assert targeted, f"Expected at least one RAG prompt record for flaw={targeted_flaw}"

    for rec in targeted:
        prompt = rec.llm_prompt
        assert targeted_flaw in prompt
        assert "Author:" in prompt
        assert ("False 9" in prompt) or ("Inverted Fullback" in prompt)


def test_fastapi_dry_run() -> None:
    client = TestClient(fastapi_app)
    response = client.get("/api/v1/coach/advice?skip_llm=true")

    assert response.status_code == 200
    payload = response.json()

    assert payload["pipeline"]["rule_engine"] == "success"

    advice_items = payload.get("advice_items")
    assert isinstance(advice_items, list) and advice_items, (
        "Expected non-empty advice_items"
    )

    for item in advice_items:
        assert item.get("tactical_instruction") is None
        assert isinstance(item.get("frame_idx"), int)
        assert isinstance(item.get("team"), str) and item["team"]
        assert isinstance(item.get("flaw"), str) and item["flaw"]
        assert isinstance(item.get("evidence"), str) and item["evidence"]
        assert (
            isinstance(item.get("matched_philosophy_author"), str)
            and item["matched_philosophy_author"]
        )
