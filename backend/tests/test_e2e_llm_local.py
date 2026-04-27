"""Tests for job-time local LLM completion helper (Ollama path)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from scripts.rag_coach import GeneratedPromptRecord  # noqa: E402
from scripts import e2e_llm_local  # noqa: E402
from services.errors import EngineRoutingError  # noqa: E402


def _sample_record(*, llm_prompt: str) -> GeneratedPromptRecord:
    return GeneratedPromptRecord(
        frame_idx=0,
        team="team_0",
        flaw="Midfield Disconnect",
        severity="high",
        evidence="Synthetic.",
        frequency_pct=12.5,
        matched_philosophy_author="Test",
        matched_quote_excerpt="Quote.",
        fc_role_recommendations=["CM"],
        llm_prompt=llm_prompt,
    )


def test_run_llm_local_maps_completions() -> None:
    records = [_sample_record(llm_prompt="p1"), _sample_record(llm_prompt="p2")]

    async def _run():
        with patch.object(
            e2e_llm_local,
            "get_tactical_advice",
            new_callable=AsyncMock,
            side_effect=["Insight one", "Insight two"],
        ) as mock_advice:
            out = await e2e_llm_local.run_llm_local(records, concurrency=2)
            assert mock_advice.await_count == 2
            assert mock_advice.await_args_list[0][0][1] == "local"
            return out

    out = asyncio.run(_run())

    assert len(out) == 2
    assert out[0]["tactical_instruction"] == "Insight one"
    assert out[0]["llm_error"] is None
    assert out[1]["tactical_instruction"] == "Insight two"


def test_run_llm_local_sync_wrapper() -> None:
    """Asyncio entry point used by run_e2e_cloud."""
    records = [_sample_record(llm_prompt="only")]
    with patch.object(
        e2e_llm_local,
        "get_tactical_advice",
        new_callable=AsyncMock,
        return_value="Done",
    ):
        out = asyncio.run(e2e_llm_local.run_llm_local(records, concurrency=1))
    assert out[0]["tactical_instruction"] == "Done"


def test_run_llm_local_empty_prompt_skips_llm() -> None:
    records = [_sample_record(llm_prompt="   ")]
    with patch.object(e2e_llm_local, "get_tactical_advice", new_callable=AsyncMock) as mock_advice:
        out = asyncio.run(e2e_llm_local.run_llm_local(records))
    mock_advice.assert_not_called()
    assert out[0]["tactical_instruction"] is None
    assert out[0]["llm_error"] == "empty_llm_prompt"


def test_run_llm_local_engine_routing_error_on_card() -> None:
    records = [_sample_record(llm_prompt="x")]
    err = EngineRoutingError(
        status_code=424,
        code="OLLAMA_UNAVAILABLE",
        message="Ollama not reachable",
    )
    with patch.object(
        e2e_llm_local,
        "get_tactical_advice",
        new_callable=AsyncMock,
        side_effect=err,
    ):
        out = asyncio.run(e2e_llm_local.run_llm_local(records))
    assert out[0]["tactical_instruction"] is None
    assert "OLLAMA_UNAVAILABLE" in (out[0]["llm_error"] or "")
