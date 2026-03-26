"""
FastAPI entrypoint for the AI Coaching Engine pipeline.

Run from the ``backend`` directory::

    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from llm_service import gemini_is_configured, generate_coaching_advice
from scripts.rag_coach import run as run_rag_synthesizer
from scripts.tactical_rule_engine import run_engine

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_ROOT / ".env")

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Gaffer's Guide — Coaching API",
    version="1.0.0",
    description="Tactical rule engine + RAG + optional LLM coaching advice.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CoachingAdviceItem(BaseModel):
    """Single coaching recommendation for one flaw at one frame."""

    frame_idx: int
    team: str = Field(description="Affected team key: team_0 or team_1")
    flaw: str
    severity: str
    evidence: str
    matched_philosophy_author: str
    fc25_player_roles: list[str] | None = Field(
        default=None,
        description="Recommended EA FC 25 player roles when present in the knowledge base.",
    )
    tactical_instruction: str | None = Field(
        default=None,
        description="Final coaching text (LLM output when enabled).",
    )
    llm_error: str | None = Field(
        default=None,
        description="Populated when the LLM call failed for this item.",
    )


class CoachAdviceResponse(BaseModel):
    """Frontend-ready payload after running the full pipeline."""

    generated_at: str = Field(description="UTC ISO-8601 timestamp.")
    pipeline: dict[str, Any] = Field(
        description="Summary of steps executed (rule engine, RAG, LLM).",
    )
    advice_items: list[CoachingAdviceItem]


def _resolve_llm_credentials() -> tuple[str | None, str, str | None]:
    """Return (api_key, model, base_url) for OpenAI-compatible APIs."""

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    return api_key, model, base_url


async def _complete_coaching_instruction(
    client: AsyncOpenAI | None,
    model: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str | None]:
    """Return (content, error_message)."""

    if client is None:
        return None, None

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.35,
                max_tokens=600,
            )
            choice = response.choices[0].message.content
            return (choice.strip() if choice else None, None)
        except Exception as exc:
            LOGGER.exception("LLM completion failed")
            return None, str(exc)


async def _complete_coaching_instruction_gemini(
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str | None]:
    """Return (content, error_message) using Google Gemini."""

    async with semaphore:
        try:
            text = await asyncio.to_thread(generate_coaching_advice, user_prompt)
            return (text if text else None, None)
        except Exception as exc:
            LOGGER.exception("Gemini completion failed")
            return None, str(exc)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/api/v1/coach/advice",
    response_model=CoachAdviceResponse,
    tags=["coaching"],
)
async def get_coach_advice(
    skip_llm: Annotated[
        bool,
        Query(description="If true, build prompts only and skip remote LLM calls."),
    ] = False,
    llm_concurrency: Annotated[
        int,
        Query(ge=1, le=16, description="Max parallel LLM requests."),
    ] = 4,
) -> CoachAdviceResponse:
    """
    Run the tactical pipeline: metrics → triggers → RAG prompts → optional LLM completions.

    Requires ``backend/output/tactical_metrics.json`` from upstream analytics.
    Set ``GEMINI_API_KEY`` for Google Gemini (preferred), or ``LLM_API_KEY`` /
    ``OPENAI_API_KEY`` for OpenAI-compatible APIs.
    """

    generated_at = datetime.now(timezone.utc).isoformat()
    pipeline: dict[str, Any] = {
        "rule_engine": "pending",
        "rag_synthesizer": "pending",
        "llm": "skipped" if skip_llm else "pending",
    }

    try:
        await asyncio.to_thread(run_engine)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "hint": "Produce tactical_metrics.json (e.g. via your analytics pipeline) under backend/output/.",
            },
        ) from exc

    pipeline["rule_engine"] = "success"

    try:
        records = await asyncio.to_thread(run_rag_synthesizer)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    pipeline["rag_synthesizer"] = "success"

    api_key, model, base_url = _resolve_llm_credentials()
    use_gemini = not skip_llm and gemini_is_configured()
    client: AsyncOpenAI | None = None
    if not skip_llm and not use_gemini and api_key:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncOpenAI(**kwargs)
    elif not skip_llm and not use_gemini and not api_key:
        pipeline["llm"] = "skipped_missing_api_key"
        LOGGER.warning(
            "No GEMINI_API_KEY or LLM_API_KEY; returning advice without tactical_instruction text."
        )

    semaphore = asyncio.Semaphore(llm_concurrency)
    if use_gemini:
        llm_tasks = [
            _complete_coaching_instruction_gemini(rec.llm_prompt, semaphore)
            for rec in records
        ]
    else:
        llm_tasks = [
            _complete_coaching_instruction(client, model, rec.llm_prompt, semaphore)
            for rec in records
        ]
    llm_results = await asyncio.gather(*llm_tasks) if llm_tasks else []

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if skip_llm:
        pipeline["llm"] = "skipped_by_query"
    elif use_gemini:
        pipeline["llm"] = f"ok ({gemini_model})"
    elif client is not None:
        pipeline["llm"] = f"ok ({model})"
    else:
        pipeline.setdefault("llm", pipeline.get("llm", "skipped"))

    advice_items: list[CoachingAdviceItem] = []
    for rec, (instruction, err) in zip(records, llm_results, strict=True):
        advice_items.append(
            CoachingAdviceItem(
                frame_idx=rec.frame_idx,
                team=rec.team,
                flaw=rec.flaw,
                severity=rec.severity,
                evidence=rec.evidence,
                matched_philosophy_author=rec.matched_philosophy_author,
                fc25_player_roles=rec.fc_role_recommendations,
                tactical_instruction=instruction,
                llm_error=err,
            )
        )

    return CoachAdviceResponse(
        generated_at=generated_at,
        pipeline=pipeline,
        advice_items=advice_items,
    )
