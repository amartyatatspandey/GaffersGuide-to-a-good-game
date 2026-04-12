"""Local Ollama completions for the CV→…→RAG E2E pipeline (job-time LLM)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from scripts.rag_coach import GeneratedPromptRecord
from services.errors import EngineRoutingError
from services.llm_router import get_tactical_advice


async def run_llm_local(
    records: list[GeneratedPromptRecord],
    *,
    concurrency: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run Ollama completions for each synthesized prompt (mirrors cloud ``run_llm`` shape).

    Caller must run ``ensure_ollama_available()`` before this (fail fast after CV / RAG).
    """
    limit = concurrency if concurrency is not None else int(os.getenv("OLLAMA_JOB_LLM_CONCURRENCY", "4"))
    limit = max(1, min(limit, 16))
    semaphore = asyncio.Semaphore(limit)

    async def _one(record: GeneratedPromptRecord) -> dict[str, Any]:
        payload = record.model_dump()
        prompt = record.llm_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            payload["tactical_instruction"] = None
            payload["llm_error"] = "empty_llm_prompt"
            return payload
        async with semaphore:
            try:
                text = await get_tactical_advice(prompt, "local")
                payload["tactical_instruction"] = text
                payload["llm_error"] = None
                return payload
            except EngineRoutingError as exc:
                payload["tactical_instruction"] = None
                payload["llm_error"] = f"{exc.code}: {exc.message}"
                return payload
            except Exception as exc:  # noqa: BLE001
                payload["tactical_instruction"] = None
                payload["llm_error"] = str(exc)
                return payload

    if not records:
        return []
    return list(await asyncio.gather(*[_one(r) for r in records]))
