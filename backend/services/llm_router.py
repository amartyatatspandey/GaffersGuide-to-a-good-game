from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

from llm_service import gemini_is_configured, generate_coaching_advice
from openai import AsyncOpenAI

from services.errors import EngineRoutingError
from services.ollama_client import (
    ensure_ollama_available,
    generate_local_advice,
    start_ollama_for_app_lifecycle,
    stop_ollama_for_app_lifecycle,
)

LLMEngine = Literal["local", "cloud"]


async def _generate_cloud(prompt: str) -> str:
    """Generate coaching text using Gemini or OpenAI-compatible cloud APIs."""
    if gemini_is_configured():
        return await asyncio.to_thread(generate_coaching_advice, prompt)

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EngineRoutingError(
            status_code=424,
            code="CLOUD_LLM_NOT_CONFIGURED",
            message="Cloud LLM selected but no API key is configured.",
            hint="Set GEMINI_API_KEY or LLM_API_KEY/OPENAI_API_KEY.",
        )

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = AsyncOpenAI(**kwargs)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=600,
        )
        out = response.choices[0].message.content
        if not out:
            raise EngineRoutingError(
                status_code=503,
                code="CLOUD_LLM_EMPTY_RESPONSE",
                message="Cloud LLM returned an empty response.",
            )
        return out.strip()
    except EngineRoutingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EngineRoutingError(
            status_code=503,
            code="CLOUD_LLM_UPSTREAM_ERROR",
            message=str(exc),
        ) from exc


async def _generate_local(prompt: str) -> str:
    """Generate coaching text using local Ollama daemon."""
    return await generate_local_advice(prompt)


async def get_tactical_advice(tracking_data: str, llm_engine: str = "cloud") -> str:
    """Factory-style LLM routing for tactical advice generation."""
    if llm_engine == "cloud":
        return await _generate_cloud(tracking_data)
    if llm_engine == "local":
        return await _generate_local(tracking_data)
    raise EngineRoutingError(
        status_code=409,
        code="ENGINE_MODE_UNSUPPORTED",
        message=f"Unsupported llm_engine value: {llm_engine}",
    )


async def generate_coaching_text(prompt: str, llm_engine: LLMEngine) -> str:
    """Backward-compatible alias for existing callers."""
    return await get_tactical_advice(prompt, llm_engine=llm_engine)


__all__ = [
    "LLMEngine",
    "get_tactical_advice",
    "generate_coaching_text",
    "ensure_ollama_available",
    "start_ollama_for_app_lifecycle",
    "stop_ollama_for_app_lifecycle",
]
