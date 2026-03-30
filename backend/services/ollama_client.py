from __future__ import annotations

import os

import httpx

from services.errors import EngineRoutingError

SYSTEM_PROMPT = (
    "You are an elite UEFA Pro License tactical analyst. Analyze the provided "
    "player tracking data (distance, speed, positioning). Be concise, brutally "
    "analytical, and focus on physical load and spatial structure. Do not use "
    "generic cliches."
)


def _base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3")


def _timeout_seconds() -> float:
    return float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))


def _offline_error(*, hint: str | None = None) -> EngineRoutingError:
    return EngineRoutingError(
        status_code=424,
        code="OLLAMA_UNAVAILABLE",
        message="OLLAMA_UNAVAILABLE: Local LLM engine is offline. Please start Ollama.",
        hint=hint,
    )


async def ensure_ollama_available() -> None:
    """Preflight-check that local Ollama daemon is reachable."""
    base_url = _base_url()
    timeout_s = _timeout_seconds()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            res = await client.get(f"{base_url}/api/tags")
        if res.status_code >= 400:
            raise _offline_error(
                hint="Verify Ollama is running and listening on OLLAMA_BASE_URL.",
            )
    except httpx.ConnectError as exc:
        raise _offline_error(
            hint="Run `ollama serve` and ensure `llama3` is pulled locally.",
        ) from exc


async def generate_local_advice(tracking_data_csv: str) -> str:
    """
    Generate tactical advice from local Ollama using a fixed analyst persona.
    """
    base_url = _base_url()
    model = _model_name()
    timeout_s = _timeout_seconds()
    combined_text = f"{SYSTEM_PROMPT}\n\nTracking Data:\n{tracking_data_csv}"

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            res = await client.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": combined_text, "stream": False},
            )
    except httpx.ConnectError as exc:
        raise _offline_error(
            hint="Run `ollama serve` and ensure `llama3` is available.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise EngineRoutingError(
            status_code=503,
            code="LOCAL_LLM_UPSTREAM_ERROR",
            message=str(exc),
        ) from exc

    if res.status_code >= 400:
        raise EngineRoutingError(
            status_code=503,
            code="LOCAL_LLM_UPSTREAM_ERROR",
            message=f"Ollama error {res.status_code}: {res.text}",
        )

    data = res.json()
    text = str(data.get("response", "")).strip()
    if not text:
        raise EngineRoutingError(
            status_code=503,
            code="LOCAL_LLM_EMPTY_RESPONSE",
            message="Ollama returned an empty response.",
        )
    return text

