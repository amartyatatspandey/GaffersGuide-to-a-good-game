from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Final

import httpx

from services.errors import EngineRoutingError

logger = logging.getLogger(__name__)

OLLAMA_AUTO_START_ENV: Final = "OLLAMA_AUTO_START"
OLLAMA_AUTO_START_IN_CLOUD_ENV: Final = "OLLAMA_AUTO_START_IN_CLOUD"
OLLAMA_MANAGED_LIFECYCLE_ENV: Final = "OLLAMA_MANAGED_LIFECYCLE"
OLLAMA_PULL_ON_START_ENV: Final = "OLLAMA_PULL_ON_START"

# Populated only when this process spawned ``ollama serve`` for managed lifecycle (see shutdown).
_lifecycle_popen: subprocess.Popen | None = None

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


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _ollama_executable() -> str | None:
    """Return path to ``ollama`` CLI if on PATH, else None."""
    return shutil.which("ollama")


def _cloud_run_guard_allows_fork() -> bool:
    """Forking a local daemon is blocked on Cloud Run unless explicitly allowed."""
    if not os.getenv("K_SERVICE", "").strip():
        return True
    return _env_truthy(OLLAMA_AUTO_START_IN_CLOUD_ENV)


def _should_attempt_auto_start() -> bool:
    """
    Auto-spawn ``ollama serve`` when the daemon is down and forking is allowed.

    - ``OLLAMA_AUTO_START=0|false|no``: never auto-spawn from request preflight.
    - ``OLLAMA_AUTO_START=1|true|yes``: auto-spawn when allowed (see Cloud Run below).
    - **Unset (default):** auto-spawn on non-Cloud Run hosts only (fixes empty-env laptops
      where users expect Ollama to come up without extra configuration).

    On Cloud Run (``K_SERVICE``), unset defaults to **off** unless ``OLLAMA_AUTO_START_IN_CLOUD=1``.
    """
    raw = os.getenv(OLLAMA_AUTO_START_ENV, "").strip().lower()
    allow_fork = _cloud_run_guard_allows_fork()
    if raw in ("0", "false", "no"):
        out = False
    elif raw in ("1", "true", "yes"):
        out = allow_fork
    else:
        # unset: default on outside Cloud Run (see debug session bb63ae: empty env skipped spawn).
        out = allow_fork
    return out


def _should_manage_lifecycle() -> bool:
    """
    Start ``ollama serve`` on API startup and stop it on shutdown (this process only).

    Enabled when either ``OLLAMA_MANAGED_LIFECYCLE=1`` or ``GAFFERS_DEFAULT_LLM_ENGINE=local``
    (offline/local LLM as default). Same Cloud Run guard as auto-start: set
    ``OLLAMA_AUTO_START_IN_CLOUD=1`` if you truly need this in K_SERVICE.
    """
    if _env_truthy(OLLAMA_MANAGED_LIFECYCLE_ENV):
        return _cloud_run_guard_allows_fork()
    if os.getenv("GAFFERS_DEFAULT_LLM_ENGINE", "").strip().lower() == "local":
        return _cloud_run_guard_allows_fork()
    return False


def _popen_ollama_serve(exe: str) -> subprocess.Popen:
    return subprocess.Popen(  # noqa: S603 — argv list, exe from shutil.which
        [exe, "serve"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _spawn_ollama_serve() -> None:
    """Start ``ollama serve`` detached (request-time auto-start). Caller must verify the daemon comes up."""
    exe = _ollama_executable()
    if not exe:
        return
    try:
        _popen_ollama_serve(exe)
        logger.info("Spawned `ollama serve` (OLLAMA_AUTO_START); waiting for /api/tags …")
    except OSError as exc:
        logger.warning("Could not spawn ollama serve: %s", exc)


def _terminate_managed_ollama_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error stopping managed Ollama process: %s", exc)


async def start_ollama_for_app_lifecycle() -> None:
    """
    If ``OLLAMA_MANAGED_LIFECYCLE=1``, ensure Ollama is up when the API boots.

    If ``/api/tags`` already works, does nothing (does not take ownership, so shutdown
    will not kill a daemon you started yourself). If this function spawns ``ollama serve``,
    :func:`stop_ollama_for_app_lifecycle` will terminate that child when the app exits.

    Does not pull models unless ``OLLAMA_PULL_ON_START=1`` (can be slow).
    """
    global _lifecycle_popen

    if not _should_manage_lifecycle():
        return

    base_url = _base_url()
    timeout_s = _timeout_seconds()
    try:
        res = await _probe_tags(base_url, timeout=timeout_s)
        if res.status_code < 400:
            logger.info(
                "Ollama already reachable at %s; managed lifecycle did not spawn a new daemon.",
                base_url,
            )
            return
    except httpx.ConnectError:
        pass

    exe = _ollama_executable()
    if not exe:
        logger.warning(
            "%s is set but `ollama` is not on PATH; cannot start local LLM with the app.",
            OLLAMA_MANAGED_LIFECYCLE_ENV,
        )
        return

    try:
        _lifecycle_popen = _popen_ollama_serve(exe)
        logger.info(
            "Spawned `ollama serve` for managed lifecycle (pid=%s); waiting for /api/tags …",
            _lifecycle_popen.pid,
        )
    except OSError as exc:
        logger.warning("Could not spawn ollama serve for lifecycle: %s", exc)
        _lifecycle_popen = None
        return

    try:
        await _wait_for_ollama_after_spawn(base_url)
    except EngineRoutingError:
        _terminate_managed_ollama_process(_lifecycle_popen)
        _lifecycle_popen = None
        logger.error(
            "Managed Ollama start failed after spawn; local LLM will stay offline until fixed."
        )
        return

    if _env_truthy(OLLAMA_PULL_ON_START_ENV):
        model = _model_name()
        loop = asyncio.get_running_loop()

        def _pull() -> None:
            try:
                subprocess.run(  # noqa: S603
                    [exe, "pull", model],
                    check=False,
                    timeout=900,
                    capture_output=True,
                )
                logger.info("Finished `ollama pull %s` (OLLAMA_PULL_ON_START).", model)
            except subprocess.TimeoutExpired:
                logger.warning("ollama pull %s timed out; run manually if needed.", model)
            except OSError as exc:
                logger.warning("ollama pull %s failed: %s", model, exc)

        await loop.run_in_executor(None, _pull)


def stop_ollama_for_app_lifecycle() -> None:
    """Stop Ollama only if this process started it via :func:`start_ollama_for_app_lifecycle`."""
    global _lifecycle_popen

    proc = _lifecycle_popen
    _lifecycle_popen = None
    if proc is None:
        return
    logger.info("Stopping managed Ollama (pid=%s) on app shutdown.", proc.pid)
    _terminate_managed_ollama_process(proc)


def _offline_error(*, hint: str | None = None) -> EngineRoutingError:
    return EngineRoutingError(
        status_code=424,
        code="OLLAMA_UNAVAILABLE",
        message="OLLAMA_UNAVAILABLE: Local LLM engine is offline. Please start Ollama.",
        hint=hint,
    )


def _not_installed_error() -> EngineRoutingError:
    return EngineRoutingError(
        status_code=424,
        code="OLLAMA_NOT_INSTALLED",
        message=(
            "OLLAMA_NOT_INSTALLED: The `ollama` CLI was not found on PATH. "
            "Install from https://ollama.com and ensure `ollama` is available to the backend process."
        ),
        hint="After install, run `ollama serve` or set OLLAMA_AUTO_START=1 (non-Cloud Run) to auto-start.",
    )


def _start_failed_error() -> EngineRoutingError:
    return EngineRoutingError(
        status_code=424,
        code="OLLAMA_START_FAILED",
        message=(
            "OLLAMA_START_FAILED: Ollama did not become reachable after auto-start. "
            "Check OLLAMA_BASE_URL matches where `ollama serve` binds (default 127.0.0.1:11434)."
        ),
        hint="Run `ollama serve` manually in a terminal and confirm `curl $OLLAMA_BASE_URL/api/tags`.",
    )


async def _probe_tags(base_url: str, *, timeout: float) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.get(f"{base_url}/api/tags")


async def _wait_for_ollama_after_spawn(base_url: str) -> None:
    """Poll /api/tags after spawning ``ollama serve``."""
    probe_timeout = min(5.0, _timeout_seconds())
    for attempt in range(12):
        await asyncio.sleep(0.4)
        try:
            res = await _probe_tags(base_url, timeout=probe_timeout)
            if res.status_code < 400:
                logger.info("Ollama responded on /api/tags after auto-start (attempt %s).", attempt + 1)
                return
        except httpx.ConnectError:
            continue
    raise _start_failed_error()


async def ensure_ollama_available() -> None:
    """
    Preflight-check that local Ollama daemon is reachable.

    If connection fails and ``ollama`` is not on PATH, raises ``OLLAMA_NOT_INSTALLED``.
    If ``OLLAMA_AUTO_START`` is set (and Cloud Run guard allows), spawns ``ollama serve``
    and retries. Otherwise raises ``OLLAMA_UNAVAILABLE``.
    """
    base_url = _base_url()
    timeout_s = _timeout_seconds()
    try:
        res = await _probe_tags(base_url, timeout=timeout_s)
        if res.status_code >= 400:
            raise _offline_error(
                hint="Verify Ollama is running and listening on OLLAMA_BASE_URL.",
            )
        return
    except httpx.ConnectError as exc:
        exe = _ollama_executable()
        if exe is None:
            raise _not_installed_error() from exc
        if _should_attempt_auto_start():
            _spawn_ollama_serve()
            await _wait_for_ollama_after_spawn(base_url)
            return
        raise _offline_error(
            hint=(
                "Run `ollama serve` (or set OLLAMA_AUTO_START=1 on a non-Cloud Run host "
                "to try spawning `ollama serve` automatically). Ensure `llama3` is pulled: `ollama pull llama3`."
            ),
        ) from exc


async def generate_local_advice(tracking_data_csv: str) -> str:
    """
    Generate tactical advice from local Ollama using a fixed analyst persona.
    """
    await ensure_ollama_available()

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
