"""Tests for Ollama preflight, auto-start policy, and error codes."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services import ollama_client  # noqa: E402
from services.errors import EngineRoutingError  # noqa: E402


@pytest.fixture(autouse=True)
def _cleanup_managed_ollama() -> None:
    yield
    ollama_client.stop_ollama_for_app_lifecycle()


def test_ensure_ollama_connect_no_binary_raises_not_installed() -> None:
    async def _run() -> None:
        with (
            patch.object(
                ollama_client,
                "_probe_tags",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("refused"),
            ),
            patch.object(ollama_client, "_ollama_executable", return_value=None),
        ):
            try:
                await ollama_client.ensure_ollama_available()
            except EngineRoutingError as exc:
                assert exc.code == "OLLAMA_NOT_INSTALLED"
            else:
                raise AssertionError("expected EngineRoutingError")

    asyncio.run(_run())


def test_default_auto_start_when_env_unset_and_not_cloud() -> None:
    """Unset OLLAMA_AUTO_START on a non-Cloud host should still attempt spawn on connect failure."""

    async def _run() -> None:
        mock_popen = MagicMock()
        ok_response = MagicMock()
        ok_response.status_code = 200

        async def _probe_side_effect(*_a: object, **_k: object) -> MagicMock:
            _probe_side_effect.calls += 1  # type: ignore[attr-defined]
            if _probe_side_effect.calls < 2:
                raise httpx.ConnectError("refused")
            return ok_response

        _probe_side_effect.calls = 0  # type: ignore[attr-defined]

        with (
            patch.dict(
                "os.environ",
                {"K_SERVICE": "", "OLLAMA_AUTO_START": ""},
                clear=False,
            ),
            patch.object(ollama_client, "_probe_tags", side_effect=_probe_side_effect),
            patch.object(
                ollama_client, "_ollama_executable", return_value="/usr/bin/ollama"
            ),
            patch("subprocess.Popen", mock_popen),
            patch.object(ollama_client.asyncio, "sleep", new_callable=AsyncMock),
        ):
            await ollama_client.ensure_ollama_available()

        mock_popen.assert_called_once()

    asyncio.run(_run())


def test_default_auto_start_off_on_cloud_when_env_unset() -> None:
    async def _run() -> None:
        mock_popen = MagicMock()
        with (
            patch.dict(
                "os.environ",
                {
                    "K_SERVICE": "svc",
                    "OLLAMA_AUTO_START": "",
                    "OLLAMA_AUTO_START_IN_CLOUD": "",
                },
                clear=False,
            ),
            patch.object(
                ollama_client,
                "_probe_tags",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("refused"),
            ),
            patch.object(
                ollama_client, "_ollama_executable", return_value="/usr/bin/ollama"
            ),
            patch("subprocess.Popen", mock_popen),
        ):
            try:
                await ollama_client.ensure_ollama_available()
            except EngineRoutingError as exc:
                assert exc.code == "OLLAMA_UNAVAILABLE"
            else:
                raise AssertionError("expected EngineRoutingError")
        mock_popen.assert_not_called()

    asyncio.run(_run())


def test_ensure_ollama_cloud_run_no_spawn_without_in_cloud_flag() -> None:
    async def _run() -> None:
        mock_popen = MagicMock()

        with (
            patch.dict(
                "os.environ",
                {
                    "OLLAMA_AUTO_START": "1",
                    "K_SERVICE": "myservice",
                    "OLLAMA_AUTO_START_IN_CLOUD": "",
                },
                clear=False,
            ),
            patch.object(
                ollama_client,
                "_probe_tags",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("refused"),
            ),
            patch.object(
                ollama_client, "_ollama_executable", return_value="/usr/bin/ollama"
            ),
            patch("subprocess.Popen", mock_popen),
        ):
            try:
                await ollama_client.ensure_ollama_available()
            except EngineRoutingError as exc:
                assert exc.code == "OLLAMA_UNAVAILABLE"
            else:
                raise AssertionError("expected EngineRoutingError")
        mock_popen.assert_not_called()

    asyncio.run(_run())


def test_ensure_ollama_cloud_run_spawns_when_in_cloud_flag() -> None:
    async def _run() -> None:
        mock_popen = MagicMock()
        ok_response = MagicMock()
        ok_response.status_code = 200

        async def _probe_side_effect(*_a: object, **_k: object) -> MagicMock:
            _probe_side_effect.calls += 1  # type: ignore[attr-defined]
            if _probe_side_effect.calls < 2:
                raise httpx.ConnectError("refused")
            return ok_response

        _probe_side_effect.calls = 0  # type: ignore[attr-defined]

        with (
            patch.dict(
                "os.environ",
                {
                    "OLLAMA_AUTO_START": "1",
                    "K_SERVICE": "myservice",
                    "OLLAMA_AUTO_START_IN_CLOUD": "1",
                },
                clear=False,
            ),
            patch.object(ollama_client, "_probe_tags", side_effect=_probe_side_effect),
            patch.object(
                ollama_client, "_ollama_executable", return_value="/usr/bin/ollama"
            ),
            patch("subprocess.Popen", mock_popen),
            patch.object(ollama_client.asyncio, "sleep", new_callable=AsyncMock),
        ):
            await ollama_client.ensure_ollama_available()

        mock_popen.assert_called_once()

    asyncio.run(_run())


def test_ensure_ollama_auto_start_spawns_and_retries() -> None:
    async def _run() -> None:
        mock_popen = MagicMock()
        ok_response = MagicMock()
        ok_response.status_code = 200

        async def _probe_side_effect(*_a: object, **_k: object) -> MagicMock:
            _probe_side_effect.calls += 1  # type: ignore[attr-defined]
            if _probe_side_effect.calls < 3:  # type: ignore[attr-defined]
                raise httpx.ConnectError("refused")
            return ok_response

        _probe_side_effect.calls = 0  # type: ignore[attr-defined]

        with (
            patch.dict(
                "os.environ", {"OLLAMA_AUTO_START": "1", "K_SERVICE": ""}, clear=False
            ),
            patch.object(ollama_client, "_probe_tags", side_effect=_probe_side_effect),
            patch.object(
                ollama_client,
                "_ollama_executable",
                return_value="/opt/homebrew/bin/ollama",
            ),
            patch("subprocess.Popen", mock_popen),
            patch.object(ollama_client.asyncio, "sleep", new_callable=AsyncMock),
        ):
            await ollama_client.ensure_ollama_available()

        mock_popen.assert_called_once()
        assert _probe_side_effect.calls >= 3  # type: ignore[attr-defined]

    asyncio.run(_run())


def test_lifecycle_start_noop_when_tags_already_ok() -> None:
    async def _run() -> None:
        ok = MagicMock()
        ok.status_code = 200
        with (
            patch.dict(
                os.environ,
                {"OLLAMA_MANAGED_LIFECYCLE": "1", "K_SERVICE": ""},
                clear=False,
            ),
            patch.object(
                ollama_client, "_probe_tags", new_callable=AsyncMock, return_value=ok
            ),
        ):
            await ollama_client.start_ollama_for_app_lifecycle()
        assert ollama_client._lifecycle_popen is None

    asyncio.run(_run())


def test_lifecycle_stop_terminates_tracked_child() -> None:
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 9999
    ollama_client._lifecycle_popen = mock_proc
    ollama_client.stop_ollama_for_app_lifecycle()
    mock_proc.terminate.assert_called_once()
    assert ollama_client._lifecycle_popen is None


def test_ensure_ollama_tags_http_error_offline() -> None:
    async def _run() -> None:
        bad = MagicMock()
        bad.status_code = 500
        with patch.object(
            ollama_client, "_probe_tags", new_callable=AsyncMock, return_value=bad
        ):
            try:
                await ollama_client.ensure_ollama_available()
            except EngineRoutingError as exc:
                assert exc.code == "OLLAMA_UNAVAILABLE"
            else:
                raise AssertionError("expected error")

    asyncio.run(_run())
