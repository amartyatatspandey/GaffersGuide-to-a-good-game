from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Callable, Literal, Protocol

import httpx

from services.errors import EngineRoutingError

CVEngine = Literal["local", "cloud"]


class CVRunner(Protocol):
    async def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path: ...


class LocalCVRunner:
    async def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path:
        # Lazy import keeps cold-start light for API-only flows.
        from scripts.run_e2e import run_e2e

        return await asyncio.to_thread(
            run_e2e,
            video_path,
            output_prefix=job_id,
            progress_callback=progress_callback,
        )


class CloudCVRunner:
    async def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path:
        webhook_url = os.getenv("MODAL_WEBHOOK_URL", "").strip()
        if not webhook_url:
            raise EngineRoutingError(
                status_code=424,
                code="CLOUD_CV_NOT_CONFIGURED",
                message="Cloud CV selected but MODAL_WEBHOOK_URL is not configured.",
                hint="Set MODAL_WEBHOOK_URL in environment variables.",
            )

        if progress_callback:
            progress_callback("Tracking Players")

        headers: dict[str, str] = {}
        api_key = os.getenv("MODAL_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout_s = float(os.getenv("MODAL_TIMEOUT_SECONDS", "600"))
        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}_report.json"

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            try:
                with video_path.open("rb") as f:
                    files = {"file": (video_path.name, f, "video/mp4")}
                    data = {"job_id": job_id}
                    res = await client.post(
                        webhook_url, headers=headers, files=files, data=data
                    )

                if res.status_code >= 400:
                    raise EngineRoutingError(
                        status_code=503,
                        code="CLOUD_CV_UPSTREAM_ERROR",
                        message=f"Modal webhook error {res.status_code}: {res.text}",
                    )

                payload = res.json()
                # Two accepted contracts:
                # 1) {"report_path": "/abs/path.json"}
                # 2) {"report_cards": [...]}
                report_path = payload.get("report_path")
                report_cards = payload.get("report_cards")

                if isinstance(report_path, str) and report_path:
                    p = Path(report_path)
                    if not p.is_file():
                        raise EngineRoutingError(
                            status_code=503,
                            code="CLOUD_CV_BAD_RESPONSE",
                            message="Modal response returned report_path that does not exist.",
                        )
                    if progress_callback:
                        progress_callback("Completed")
                    return p

                if isinstance(report_cards, list):
                    with output_path.open("w", encoding="utf-8") as f_out:
                        json.dump(report_cards, f_out, indent=2, ensure_ascii=False)
                    if progress_callback:
                        progress_callback("Completed")
                    return output_path

                raise EngineRoutingError(
                    status_code=503,
                    code="CLOUD_CV_BAD_RESPONSE",
                    message="Modal response missing both report_path and report_cards.",
                )
            except EngineRoutingError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise EngineRoutingError(
                    status_code=503,
                    code="CLOUD_CV_UPSTREAM_ERROR",
                    message=str(exc),
                ) from exc


class CVRouterFactory:
    @staticmethod
    def get(engine: CVEngine) -> CVRunner:
        if engine == "local":
            return LocalCVRunner()
        if engine == "cloud":
            return CloudCVRunner()
        raise EngineRoutingError(
            status_code=409,
            code="ENGINE_MODE_UNSUPPORTED",
            message=f"Unsupported cv_engine value: {engine}",
        )

