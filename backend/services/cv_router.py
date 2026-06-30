from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol

import httpx

from services.errors import EngineRoutingError
from services.llm_router import LLMEngine

LOGGER = logging.getLogger(__name__)

CVEngine = Literal["local", "cloud"]


class CVRunner(Protocol):
    async def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        gcs_blob_name: str = "",
        progress_callback: Callable[[str], None] | None = None,
        llm_engine: LLMEngine | None = None,
    ) -> Path: ...


class LocalCVRunner:
    async def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        gcs_blob_name: str = "",
        progress_callback: Callable[[str], None] | None = None,
        llm_engine: LLMEngine | None = None,
    ) -> Path:
        engine: LLMEngine = llm_engine if llm_engine is not None else "cloud"
        enable_zsl = os.getenv("ENABLE_ZSL", "false").lower() == "true"
        enable_parallel = os.getenv("USE_PARALLEL", "true").lower() == "true"

        # ── GCS download if local file is absent ──────────────────────
        # This happens when the upload was handled by a different Cloud Run
        # instance (or when the local /tmp file was cleaned up after GCS upload).
        local_video_was_downloaded = False
        if not video_path.is_file() and gcs_blob_name:
            try:
                from services import gcs_service  # local import avoids circular deps
                tmp_path = Path("/tmp") / f"{job_id}{video_path.suffix or '.mp4'}"
                LOGGER.info(
                    "Local video not found — downloading from GCS: %s → %s",
                    gcs_blob_name, tmp_path,
                )
                gcs_service.download_file(gcs_blob_name, tmp_path)
                video_path = tmp_path
                local_video_was_downloaded = True
            except Exception as gcs_err:  # noqa: BLE001
                raise EngineRoutingError(
                    status_code=503,
                    code="GCS_DOWNLOAD_FAILED",
                    message=f"Cannot retrieve video from GCS ({gcs_blob_name}): {gcs_err}",
                ) from gcs_err

        try:
            if enable_parallel:
                from services.parallel_pipeline import run_e2e_parallel
                return await run_e2e_parallel(
                    video_path,
                    output_prefix=job_id,
                    progress_callback=progress_callback,
                    llm_engine=engine,
                    enable_zsl=enable_zsl,
                )

            def _run() -> Path:
                from scripts.run_e2e_zsl import run_e2e_with_zsl
                return run_e2e_with_zsl(
                    video_path,
                    output_prefix=job_id,
                    progress_callback=progress_callback,
                    llm_engine=engine,
                    enable_zsl=enable_zsl,
                )

            return await asyncio.to_thread(_run)

        finally:
            # ── Cleanup local temp video after processing ─────────────
            # Only delete files we downloaded from GCS (not files the user
            # deliberately placed on disk for local dev runs).
            if local_video_was_downloaded and video_path.exists():
                video_path.unlink(missing_ok=True)
                LOGGER.debug("Cleaned up downloaded temp video: %s", video_path)


class CloudCVRunner:
    async def run(
        self,
        *,
        job_id: str,
        video_path: Path,
        gcs_blob_name: str = "",
        progress_callback: Callable[[str], None] | None = None,
        llm_engine: LLMEngine | None = None,
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

