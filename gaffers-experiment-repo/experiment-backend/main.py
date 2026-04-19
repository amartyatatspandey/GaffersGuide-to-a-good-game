from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AdviceItem,
    AdviceResponse,
    ChatRequest,
    ChatResponse,
    ChunkingPolicy,
    CreateJobResponse,
    DecoderMode,
    EngineMode,
    HardwareProfile,
    QualityMode,
    ReportEntry,
    ReportsResponse,
    RuntimeTarget,
    SlaTier,
)
from services.job_store import ExperimentJob, ExperimentJobStore
from services.observability import MetricsRegistry
from services.paths import OUTPUT_ROOT, STORE_PATH, TASK_QUEUE_PATH, UPLOAD_ROOT
from services.queue import ExperimentQueue, QueueItem
from services.task_backend import TaskPayload
from services.task_backend_factory import build_task_backend

OUTPUT_DIR = OUTPUT_ROOT
UPLOAD_DIR = UPLOAD_ROOT

metrics = MetricsRegistry()
job_store = ExperimentJobStore(STORE_PATH)
queue = ExperimentQueue(job_store, metrics, OUTPUT_DIR)
LOGGER = logging.getLogger(__name__)


def ensure_runtime_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def preflight_check() -> None:
    ensure_runtime_directories()
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    if not ffmpeg_ok:
        raise RuntimeError("ffmpeg is required for streaming decode.")
    cloud_mode = os.getenv("EXP_CLOUD_MODE", "0") == "1"
    if cloud_mode and os.getenv("EXP_TASK_BACKEND", "redis").lower() != "redis":
        raise RuntimeError("Cloud mode requires redis task backend.")


task_backend = build_task_backend()

app = FastAPI(title="Gaffers Experiment API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    preflight_check()
    if os.getenv("EXP_INLINE_WORKER", "0") == "1":
        await queue.start()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/exp/jobs", response_model=CreateJobResponse)
async def create_job(
    file: UploadFile = File(...),
    cv_engine: EngineMode = Form("local"),
    llm_engine: EngineMode = Form("local"),
    decoder_mode: DecoderMode = Form("opencv"),
    runtime_target: RuntimeTarget = Form("nvidia"),
    hardware_profile: HardwareProfile = Form("l4"),
    quality_mode: QualityMode = Form("balanced"),
    chunking_policy: ChunkingPolicy = Form("fixed"),
    max_parallel_chunks: int = Form(2),
    target_sla_tier: SlaTier = Form("tier_10m"),
    homography_weights_dir: str | None = Form(default=None),
    idempotency_key: str | None = Form(default=None),
) -> CreateJobResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 uploads supported.")

    if idempotency_key:
        existing = job_store.find_by_idempotency(idempotency_key)
        if existing is not None:
            return CreateJobResponse(
                job_id=existing.job_id,
                status=existing.status,  # type: ignore[arg-type]
                cv_engine=existing.cv_engine,  # type: ignore[arg-type]
                llm_engine=existing.llm_engine,  # type: ignore[arg-type]
                decoder_mode=existing.decoder_mode,  # type: ignore[arg-type]
                runtime_target=existing.runtime_target,  # type: ignore[arg-type]
                hardware_profile=existing.hardware_profile,  # type: ignore[arg-type]
            )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{job_id}.mp4"
    try:
        with metrics.timed("exp.upload.write_ms"):
            with video_path.open("wb") as out_file:
                shutil.copyfileobj(file.file, out_file)
    finally:
        await file.close()

    now = datetime.now(timezone.utc).isoformat()
    job_store.create(
        ExperimentJob(
            job_id=job_id,
            status="pending",
            current_step="Pending",
            cv_engine=cv_engine,
            llm_engine=llm_engine,
            decoder_mode=decoder_mode,
            runtime_target=runtime_target,
            hardware_profile=hardware_profile,
            quality_mode=quality_mode,
            chunking_policy=chunking_policy,
            max_parallel_chunks=max(1, int(max_parallel_chunks)),
            target_sla_tier=target_sla_tier,
            stage="queued",
            source_video_path=str(video_path),
            source_video_uri=f"file://{video_path}",
            idempotency_key=idempotency_key,
            created_at=now,
            updated_at=now,
        )
    )
    payload = TaskPayload(
        job_id=job_id,
        video_path=video_path,
        cv_engine=cv_engine,
        llm_engine=llm_engine,
        decoder_mode=decoder_mode,
        runtime_target=runtime_target,
        hardware_profile=hardware_profile,
        quality_mode=quality_mode,
        chunking_policy=chunking_policy,
        max_parallel_chunks=max(1, int(max_parallel_chunks)),
        target_sla_tier=target_sla_tier,
        enqueued_at_epoch_ms=time.time() * 1000.0,
        homography_weights_dir=Path(homography_weights_dir).expanduser().resolve() if homography_weights_dir else None,
    )
    task_backend.enqueue(payload)
    if os.getenv("EXP_INLINE_WORKER", "0") == "1":
        await queue.enqueue(
            QueueItem(
                job_id=payload.job_id,
                video_path=payload.video_path,
                decoder_mode=payload.decoder_mode,
                cv_engine=payload.cv_engine,
                llm_engine=payload.llm_engine,
                runtime_target=payload.runtime_target,
                hardware_profile=payload.hardware_profile,
                quality_mode=payload.quality_mode,
                chunking_policy=payload.chunking_policy,
                max_parallel_chunks=payload.max_parallel_chunks,
                target_sla_tier=payload.target_sla_tier,
                enqueued_at_epoch_ms=payload.enqueued_at_epoch_ms,
                homography_weights_dir=payload.homography_weights_dir,
            )
        )
    return CreateJobResponse(
        job_id=job_id,
        status="pending",
        cv_engine=cv_engine,
        llm_engine=llm_engine,
        decoder_mode=decoder_mode,
        runtime_target=runtime_target,
        hardware_profile=hardware_profile,
    )


@app.get("/api/exp/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "current_step": job.current_step,
        "cv_engine": job.cv_engine,
        "llm_engine": job.llm_engine,
        "decoder_mode": job.decoder_mode,
        "runtime_target": job.runtime_target,
        "hardware_profile": job.hardware_profile,
        "quality_mode": job.quality_mode,
        "chunking_policy": job.chunking_policy,
        "max_parallel_chunks": job.max_parallel_chunks,
        "target_sla_tier": job.target_sla_tier,
        "stage": job.stage,
        "result_path": job.result_path,
        "result_uri": job.result_uri,
        "tracking_data_path": job.tracking_data_path,
        "tracking_data_uri": job.tracking_data_uri,
        "chunks": job.chunks,
        "telemetry": {
            "queue_wait_ms": job.queue_wait_ms,
            "decode_ms": job.decode_ms,
            "infer_ms": job.infer_ms,
            "post_ms": job.post_ms,
            "frames_processed": job.frames_processed,
            "effective_fps": job.effective_fps,
            "reid_invocations": job.reid_invocations,
            "reid_ms": job.reid_ms,
            "id_switch_rate": job.id_switch_rate,
            "frames_with_homography": job.frames_with_homography,
            "frames_without_homography": job.frames_without_homography,
            "fallback_frames": job.fallback_frames,
            "calibration_latency_ms": job.calibration_latency_ms,
        },
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@app.get("/api/exp/jobs/{job_id}/tracking")
async def get_tracking(job_id: str) -> dict[str, Any]:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.tracking_data_path:
        raise HTTPException(status_code=425, detail="Tracking not ready.")
    tracking_path = Path(job.tracking_data_path)
    if not tracking_path.is_file():
        raise HTTPException(status_code=425, detail="Tracking not ready.")
    if tracking_path.suffix == ".jsonl":
        frames = [json.loads(line) for line in tracking_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {"frames": frames}
    return json.loads(tracking_path.read_text(encoding="utf-8"))


@app.get("/api/exp/reports", response_model=ReportsResponse)
async def list_reports() -> ReportsResponse:
    reports: list[ReportEntry] = []
    for job in job_store.list():
        report_path = OUTPUT_DIR / f"{job.job_id}_report.json"
        if not report_path.is_file():
            continue
        reports.append(
            ReportEntry(
                job_id=job.job_id,
                created_at=job.created_at or datetime.now(timezone.utc).isoformat(),
                report_filename=report_path.name,
            )
        )
    reports.sort(key=lambda r: r.created_at, reverse=True)
    return ReportsResponse(reports=reports)


@app.post("/api/exp/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if req.job_id:
        job = job_store.get(req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
    reply = (
        "1. Keep compact vertical spacing.\n"
        "2. Rotate weak-side midfielder earlier.\n"
        "3. Trigger pressure when center-back receives facing own goal."
    )
    return ChatResponse(reply=reply)


@app.get("/api/exp/coach/advice", response_model=AdviceResponse)
async def coach_advice(
    job_id: str = Query(...),
) -> AdviceResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    report_path = OUTPUT_DIR / f"{job_id}_report.json"
    if not report_path.is_file():
        raise HTTPException(status_code=425, detail="Report not ready.")
    cards = json.loads(report_path.read_text(encoding="utf-8"))
    items: list[AdviceItem] = []
    for card in cards:
        text = str(card.get("tactical_instruction", ""))
        steps = [line.strip() for line in text.splitlines() if line.strip()]
        items.append(
            AdviceItem(
                frame_idx=int(card.get("frame_idx", 0)),
                team=str(card.get("team", "team_0")),
                flaw=str(card.get("flaw", "Signal detected")),
                severity=str(card.get("severity", "medium")),
                evidence=str(card.get("evidence", "")),
                matched_philosophy_author=str(card.get("matched_philosophy_author", "")),
                tactical_instruction=text,
                tactical_instruction_steps=steps,
            )
        )
    return AdviceResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        pipeline={"engine": "experiment", "llm": "templated"},
        advice_items=items,
    )


@app.get("/api/exp/metrics")
async def get_metrics() -> dict[str, object]:
    return metrics.snapshot()


@app.websocket("/ws/exp/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            job = job_store.get(job_id)
            if not job:
                await websocket.send_json(
                    {
                        "job_id": job_id,
                        "status": "error",
                        "current_step": "Unknown job",
                        "stage": "error",
                        "error": "job_not_found",
                    }
                )
                return
            await websocket.send_json(
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "current_step": job.current_step,
                    "stage": job.stage,
                    "runtime_target": job.runtime_target,
                    "hardware_profile": job.hardware_profile,
                    "result_path": job.result_path,
                    "result_uri": job.result_uri,
                    "tracking_data_path": job.tracking_data_path,
                    "tracking_data_uri": job.tracking_data_uri,
                    "chunks": job.chunks,
                    "telemetry": {
                        "queue_wait_ms": job.queue_wait_ms,
                        "decode_ms": job.decode_ms,
                        "infer_ms": job.infer_ms,
                        "post_ms": job.post_ms,
                        "frames_processed": job.frames_processed,
                        "effective_fps": job.effective_fps,
                        "reid_invocations": job.reid_invocations,
                        "reid_ms": job.reid_ms,
                        "id_switch_rate": job.id_switch_rate,
                        "frames_with_homography": job.frames_with_homography,
                        "frames_without_homography": job.frames_without_homography,
                        "fallback_frames": job.fallback_frames,
                        "calibration_latency_ms": job.calibration_latency_ms,
                    },
                    "error": job.error,
                }
            )
            if job.status in ("done", "error"):
                return
            await asyncio.sleep(0.5)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
